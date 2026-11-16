"""End-to-end smoke test on a tiny synthetic dataset (a few minutes on CPU).

Runs prepare -> DAMSM -> train (1 epoch) for every model config (ablations on top of
gamf.yaml), then sample -> evaluate (SSIM) for GAMF-GAN. No downloads, no real data needed.
Run it on the GPU machine before launching the real experiments:

    python tests/smoke_test.py [--keep] [--models gamf dfgan]
"""
import argparse
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
WORDS = "this bird flower has small large red blue yellow white black wings petals beak long short".split()


def make_raw(raw):
    rng = random.Random(0)
    for cls in ("c1", "c2", "c3"):
        (raw / "images" / cls).mkdir(parents=True)
        (raw / "text_c10" / cls).mkdir(parents=True)
        for i in range(8):
            name = f"{cls}_img{i}"
            noise = np.random.default_rng(rng.randrange(10 ** 9)).integers(0, 256, (35, 38, 3), dtype=np.uint8)
            Image.fromarray(noise).resize((300, 280), Image.BILINEAR).save(raw / "images" / cls / f"{name}.jpg")
            caps = [" ".join(rng.choices(WORDS, k=rng.randint(5, 22))) + "." for _ in range(10)]
            (raw / "text_c10" / cls / f"{name}.txt").write_text("\n".join(caps), encoding="utf-8")
    # deliberate leak: a resized, re-encoded copy of a training-class image placed in the test class
    with Image.open(raw / "images" / "c1" / "c1_img0.jpg") as im:
        im.resize((320, 300), Image.BICUBIC).save(raw / "images" / "c3" / "c3_leak.jpg", quality=80)
    (raw / "text_c10" / "c3" / "c3_leak.txt").write_text(
        (raw / "text_c10" / "c1" / "c1_img0.txt").read_text(encoding="utf-8"), encoding="utf-8")
    (raw / "trainvalclasses.txt").write_text("c1\nc2\n")
    (raw / "testclasses.txt").write_text("c3\n")


def check_leakage(tmp, base, common):
    """The planted near-duplicate must be rejected, the audit must pass, and training
    code must be unable to open the test split."""
    import json
    sys.path.insert(0, str(ROOT))
    from gamf.config import load_config
    from gamf.data.dataset import T2IDataset
    meta = json.loads((tmp / "data" / "toy" / "split.json").read_text(encoding="utf-8"))
    test_ids = {r["image_id"] for r in meta["splits"]["test"]}
    train_ids = {r["image_id"] for r in meta["splits"]["train"] + meta["splits"]["val"]}
    assert "c1_img0" in train_ids
    assert meta["leakage_check"]["passed"], meta["leakage_check"]
    assert "c3_leak" not in test_ids, "planted near-duplicate reached the test split"
    # the test split takes the whole test pool except one image, so the planted copy must have
    # been examined and rejected rather than merely not drawn
    assert meta["rejected"]["test_pool"].get("near_duplicate_of_train_or_val", 0) >= 1
    # the audit itself must also detect the copy if it is forced into the test split
    from gamf.data.leakage import audit
    raw = Path(meta["raw_root"])
    forced = json.loads(json.dumps(meta))
    forced["splits"]["test"].append({"image_id": "c3_leak", "path": "images/c3/c3_leak.jpg",
                                     "class_id": "c3", "captions": ["x"]})
    rep = audit(forced, 4)
    assert not rep["passed"] and rep["checks"]["train-test"]["near_duplicates"] >= 1, rep
    cfg = load_config([base[1]], common)
    try:
        T2IDataset(cfg, "test", train=False)
        raise AssertionError("test split opened without allow_test")
    except PermissionError:
        pass
    print(f"LEAKAGE CHECK OK: planted duplicate rejected; test-pool rejections "
          f"{meta['rejected']['test_pool']}; audit {meta['leakage_check']['checks']}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--keep", action="store_true")
    p.add_argument("--models", nargs="*")
    args = p.parse_args()
    tmp = Path(tempfile.mkdtemp(prefix="gamf_smoke_"))
    make_raw(tmp / "raw")
    common = [
        f"paths.data_root={tmp.as_posix()}/data", f"paths.runs_root={tmp.as_posix()}/runs",
        "data.name=toy", "data.format=reed", f"data.raw_root={tmp.as_posix()}/raw",
        "data.image_dir=images", "data.split_seed=0",
        "data.subset={'train': 14, 'val': 2, 'test': 8}", "data.captions_per_image=10",
        "data.min_side=256", "data.num_workers=0",
        "damsm.epochs=1", "damsm.batch_size=4", "damsm.pretrained_backbone=False",
        "train.epochs=1", "train.stage_epochs=[1, 1, 1]", "train.batch_size=4", "train.amp=none", "train.eval_every=1",
        "train.log_every=1", "loss.vgg_pretrained=False",
        "eval.samples_per_caption=2", "eval.batch_size=4",
    ]
    base = ["--config", ROOT / "configs" / "base.yaml"]

    def py(module, *a):
        cmd = [sys.executable, "-m", module, *[str(x) for x in a]]
        print(">>", module, flush=True)
        subprocess.run(cmd, check=True, cwd=ROOT)

    py("gamf.data.prepare", *base, "--set", *common)
    check_leakage(tmp, base, common)
    py("gamf.train_damsm", *base, "--set", *common)
    models = args.models or sorted(f.stem for f in (ROOT / "configs" / "models").glob("*.yaml"))
    for m in models:
        cfgs = [ROOT / "configs" / "models" / f"{m}.yaml"]
        if m.startswith("gamf_"):
            cfgs.insert(0, ROOT / "configs" / "models" / "gamf.yaml")
        py("gamf.train", *base, *cfgs, "--set", *common)
    run = tmp / "runs" / "toy" / "gamf" / "seed_0"
    if (run / "ckpt" / "best.pt").exists():
        py("gamf.sample", "--run", run)
        py("gamf.evaluate", *base, ROOT / "configs" / "models" / "gamf.yaml", "--set", *common,
           "--gen-dir", run / "samples" / "test", "--out", run / "metrics.json", "--metrics", "ssim")
        print((run / "metrics.json").read_text())
    print(f"SMOKE TEST PASSED ({len(models)} model configs)")
    if args.keep:
        print(f"outputs kept in {tmp}")
    else:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
