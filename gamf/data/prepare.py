"""Build reproducible dataset subsets and splits (reviewer comments R2-3, R3-3).

For each dataset this script writes to ``<data_root>/<name>/``:

* ``split.json``      -- the exact train/val/test records (image path, all
                         captions, class, fixed evaluation caption index),
                         the sampling seed, filters and rejection counts;
* ``vocab.json``      -- vocabulary built from the *training* captions only;
* ``captions_<split>.csv`` -- evaluation captions for val/test (also used to
                         run external models such as DF-GAN or GALIP on the
                         very same captions);
* ``real_256/<split>/`` -- val/test reference images, resized (shorter side
                         256) and centre-cropped to 256x256, used by FID/KID/IS
                         and as the paired reference for SSIM/LPIPS.

Split policy
------------
CUB-200-2011 / Oxford-102: class-disjoint split of Reed et al. (2016), as in
AttnGAN/DM-GAN: train+val images come from the train classes, test images from
the unseen test classes. MS COCO: train+val from train2014, test from val2014.
The test pool is never used for training or model selection. Test candidates
that duplicate or near-duplicate (dHash) a training/validation image are rejected,
and the written split is audited by gamf.data.leakage before any training.

Usage::

    python -m gamf.data.prepare --config configs/base.yaml configs/datasets/cub.yaml
"""
import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from gamf.config import add_config_args, dataset_dir, load_config
from gamf.data.leakage import audit, dhash, md5, min_hamming
from gamf.data.text import tokenize
from gamf.utils import mark_done, save_image_atomic, write_json


def _lines(path):
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    return [line.strip() for line in text.splitlines() if line.strip()]


def load_reed(raw, image_dir, text_dir, train_classes, test_classes):
    """CUB / Oxford-102 with the captions of Reed et al. (text_c10)."""
    train_cls = set(_lines(raw / train_classes))
    test_cls = set(_lines(raw / test_classes))
    images = {p.stem: p for p in (raw / image_dir).rglob("*.jpg")}
    records = []
    for cls_dir in sorted((raw / text_dir).iterdir()):
        if not cls_dir.is_dir():
            continue
        pool = ("train" if cls_dir.name in train_cls
                else "test" if cls_dir.name in test_cls else None)
        if pool is None:
            continue
        for txt in sorted(cls_dir.glob("*.txt")):
            if txt.stem in images:
                records.append({
                    "image_id": txt.stem,
                    "path": images[txt.stem].relative_to(raw).as_posix(),
                    "class_id": cls_dir.name,
                    "captions": _lines(txt),
                    "pool": pool,
                })
    if not records:
        raise SystemExit(f"No records found under {raw}; check data.* paths in the config.")
    return records


def load_coco(raw):
    records = []
    for folder, pool in (("train2014", "train"), ("val2014", "test")):
        with open(raw / "annotations" / f"captions_{folder}.json", encoding="utf-8") as f:
            ann = json.load(f)
        captions = defaultdict(list)
        for a in sorted(ann["annotations"], key=lambda a: a["id"]):
            captions[a["image_id"]].append(a["caption"].strip())
        for im in sorted(ann["images"], key=lambda im: im["id"]):
            records.append({
                "image_id": str(im["id"]),
                "path": f"{folder}/{im['file_name']}",
                "class_id": None,
                "captions": captions[im["id"]],
                "pool": pool,
            })
    return records


def _check_image(path, min_side):
    try:
        with Image.open(path) as im:
            if im.mode != "RGB":
                return "not_rgb"
            if min(im.size) < min_side:
                return "too_small"
    except OSError:
        return "unreadable"
    return None


def _sample(pool, n, rng, raw, min_side, n_caps, forbid=None, max_hamming=4):
    """Draw n valid records. Every chosen image is fingerprinted (MD5, dHash); a candidate
    is rejected if it duplicates an already chosen image or, when ``forbid`` is given
    (fingerprints of the training/validation images), if it is an exact or near
    duplicate of one of them -- the replacement is drawn from the remaining pool."""
    pool = list(pool)
    rng.shuffle(pool)
    chosen, rejected = [], Counter()
    seen_md5 = set(forbid["md5"]) if forbid else set()
    forbid_dh = list(forbid["dhash"]) if forbid else []
    for rec in pool:
        if len(chosen) == n:
            break
        if len(rec["captions"]) < n_caps:
            rejected["too_few_captions"] += 1
            continue
        path = raw / rec["path"]
        why = _check_image(path, min_side)
        if why:
            rejected[why] += 1
            continue
        h_md5, h_d = md5(path), dhash(path)
        if h_md5 in seen_md5:
            rejected["exact_duplicate"] += 1
            continue
        if forbid_dh and min_hamming(h_d, forbid_dh) <= max_hamming:
            rejected["near_duplicate_of_train_or_val"] += 1
            continue
        seen_md5.add(h_md5)
        chosen.append(dict(rec, captions=rec["captions"][:n_caps], md5=h_md5, dhash=h_d))
    if len(chosen) < n:
        raise SystemExit(f"Only {len(chosen)} valid images available, {n} requested.")
    return chosen, rejected


def resize_center_crop(img, size):
    w, h = img.size
    scale = size / min(w, h)
    img = img.resize((max(size, round(w * scale)), max(size, round(h * scale))),
                     Image.BICUBIC)
    w, h = img.size
    left, top = (w - size) // 2, (h - size) // 2
    return img.crop((left, top, left + size, top + size))


def main():
    args = add_config_args(argparse.ArgumentParser(description=__doc__)).parse_args()
    cfg = load_config(args.config, args.set)
    d = cfg["data"]
    raw = Path(d["raw_root"])
    out = dataset_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "prepare_done.json").exists():
        print(f"{out} already prepared")
        return
    # Interrupted before completion: the split is re-drawn deterministically (same seed),
    # reference images already written atomically are reused.
    rng = random.Random(d["split_seed"])

    if d["format"] == "reed":
        records = load_reed(raw, d["image_dir"], d.get("text_dir", "text_c10"),
                            d.get("train_classes", "trainvalclasses.txt"),
                            d.get("test_classes", "testclasses.txt"))
    elif d["format"] == "coco":
        records = load_coco(raw)
    else:
        raise SystemExit(f"Unknown data.format {d['format']!r}")

    sizes = d["subset"]
    train_pool = [r for r in records if r["pool"] == "train"]
    test_pool = [r for r in records if r["pool"] == "test"]
    trval, rej_train = _sample(train_pool, sizes["train"] + sizes["val"], rng, raw,
                               d["min_side"], d["captions_per_image"])
    max_ham = d.get("dup_hamming", 4)
    forbid = {"md5": {r["md5"] for r in trval}, "dhash": [r["dhash"] for r in trval]}
    test, rej_test = _sample(test_pool, sizes["test"], rng, raw, d["min_side"],
                             d["captions_per_image"], forbid=forbid, max_hamming=max_ham)
    splits = {"train": trval[:sizes["train"]], "val": trval[sizes["train"]:], "test": test}
    for name in ("val", "test"):
        for rec in splits[name]:
            rec["eval_caption_idx"] = rng.randrange(len(rec["captions"]))
    for recs in splits.values():
        for rec in recs:
            rec.pop("pool", None)

    counts = Counter(t for rec in splits["train"] for c in rec["captions"] for t in tokenize(c))
    words = [w for w, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
             if n >= d.get("min_word_freq", 1)]
    word2idx = {"<pad>": 0, "<unk>": 1}
    word2idx.update({w: i + 2 for i, w in enumerate(words)})

    meta = {
        "name": d["name"], "raw_root": str(raw.resolve()), "split_seed": d["split_seed"],
        "filters": {"min_side": d["min_side"], "rgb_only": True,
                    "captions_per_image": d["captions_per_image"]},
        "pool_sizes": {"train": len(train_pool), "test": len(test_pool)},
        "rejected": {"train_pool": dict(rej_train), "test_pool": dict(rej_test)},
        "counts": {k: len(v) for k, v in splits.items()},
        "vocabulary_source": "training captions only",
        "splits": splits,
    }
    meta["leakage_check"] = audit(meta, max_ham)
    write_json(meta, out / "split.json")
    if not meta["leakage_check"]["passed"]:
        raise SystemExit(f"Leakage detected: {meta['leakage_check']['checks']}")
    write_json({"word2idx": word2idx}, out / "vocab.json")

    for name in ("val", "test"):
        real_dir = out / "real_256" / name
        real_dir.mkdir(parents=True, exist_ok=True)
        with open(out / f"captions_{name}.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["image_id", "caption_idx", "caption", "real_path"])
            for rec in splits[name]:
                real = real_dir / f"{rec['image_id']}.png"
                if not real.exists():
                    with Image.open(raw / rec["path"]) as im:
                        save_image_atomic(resize_center_crop(im.convert("RGB"), 256), real)
                for ci, cap in enumerate(rec["captions"]):
                    writer.writerow([rec["image_id"], ci, cap, real.as_posix()])

    mark_done(out / "prepare_done.json", counts={k: len(v) for k, v in splits.items()})
    print(f"{d['name']}: " + ", ".join(f"{k}={len(v)}" for k, v in splits.items())
          + f", vocab={len(word2idx)}")


if __name__ == "__main__":
    main()
