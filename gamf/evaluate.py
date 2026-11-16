"""Evaluate a directory of generated images on a held-out split.

Works for any model whose files are named ``<image_id>_c<caption_idx>_s<k>.png``
and were generated from ``captions_<split>.csv`` (GAMF-GAN, re-implemented
baselines, or released checkpoints of external models).

    python -m gamf.evaluate --config configs/base.yaml configs/datasets/cub.yaml \
        --gen-dir runs/cub/gamf/seed_0/samples/test --out runs/cub/gamf/seed_0/metrics.json
"""
import argparse
import csv
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor

from gamf import metrics as M
from gamf.config import add_config_args, damsm_path, dataset_dir, load_config
from gamf.utils import get_device, hardware_info, write_json

NAME = re.compile(r"^(?P<iid>.+)_c(?P<ci>\d+)_s(?P<k>\d+)\.png$")
ALL = ("fidelity", "ssim", "lpips", "lpips_div", "clip", "clip_rprec", "damsm_rprec")


def _load(path, size=256):
    with Image.open(path) as im:
        im = im.convert("RGB")
        if im.size != (size, size):
            im = im.resize((size, size), Image.BICUBIC)
        return to_tensor(im)  # [0, 1]


def main():
    p = add_config_args(argparse.ArgumentParser(description=__doc__))
    p.add_argument("--gen-dir", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--out", required=True)
    p.add_argument("--metrics", nargs="+", default=list(ALL), choices=ALL)
    args = p.parse_args()
    cfg = load_config(args.config, args.set)
    ecfg = cfg["eval"]
    device = get_device()
    root = dataset_dir(cfg)
    gen_dir = Path(args.gen_dir)
    real_dir = root / "real_256" / args.split

    captions = {}
    with open(root / f"captions_{args.split}.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            captions[(r["image_id"], int(r["caption_idx"]))] = r
    rows = []
    for fp in sorted(gen_dir.glob("*.png")):
        m = NAME.match(fp.name)
        if not m:
            raise SystemExit(f"Unexpected file name: {fp.name}")
        key = (m["iid"], int(m["ci"]))
        if key not in captions:
            raise SystemExit(f"{fp.name}: caption not in captions_{args.split}.csv")
        rows.append({"file": fp, "image_id": key[0], "caption_idx": key[1], "sample": int(m["k"]),
                     "caption": captions[key]["caption"], "real": captions[key]["real_path"]})
    if not rows:
        raise SystemExit(f"No images in {gen_dir}")

    res = {"split": args.split, "n_images": len(rows), "gen_dir": str(gen_dir),
           "n_captions": len({(r["image_id"], r["caption_idx"]) for r in rows}),
           "samples_per_caption": max(r["sample"] for r in rows) + 1,
           "n_reference": len(list(real_dir.glob("*.png"))),
           "eval_seed": ecfg["seed"], "r_precision_candidates": ecfg["r_precision_candidates"],
           "protocol": {"ssim": "Gaussian 11/1.5, K1=0.01, K2=0.03, RGB 256px",
                        "reference": "caption's source image, resize-256 + centre crop"},
           "hardware": hardware_info()}
    per = {r["file"].name: {"image_id": r["image_id"], "caption_idx": r["caption_idx"]} for r in rows}
    bs = ecfg["batch_size"]

    if "fidelity" in args.metrics:
        res.update(M.fidelity_metrics(gen_dir, real_dir, device))

    if "ssim" in args.metrics or "lpips" in args.metrics:
        lp = M.LPIPSMetric(device, ecfg["lpips_net"]) if "lpips" in args.metrics else None
        for i in range(0, len(rows), bs):
            chunk = rows[i:i + bs]
            g = torch.stack([_load(r["file"]) for r in chunk]).to(device)
            t = torch.stack([_load(r["real"]) for r in chunk]).to(device)
            s = M.ssim(g, t).cpu().numpy() if "ssim" in args.metrics else None
            l = lp(g * 2 - 1, t * 2 - 1).cpu().numpy() if lp else None
            for j, r in enumerate(chunk):
                if s is not None:
                    per[r["file"].name]["ssim"] = float(s[j])
                if l is not None:
                    per[r["file"].name]["lpips"] = float(l[j])
        if "ssim" in args.metrics:
            res["SSIM"] = float(np.mean([v["ssim"] for v in per.values()]))
        if lp:
            res["LPIPS"] = float(np.mean([v["lpips"] for v in per.values()]))
            res["lpips_impl"] = lp.version

    if "lpips_div" in args.metrics:
        lp = M.LPIPSMetric(device, ecfg["lpips_net"])
        res["LPIPS_div"] = M.lpips_diversity(M.group_by_caption(rows),
                                             lambda f: _load(f) * 2 - 1, lp, device)

    if "clip" in args.metrics or "clip_rprec" in args.metrics:
        clip = M.CLIPMetric(device, ecfg["clip_model"])
        texts = sorted({r["caption"] for r in rows} | {c["caption"] for c in captions.values()})
        t_index = {t: i for i, t in enumerate(texts)}
        tf = torch.cat([clip.text_features(texts[i:i + 256]) for i in range(0, len(texts), 256)])
        img_f = []
        for i in range(0, len(rows), bs):
            ims = [Image.open(r["file"]).convert("RGB") for r in rows[i:i + bs]]
            img_f.append(clip.image_features(ims))
        img_f = torch.cat(img_f)
        true = torch.tensor([t_index[r["caption"]] for r in rows])
        cos = (img_f * tf[true.to(tf.device)]).sum(-1).cpu().numpy()
        for j, r in enumerate(rows):
            per[r["file"].name]["clip_cos"] = float(cos[j])
        res["CLIP_cos"] = float(cos.mean())
        res["CLIPScore"] = float(100 * np.clip(cos, 0, None).mean())
        res["clip_model"] = clip.model_id
        if "clip_rprec" in args.metrics:
            res["CLIP_R@1"] = M.r_precision(img_f, tf, true, ecfg["r_precision_candidates"], ecfg["seed"])

    if "damsm_rprec" in args.metrics:
        from gamf.data.dataset import T2IDataset
        from gamf.models.text_encoder import load_damsm
        text_enc, img_enc = load_damsm(damsm_path(cfg), device)
        ds = T2IDataset(cfg, args.split, train=False, allow_test=True)
        texts = sorted({c["caption"] for c in captions.values()})
        t_index = {t: i for i, t in enumerate(texts)}
        sent = []
        with torch.no_grad():
            for i in range(0, len(texts), 256):
                enc = [ds.encode_caption(t) for t in texts[i:i + 256]]
                _, s = text_enc(torch.stack([e[0] for e in enc]).to(device),
                                torch.tensor([e[1] for e in enc]))
                sent.append(torch.nn.functional.normalize(s, dim=-1))
            sent = torch.cat(sent)
            glob = []
            for i in range(0, len(rows), bs):
                g = torch.stack([_load(r["file"]) for r in rows[i:i + bs]]).to(device) * 2 - 1
                glob.append(torch.nn.functional.normalize(img_enc(g)[1], dim=-1))
            glob = torch.cat(glob)
        true = torch.tensor([t_index[r["caption"]] for r in rows])
        res["DAMSM_R@1"] = M.r_precision(glob, sent, true, ecfg["r_precision_candidates"], ecfg["seed"])

    with open(Path(args.out).with_suffix(".per_image.csv"), "w", newline="", encoding="utf-8") as f:
        keys = sorted({k for v in per.values() for k in v})
        w = csv.writer(f)
        w.writerow(["file"] + keys)
        for name, v in per.items():
            w.writerow([name] + [v.get(k, "") for k in keys])
    write_json(res, args.out)  # written last and atomically: its presence marks completion
    print({k: v for k, v in res.items() if isinstance(v, float)})


if __name__ == "__main__":
    main()
