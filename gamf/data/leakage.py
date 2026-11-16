"""Train/test leakage control.

The held-out test split must never be seen by any network before final
evaluation. This module provides
* image fingerprints (exact MD5 and 64-bit difference hash) used by
  ``gamf.data.prepare`` to reject test candidates that duplicate or
  near-duplicate a training/validation image (e.g. resized copies);
* ``audit``: a full verification of a written ``split.json`` (disjoint image
  identifiers, files, content hashes, near-duplicates and, for the class-disjoint
  datasets, classes), whose report is stored in ``split.json`` and checked by every
  training script before it starts.

    python -m gamf.data.leakage --config configs/base.yaml configs/datasets/cub.yaml
"""
import argparse
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

from gamf.config import add_config_args, dataset_dir, load_config
from gamf.utils import read_json, write_json


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dhash(path, size=8):
    """64-bit difference hash; robust to resizing, re-encoding and small colour changes."""
    with Image.open(path) as im:
        g = np.asarray(im.convert("L").resize((size + 1, size), Image.BILINEAR), dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int(np.packbits(bits).view(">u8")[0])


def min_hamming(query, pool):
    """Smallest Hamming distance between one 64-bit hash and an array of hashes."""
    if len(pool) == 0:
        return 64
    x = np.bitwise_xor(np.asarray(pool, dtype=np.uint64), np.uint64(query))
    return int(np.unpackbits(x.view(np.uint8).reshape(-1, 8), axis=1).sum(1).min())


def audit(meta, max_hamming):
    """Return a report dict; ``report['passed']`` is False if any overlap exists."""
    raw = Path(meta["raw_root"])
    splits = meta["splits"]
    report = {"max_hamming": max_hamming, "checks": {}}
    fp = {}
    for name, recs in splits.items():
        fp[name] = {"ids": {r["image_id"] for r in recs}, "paths": {r["path"] for r in recs},
                    "md5": {r.get("md5") or md5(raw / r["path"]) for r in recs},
                    "dhash": [r["dhash"] if "dhash" in r else dhash(raw / r["path"]) for r in recs],
                    "classes": {r["class_id"] for r in recs if r["class_id"] is not None}}
    pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    ok = True
    for a, b in pairs:
        near = sum(min_hamming(h, fp[a]["dhash"]) <= max_hamming for h in fp[b]["dhash"])
        c = {"shared_image_ids": len(fp[a]["ids"] & fp[b]["ids"]),
             "shared_files": len(fp[a]["paths"] & fp[b]["paths"]),
             "identical_content": len(fp[a]["md5"] & fp[b]["md5"]),
             "near_duplicates": int(near)}
        if b == "test" and fp["test"]["classes"]:
            c["shared_classes"] = len(fp[a]["classes"] & fp["test"]["classes"])
        report["checks"][f"{a}-{b}"] = c
        if b == "test" and any(c.values()):
            ok = False
        if (a, b) == ("train", "val") and (c["shared_image_ids"] or c["shared_files"] or c["identical_content"]):
            ok = False
    report["passed"] = ok
    report["policy"] = ("Test images share no identifier, file, content hash or near-duplicate (dHash Hamming "
                        f"distance <= {max_hamming}) with training/validation images; for CUB-200-2011 and "
                        "Oxford-102 the test classes are disjoint from the training classes. The vocabulary, "
                        "the DAMSM encoders, all generators and all loss-weight/checkpoint selections use the "
                        "training and validation splits only.")
    return report


def require_clean(cfg):
    """Called by every training script: refuse to train on an unaudited or leaking split."""
    meta = read_json(dataset_dir(cfg) / "split.json")
    rep = meta.get("leakage_check")
    if not rep or not rep.get("passed"):
        raise SystemExit("split.json has no passed leakage check; rerun gamf.data.prepare "
                         "or python -m gamf.data.leakage")


def main():
    p = add_config_args(argparse.ArgumentParser(description=__doc__,
                                                formatter_class=argparse.RawDescriptionHelpFormatter))
    args = p.parse_args()
    cfg = load_config(args.config, args.set)
    path = dataset_dir(cfg) / "split.json"
    meta = read_json(path)
    rep = audit(meta, cfg["data"].get("dup_hamming", 4))
    meta["leakage_check"] = rep
    write_json(meta, path)
    for k, v in rep["checks"].items():
        print(f"{k:12s} {v}")
    print("PASSED" if rep["passed"] else "FAILED: leakage detected")
    if not rep["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
