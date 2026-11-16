"""Download the raw datasets and arrange them in the layout expected by
configs/datasets/*.yaml (raw/cub, raw/flowers, raw/coco).

    python scripts/download_data.py --datasets cub flowers coco [--raw raw]

Sizes: CUB ~1.1 GB, Oxford-102 ~0.33 GB, COCO 2014 ~19 GB (train2014 13 GB,
val2014 6 GB, captions 0.25 GB). Downloads resume if interrupted.

Captions of Reed et al. (2016) for CUB and Oxford-102 are on Google Drive
(links from github.com/reedscot/cvpr2016) and are fetched with ``gdown``
(pip install gdown). If Drive refuses the download, fetch the two archives
manually in a browser, put them in raw/_downloads/ as reed_birds.tar.gz and
reed_flowers.tar.gz, and rerun this script.
"""
import argparse
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

URLS = {
    "cub_images": ("https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1",
                   "CUB_200_2011.tgz"),
    "flowers_images": ("https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz", "102flowers.tgz"),
    "coco_train": ("http://images.cocodataset.org/zips/train2014.zip", "train2014.zip"),
    "coco_val": ("http://images.cocodataset.org/zips/val2014.zip", "val2014.zip"),
    "coco_ann": ("http://images.cocodataset.org/annotations/annotations_trainval2014.zip",
                 "annotations_trainval2014.zip"),
}
REED_DRIVE = {"cub": ("0B0ywwgffWnLLZW9uVHNjb2JmNlE", "reed_birds.tar.gz"),
              "flowers": ("0B0ywwgffWnLLcms2WWJQRFNSWXM", "reed_flowers.tar.gz")}


def download(url, dest):
    if dest.exists() and dest.stat().st_size > 0 and not dest.with_suffix(dest.suffix + ".part").exists():
        print(f"[skip] {dest.name}")
        return dest
    part = dest.with_suffix(dest.suffix + ".part")
    done = part.stat().st_size if part.exists() else 0
    req = urllib.request.Request(url, headers={"Range": f"bytes={done}-"} if done else {})
    print(f"[get ] {dest.name}")
    with urllib.request.urlopen(req) as r, open(part, "ab" if done else "wb") as f:
        total = done + int(r.headers.get("Content-Length", 0))
        while chunk := r.read(1 << 22):
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r       {done / 1e9:.2f} / {total / 1e9:.2f} GB", end="", flush=True)
    print()
    part.replace(dest)
    return dest


def extract(archive, to):
    marker = to / f".extracted_{archive.name}"
    if marker.exists():
        return
    print(f"[untar] {archive.name}")
    to.mkdir(parents=True, exist_ok=True)
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as z:
            z.extractall(to)
    else:
        with tarfile.open(archive) as t:
            t.extractall(to, filter="data") if sys.version_info >= (3, 12) else t.extractall(to)
    marker.touch()


def reed_captions(name, dl, work, target):
    """Find text_c10 + class split files inside the Reed archive and copy them."""
    if (target / "text_c10").exists() and (target / "testclasses.txt").exists():
        return
    fid, fname = REED_DRIVE[name]
    archive = dl / fname
    if not archive.exists():
        try:
            subprocess.run([sys.executable, "-m", "gdown", fid, "-O", str(archive)], check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            sys.exit(f"Could not fetch Reed captions for {name}. Download "
                     f"https://drive.google.com/open?id={fid} manually to {archive} and rerun.")
    extract(archive, work / f"reed_{name}")
    text_dirs = list((work / f"reed_{name}").rglob("text_c10"))
    if not text_dirs:
        sys.exit(f"text_c10 not found in {archive}")
    src = text_dirs[0].parent
    shutil.copytree(src / "text_c10", target / "text_c10", dirs_exist_ok=True)
    for f in ("trainvalclasses.txt", "testclasses.txt", "allclasses.txt"):
        if (src / f).exists():
            shutil.copy2(src / f, target / f)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", nargs="+", default=["cub", "flowers", "coco"])
    p.add_argument("--raw", default="raw")
    a = p.parse_args()
    raw = Path(a.raw)
    dl, work = raw / "_downloads", raw / "_extract"
    dl.mkdir(parents=True, exist_ok=True)

    if "cub" in a.datasets:
        tgt = raw / "cub"
        extract(download(*URLS["cub_images"][:1], dl / URLS["cub_images"][1]), work / "cub")
        if not (tgt / "images").exists():
            tgt.mkdir(parents=True, exist_ok=True)
            shutil.move(str(next((work / "cub").rglob("CUB_200_2011")) / "images"), tgt / "images")
        reed_captions("cub", dl, work, tgt)
    if "flowers" in a.datasets:
        tgt = raw / "flowers"
        extract(download(*URLS["flowers_images"][:1], dl / URLS["flowers_images"][1]), work / "flowers")
        if not (tgt / "jpg").exists():
            tgt.mkdir(parents=True, exist_ok=True)
            shutil.move(str(work / "flowers" / "jpg"), tgt / "jpg")
        reed_captions("flowers", dl, work, tgt)
    if "coco" in a.datasets:
        tgt = raw / "coco"
        for key in ("coco_ann", "coco_val", "coco_train"):
            extract(download(URLS[key][0], dl / URLS[key][1]), tgt)

    print("Done. Check layout:", *[str(raw / d) for d in a.datasets])
    print("Archives in raw/_downloads and raw/_extract can be deleted after a successful prepare step.")


if __name__ == "__main__":
    main()
