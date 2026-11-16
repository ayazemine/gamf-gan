"""Qualitative figures for the revised manuscript (R3-8: captions must state
how examples were chosen -- here they are RANDOM test captions drawn with a
fixed seed, never hand-picked, unless 'failures' is requested).

    # Figs. 6-8: real vs. all models, one figure per dataset
    python -m gamf.analysis.make_figures compare --runs runs --data data --dataset cub \
        --models attngan dmgan stylegan_t gamf --n 6 --out results/fig_compare_cub.svg
    # Fig. 9: captions with several GAMF-GAN samples (alignment + diversity)
    python -m gamf.analysis.make_figures captions --runs runs --data data --dataset cub --n 5 --k 4
    # New: outputs of the three stages (x1 64px, x2 128px, x3 256px); needs sample --save-stages
    python -m gamf.analysis.make_figures stages --runs runs --data data --dataset cub --n 5
    # New: failure cases = lowest CLIP similarity on the test split (honest limitations)
    python -m gamf.analysis.make_figures failures --runs runs --data data --dataset cub --n 6
"""
import argparse
import csv
import random
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

from gamf.analysis.figstyle import apply_style, save_figure
from gamf.analysis.latex import PLAIN_NAMES, dataset_name

LABEL = PLAIN_NAMES


def _random_note(a):
    return (f"The {a.n} captions were drawn uniformly at random from the held-out test split of "
            f"{dataset_name(a.dataset)} (selection seed {a.pick_seed}); the examples are not curated. "
            f"Generators of training seed {a.seed_run}; first sample of each caption unless stated otherwise.")


def _write_caption(out, text):
    path = Path(out).with_suffix(".caption.txt")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")
    print(f"wrote {path}")


def _captions(data, dataset, split="test"):
    with open(Path(data) / dataset / f"captions_{split}.csv", encoding="utf-8") as f:
        return {(r["image_id"], int(r["caption_idx"])): r for r in csv.DictReader(f)}


def _pick(run, n, seed):
    files = sorted((run / "samples" / "test").glob("*_s0.png"))
    return random.Random(seed).sample(files, min(n, len(files)))


def _key(fname):
    stem = fname.rsplit("_s", 1)[0]
    iid, ci = stem.rsplit("_c", 1)
    return iid, int(ci)


def _grid(rows, col_titles, captions, out, caption, cell=1.6, wrap=34):
    apply_style()
    n_r, n_c = len(rows), len(col_titles)
    fig, axes = plt.subplots(n_r, n_c + 1, figsize=(cell * n_c + 2.6, cell * n_r),
                             gridspec_kw={"width_ratios": [1.6] + [1] * n_c}, squeeze=False)
    for i, (imgs, cap) in enumerate(zip(rows, captions)):
        axes[i][0].text(0, 0.5, textwrap.fill(cap, wrap), fontsize=6.5, va="center", ha="left")
        axes[i][0].axis("off")
        for j, path in enumerate(imgs):
            ax = axes[i][j + 1]
            if path is not None and Path(path).exists():
                ax.imshow(Image.open(path).convert("RGB"))
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(col_titles[j], fontsize=8)
    fig.tight_layout(pad=0.3)
    save_figure(fig, out)
    _write_caption(out, caption)


def compare(a):
    caps = _captions(a.data, a.dataset)
    ref_run = Path(a.runs) / a.dataset / a.reference / f"seed_{a.seed_run}"
    rows, texts = [], []
    for f in _pick(ref_run, a.n, a.pick_seed):
        key = _key(f.name)
        rows.append([caps[key]["real_path"]] + [Path(a.runs) / a.dataset / m / f"seed_{a.seed_run}" / "samples"
                                                / "test" / f.name for m in a.models])
        texts.append(caps[key]["caption"])
    names = [LABEL.get(m, m) for m in a.models]
    cap = (f"Real and generated images on {dataset_name(a.dataset)} for {', '.join(names)}. "
           "All models received the same caption (left) and were trained under the identical protocol. "
           + _random_note(a))
    _grid(rows, ["Real"] + names, texts, a.out or f"results/fig_compare_{a.dataset}.svg", cap)


def captions(a):
    caps = _captions(a.data, a.dataset)
    run = Path(a.runs) / a.dataset / a.reference / f"seed_{a.seed_run}"
    rows, texts = [], []
    for f in _pick(run, a.n, a.pick_seed):
        key = _key(f.name)
        rows.append([caps[key]["real_path"]] + [f.with_name(f.name[:-6] + f"s{k}.png") for k in range(a.k)])
        texts.append(caps[key]["caption"])
    cap = (f"Text-to-image correspondence and diversity of {LABEL.get(a.reference, a.reference)} on "
           f"{dataset_name(a.dataset)}: the real image of each caption and {a.k} samples generated from the same "
           "caption with different noise vectors. " + _random_note(a))
    _grid(rows, ["Real"] + [f"Sample {k + 1}" for k in range(a.k)], texts,
          a.out or f"results/fig_captions_{a.dataset}.svg", cap)


def stages(a):
    caps = _captions(a.data, a.dataset)
    run = Path(a.runs) / a.dataset / a.reference / f"seed_{a.seed_run}"
    rows, texts = [], []
    for f in _pick(run, a.n, a.pick_seed):
        key = _key(f.name)
        rows.append([run / "stages" / "test" / f"{f.stem}_stage{s}.png" for s in (1, 2, 3)])
        texts.append(caps[key]["caption"])
    cap = (f"Outputs of the three stages of the stage-wise model on {dataset_name(a.dataset)}: $x_1$ after the "
           "word-attention stage (64 px), $x_2$ after the dynamic-memory stage (128 px) and $x_3$ after the "
           "style-based stage (256 px), all from one forward pass. " + _random_note(a))
    _grid(rows, [r"$x_1$ (attention, 64 px)", r"$x_2$ (memory, 128 px)", r"$x_3$ (style, 256 px)"],
          texts, a.out or f"results/fig_stages_{a.dataset}.svg", cap)


def failures(a):
    caps = _captions(a.data, a.dataset)
    run = Path(a.runs) / a.dataset / a.reference / f"seed_{a.seed_run}"
    with open(run / "metrics.per_image.csv", encoding="utf-8") as f:
        per = [r for r in csv.DictReader(f) if r.get("clip_cos")]
    per.sort(key=lambda r: float(r["clip_cos"]))
    rows, texts, skipped = [], [], 0
    for r in per:
        if len(rows) == a.n:
            break
        key = _key(r["file"])
        if any(w in caps[key]["caption"].lower() for w in a.exclude_words):
            skipped += 1  # caption wording unsuitable for print; disclosed in the figure caption
            continue
        rows.append([caps[key]["real_path"], run / "samples" / "test" / r["file"]])
        texts.append(f"{caps[key]['caption']} (CLIP {float(r['clip_cos']):.2f})")
    cap = (f"Failure cases of GAMF-GAN on {dataset_name(a.dataset)}: the {a.n} test images with the lowest "
           f"CLIP image-text similarity (value in parentheses), generator of training seed {a.seed_run}. "
           "These examples were selected deliberately to illustrate the limitations of the model.")
    if skipped:
        cap += (f" {skipped} caption(s) among the lowest-ranked ones were skipped because their wording is "
                "unsuitable for publication.")
    _grid(rows, ["Real", "GAMF-GAN"], texts, a.out or f"results/fig_failures_{a.dataset}.svg", cap)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("kind", choices=["compare", "captions", "stages", "failures"])
    p.add_argument("--runs", default="runs")
    p.add_argument("--data", default="data")
    p.add_argument("--dataset", required=True)
    p.add_argument("--models", nargs="+", default=["attngan", "dmgan", "stylegan_t", "dfgan", "gamf"])
    p.add_argument("--reference", default="gamf")
    p.add_argument("--seed-run", type=int, default=0, help="which training seed to show")
    p.add_argument("--pick-seed", type=int, default=2026, help="seed of the random caption selection")
    p.add_argument("--n", type=int, default=6)
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--exclude-words", nargs="*", default=[],
                   help="failures: skip captions containing any of these words (disclosed in the caption)")
    p.add_argument("--out", help="output stem; .svg, .pdf and .png are written")
    a = p.parse_args()
    {"compare": compare, "captions": captions, "stages": stages, "failures": failures}[a.kind](a)


if __name__ == "__main__":
    main()
