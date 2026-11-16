"""Convergence/stability figure (replaces manuscript Fig. 10; R2-5, R3-6, R3-8).

(a) validation FID vs. epoch, mean +/- std over seeds, per model;
(b) generator and discriminator losses (moving average), mean over seeds.

    python -m gamf.analysis.plot_curves --runs runs --dataset cub \
        --models attngan dmgan stylegan_t gamf --out results/fig_convergence_cub.svg
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from gamf.analysis.figstyle import apply_style, save_figure
from gamf.analysis.latex import PLAIN_NAMES, dataset_name


def _read(path, x, ys):
    data = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            data[x].append(float(r[x]))
            for y in ys:
                data[y].append(float(r[y]))
    return {k: np.array(v) for k, v in data.items()}


def _smooth(v, k=20):
    return np.convolve(v, np.ones(k) / k, mode="valid") if len(v) > k else v


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", default="runs")
    p.add_argument("--dataset", required=True)
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--labels", nargs="*")
    p.add_argument("--out", required=True, help="output stem; .svg, .pdf and .png are written")
    args = p.parse_args()
    labels = dict(zip(args.models, args.labels or [PLAIN_NAMES.get(m, m) for m in args.models]))
    apply_style()
    fig, ax = plt.subplots(1, 3, figsize=(7.16, 2.2))  # IEEE double-column width
    for model in args.models:
        seeds = [s for s in sorted((Path(args.runs) / args.dataset / model).glob("seed_*"))
                 if (s / "val_curve.csv").exists() and (s / "train_log.csv").exists()]
        if not seeds:
            continue
        curves = [_read(s / "val_curve.csv", "epoch", ["val_FID"]) for s in seeds]
        n = min(len(c["epoch"]) for c in curves)
        ep = curves[0]["epoch"][:n]
        fid = np.stack([c["val_FID"][:n] for c in curves])
        m, s = fid.mean(0), fid.std(0)
        line, = ax[0].plot(ep, m, label=f"{labels[model]} (n={len(seeds)})")
        with open(seeds[0] / "val_curve.csv", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for r0, r1 in zip(rows, rows[1:]):
            if "phase" in r0 and r0["phase"] != r1["phase"]:
                ax[0].axvline(float(r0["epoch"]), color=line.get_color(), ls=":", lw=0.8)
        ax[0].fill_between(ep, m - s, m + s, alpha=0.2, color=line.get_color())
        logs = [_read(s / "train_log.csv", "iter", ["d_loss", "g_adv"]) for s in seeds]
        k = min(len(l["iter"]) for l in logs)
        for j, key in ((1, "g_adv"), (2, "d_loss")):
            v = np.stack([_smooth(l[key][:k]) for l in logs]).mean(0)
            ax[j].plot(logs[0]["iter"][:len(v)], v, color=line.get_color(), label=labels[model])
    ax[0].set(xlabel="Epoch", ylabel="Validation FID", title="(a) Validation FID (mean $\\pm$ std)")
    ax[1].set(xlabel="Iteration", ylabel="Loss", title="(b) Generator adversarial loss")
    ax[2].set(xlabel="Iteration", ylabel="Loss", title="(c) Discriminator loss")
    for a in ax:
        a.grid(alpha=0.3)
    ax[0].legend(fontsize=8)
    fig.tight_layout()
    save_figure(fig, args.out)
    n_seeds = len(sorted((Path(args.runs) / args.dataset / args.models[0]).glob("seed_*")))
    cap = ("Training behaviour on the validation split of " + dataset_name(args.dataset) + ": (a) validation FID evaluated "
           f"periodically during training (line: mean, band: standard deviation over {n_seeds} independent runs); "
           "(b) generator adversarial loss and (c) discriminator loss (moving average over 20 logged iterations, "
           "averaged over runs). Dotted vertical lines mark the transitions between the training phases of "
           "stage-wise models, before which the validation FID refers to the lower-resolution stage output.")
    Path(args.out).with_suffix(".caption.txt").write_text(cap + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
