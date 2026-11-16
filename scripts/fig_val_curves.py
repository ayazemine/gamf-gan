"""Training-stability figure of the revised manuscript (R2-5, R3-6): validation FID during training,
mean +/- std over the training seeds, one panel per dataset, all compared models.

    python scripts/fig_val_curves.py --out results/final_figs/fig_val_curves.svg
"""
import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gamf.analysis.figstyle import apply_style, save_figure  # noqa: E402
from gamf.analysis.latex import PLAIN_NAMES, dataset_name  # noqa: E402

MODELS = ["gamf", "dfgan", "attngan", "dmgan", "stylegan_t"]


def curve(path):
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    by_ep = {}
    for r in rows:  # a resumed run may log an epoch twice; keep the last value
        by_ep[int(float(r["epoch"]))] = float(r["val_FID"])
    ep = sorted(by_ep)
    return np.array(ep), np.array([by_ep[e] for e in ep])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="runs")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.3))
    for ax, ds in zip(axes, ("cub", "flowers", "coco")):
        for m in MODELS:
            cs = [curve(f) for f in sorted((Path(a.runs) / ds / m).glob("seed_*/val_curve.csv"))]
            if not cs:
                continue
            n = min(len(c[0]) for c in cs)
            ep, fid = cs[0][0][:n], np.stack([c[1][:n] for c in cs])
            mu, sd = fid.mean(0), fid.std(0, ddof=1)
            line, = ax.plot(ep, mu, label=PLAIN_NAMES.get(m, m), lw=1.4 if m == "gamf" else 1.0)
            ax.fill_between(ep, np.maximum(mu - sd, 1), mu + sd, alpha=0.2, color=line.get_color(), lw=0)
        ax.set_yscale("log")
        ax.set(xlabel="Epoch", title=dataset_name(ds), xlim=(0, 120))
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel("Validation FID (log scale)")
    axes[0].legend(fontsize=6.5, loc="upper right")
    fig.tight_layout()
    save_figure(fig, a.out)


if __name__ == "__main__":
    main()
