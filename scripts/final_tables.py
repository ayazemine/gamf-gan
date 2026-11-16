"""LaTeX rows of the result tables of the revised manuscript, computed from the test-split metrics.json files
(mean +/- sample std over the three training seeds; Welch's t-test against GAMF-GAN).

    python scripts/final_tables.py --out results/final_tables.tex
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

DS = [("cub", "CUB-200-2011"), ("flowers", "Oxford-102 Flowers"), ("coco", "MS COCO")]
MODELS = [("gamf", "GAMF-GAN"), ("dfgan", "DF-GAN"), ("attngan", "AttnGAN"), ("dmgan", "DM-GAN"),
          ("stylegan_t", "TC-StyleGAN")]
ABL = [("gamf", "GAMF-GAN (full)"), ("gamf_no_clip", "w/o CLIP branch"),
       ("gamf_no_attn", "w/o word attention"), ("gamf_no_mem", "w/o dynamic memory")]
# key: (higher is better, decimals, scale)
MAIN = [("FID", False, 1, 1), ("KID", False, 2, 1000), ("IS", True, 2, 1), ("CLIPScore", True, 2, 1),
        ("CLIP_R@1", True, 1, 1), ("DAMSM_R@1", True, 1, 1)]
REF = [("SSIM", True, 3, 1), ("LPIPS", False, 3, 1), ("LPIPS_div", True, 3, 1)]


def load(runs, ds, m, key, scale):
    v = [json.loads(f.read_text(encoding="utf-8"))[key] * scale
         for f in sorted((runs / ds / m).glob("seed_*/metrics.json"))]
    assert len(v) == 3, (ds, m, key, len(v))
    return np.array(v, dtype=float)


def rows(runs, ds, models, metrics, ref="gamf"):
    data = {m: {k: load(runs, ds, m, k, sc) for k, _, _, sc in metrics} for m, _ in models}
    best = {}
    for k, hib, dec, _ in metrics:
        means = {m: round(data[m][k].mean(), dec) for m, _ in models}
        best[k] = max(means.values()) if hib else min(means.values())
    out, pvals = [], {}
    for m, name in models:
        cells = []
        for k, hib, dec, _ in metrics:
            v = data[m][k]
            s = f"{v.mean():.{dec}f} $\\pm$ {v.std(ddof=1):.{dec}f}"
            if round(v.mean(), dec) == best[k]:
                s = f"\\textbf{{{v.mean():.{dec}f}}} $\\pm$ {v.std(ddof=1):.{dec}f}"
            if m != ref:
                p = stats.ttest_ind(data[ref][k], v, equal_var=False).pvalue
                pvals[(m, k)] = p
                if p < 0.05:
                    s += "$^{*}$"
            cells.append(s)
        out.append(f"{name} & " + " & ".join(cells) + " \\\\")
    return out, pvals


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--runs", default="runs")
    a.add_argument("--out", required=True)
    a = a.parse_args()
    runs = Path(a.runs)
    lines = []
    for title, metrics in (("MAIN", MAIN), ("REF", REF)):
        lines.append(f"% ===== {title} =====")
        for ds, dname in DS:
            r, p = rows(runs, ds, MODELS, metrics)
            lines.append(f"\\multicolumn{{{len(metrics) + 1}}}{{@{{}}l}}{{\\textit{{{dname}}}}} \\\\")
            lines += r + ["\\midrule"]
            lines.append("% p-values vs GAMF-GAN: " + "; ".join(f"{m}/{k}={v:.3g}" for (m, k), v in p.items()))
    lines.append("% ===== ABLATION (CUB) =====")
    r, p = rows(runs, "cub", ABL, MAIN + REF[:2])
    lines += r
    lines.append("% p-values vs full: " + "; ".join(f"{m}/{k}={v:.3g}" for (m, k), v in p.items()))
    Path(a.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
