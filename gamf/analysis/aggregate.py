"""Aggregate metrics over seeds: mean, std, 95% CI, Welch t-test vs. GAMF-GAN.

Reads runs/<dataset>/<model>/seed_*/metrics.json and writes
results/summary.csv plus a booktabs LaTeX table per dataset (R2-5, R3-4, R3-6).

    python -m gamf.analysis.aggregate --runs runs --out results \
        --models attngan dmgan stylegan_t gamf
"""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

from gamf.analysis.latex import BASELINE_NOTE, dataset_name, model_name, table_env, write_lines

METRICS = {  # name: (higher_is_better, decimals)
    "FID": (False, 2), "KID": (False, 4), "IS": (True, 2), "SSIM": (True, 3),
    "LPIPS": (False, 3), "LPIPS_div": (True, 3), "CLIP_cos": (True, 3),
    "CLIP_R@1": (True, 1), "DAMSM_R@1": (True, 1),
}
LABELS = {"FID": r"FID$\downarrow$", "KID": r"KID$\downarrow$", "IS": r"IS$\uparrow$",
          "SSIM": r"SSIM$\uparrow$", "LPIPS": r"LPIPS$\downarrow$",
          "LPIPS_div": r"LPIPS-div$\uparrow$", "CLIP_cos": r"CLIP$\uparrow$",
          "CLIP_R@1": r"CLIP-R@1$\uparrow$", "DAMSM_R@1": r"R-prec.$\uparrow$"}


def summarize(values):
    v = np.asarray(values, dtype=float)
    n = len(v)
    std = v.std(ddof=1) if n > 1 else float("nan")
    half = stats.t.ppf(0.975, n - 1) * std / math.sqrt(n) if n > 1 else float("nan")
    return {"n": n, "mean": v.mean(), "std": std, "ci95": half}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", default="runs")
    p.add_argument("--out", default="results")
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--reference", default="gamf")
    p.add_argument("--metrics", nargs="+", default=list(METRICS))
    p.add_argument("--file", default="metrics.json")
    p.add_argument("--kind", default="results", choices=["results", "ablation"],
                   help="caption/label style of the LaTeX tables")
    args = p.parse_args()
    runs, out = Path(args.runs), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for ds_dir in sorted(d for d in runs.iterdir() if d.is_dir()):
        data = {}
        for model in args.models:
            files = sorted((ds_dir / model).glob(f"seed_*/{args.file}"))
            if files:
                data[model] = [json.loads(f.read_text(encoding="utf-8")) for f in files]
        if not data:
            continue
        ref = data.get(args.reference)
        table = {}
        for model, runs_m in data.items():
            for metric in args.metrics:
                vals = [r[metric] for r in runs_m if metric in r]
                if not vals:
                    continue
                s = summarize(vals)
                s["p_vs_ref"] = float("nan")
                if ref and model != args.reference:
                    rv = [r[metric] for r in ref if metric in r]
                    if len(rv) > 1 and len(vals) > 1:
                        s["p_vs_ref"] = stats.ttest_ind(rv, vals, equal_var=False).pvalue
                table[(model, metric)] = s
                rows.append({"dataset": ds_dir.name, "model": model, "metric": metric, **s})
        proto = next(iter(data.values()))[0]
        write_latex(ds_dir.name, args.models, args.metrics, table, proto, args.reference, args.kind,
                    out / f"table_{args.kind}_{ds_dir.name}.tex")

    with open(out / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, ["dataset", "model", "metric", "n", "mean", "std", "ci95", "p_vs_ref"])
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} rows -> {out / 'summary.csv'}")


def protocol_caption(dataset, proto, n_seeds, kind, reference):
    what = ("Ablation study" if kind == "ablation" else "Quantitative comparison") + f" on {dataset_name(dataset)}"
    parts = [what + r" (held-out test split, $256 \times 256$ px)."]
    if "n_images" in proto:
        parts.append(f"{proto['n_images']:,} generated images ({proto.get('samples_per_caption', '?')} per caption, "
                     f"{proto.get('n_captions', 0):,} test captions); FID, KID and IS against "
                     f"{proto.get('n_reference', 0):,} real test images.")
    parts.append("SSIM and LPIPS are computed against the source image of each caption; "
                 f"CLIP: {proto.get('clip_model', 'CLIP').split('/')[-1]}; R-precision: 1 true and "
                 f"{proto.get('r_precision_candidates', 100) - 1} random captions.")
    parts.append(r"Mean $\pm$ standard deviation over " + str(n_seeds) + " training seeds; best values in bold; "
                 r"$^{*}$\,$p < 0.05$ (Welch's $t$-test against " + model_name(reference) + ").")
    return " ".join(parts)


def write_latex(dataset, models, metrics, table, proto, reference, kind, path):
    metrics = [m for m in metrics if any((mo, m) in table for mo in models)]
    best = {}
    for m in metrics:
        vals = {mo: table[(mo, m)]["mean"] for mo in models if (mo, m) in table}
        if vals:
            d = METRICS[m][1]
            top = round((max if METRICS[m][0] else min)(vals.values()), d)
            best[m] = {mo for mo, v in vals.items() if round(v, d) == top}
    n_seeds = max((s["n"] for s in table.values()), default=0)
    lines = [r"\begin{tabular}{@{}l" + "c" * len(metrics) + "@{}}", r"\toprule",
             ("Variant" if kind == "ablation" else "Model") + " & "
             + " & ".join(LABELS[m] for m in metrics) + r" \\", r"\midrule"]
    for mo in models:
        cells = []
        for m in metrics:
            s = table.get((mo, m))
            if s is None:
                cells.append("--")
                continue
            d = METRICS[m][1]
            txt = f"{s['mean']:.{d}f}" + ("" if math.isnan(s["std"]) else r" $\pm$ " + f"{s['std']:.{d}f}")
            if mo in best.get(m, ()) and len(best[m]) < len(models):
                txt = r"\textbf{" + txt + "}"
            if s.get("p_vs_ref", float("nan")) < 0.05:
                txt += r"$^{*}$"
            cells.append(txt)
        lines.append(model_name(mo) + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    note = None if kind == "ablation" else BASELINE_NOTE
    write_lines(table_env(lines, protocol_caption(dataset, proto, n_seeds, kind, reference),
                          f"tab:{kind}_{dataset}", wide=len(metrics) > 5, note=note), path)


if __name__ == "__main__":
    main()
