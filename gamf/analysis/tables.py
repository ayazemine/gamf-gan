r"""LaTeX tables for the revised manuscript (complete booktabs table environments).

    python -m gamf.analysis.tables splits    --datasets cub flowers coco
    python -m gamf.analysis.tables hparams   --dataset cub --models gamf attngan dmgan stylegan_t dfgan
    python -m gamf.analysis.tables protocol  --metrics-file runs/cub/gamf/seed_0/metrics.json
    python -m gamf.analysis.tables overall   --models gamf attngan dmgan stylegan_t dfgan
    python -m gamf.analysis.tables stability --models gamf attngan dmgan stylegan_t dfgan
    python -m gamf.analysis.tables cost      --cost results/cost.json

splits    -> dataset subsets and splits                         (R2-3, R3-3)
hparams   -> training configuration of every model and loss weights (R2-3, R3-2, R3-5, R3-6)
protocol  -> evaluation protocol: implementation, reference, sample counts (R2-4, R3-4)
overall   -> results averaged over the three datasets (replaces Table II)
stability -> run-to-run variance and convergence statistics     (R2-5, R3-6)
cost      -> computational cost (replaces Table I)              (R2-6, R3-6)
hparams-all -> every hyperparameter of every run in one CSV/Markdown table
               (python -m gamf.analysis.tables hparams-all)
Per-dataset and ablation tables are produced by gamf.analysis.aggregate.
"""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import yaml

from gamf.analysis.aggregate import LABELS, METRICS, summarize
from gamf.analysis.latex import BASELINE_NOTE, dataset_name, model_name, table_env, write_lines

PM = r" $\pm$ "


def _fmt(s, d):
    if s is None:
        return "--"
    return f"{s['mean']:.{d}f}" + ("" if math.isnan(s["std"]) else PM + f"{s['std']:.{d}f}")


# ------------------------------------------------------------------------ splits
def splits(a):
    rows = [r"\begin{tabular}{@{}lrrrcl@{}}", r"\toprule",
            r"Dataset & Train & Val. & Test & Capt./img. & Test images drawn from \\", r"\midrule"]
    seeds = set()
    for ds in a.datasets:
        m = json.loads((Path(a.data) / ds / "split.json").read_text(encoding="utf-8"))
        c, seeds = m["counts"], seeds | {m["split_seed"]}
        src = "unseen classes" if ds in ("cub", "flowers") else "val2014"
        rows.append(f"{dataset_name(ds)} & {c['train']:,} & {c['val']:,} & {c['test']:,} & "
                    f"{m['filters']['captions_per_image']} & {src} \\\\")
    rows += [r"\bottomrule", r"\end{tabular}"]
    cap = ("Dataset subsets and splits. Images were sampled uniformly at random with a fixed seed "
           f"({', '.join(map(str, sorted(seeds)))}) after discarding non-RGB images and images whose shorter side "
           "is below 256 px. Training and validation images come from the training classes (CUB-200-2011, "
           "Oxford-102 Flowers; class-disjoint split of Reed et al.) or from train2014 (MS COCO). Test candidates "
           "identical to or near-duplicates of a training/validation image (difference hash) were rejected. The test "
           "split was used neither for training nor for model selection.")
    write_lines(table_env(rows, cap, "tab:splits"), a.out or "results/table_splits.tex")


# ----------------------------------------------------------------------- hparams
def _stages(c):
    g = c["model"]["generator"]
    if g == "cascade":
        return r" $\rightarrow$ ".join(c["model"]["stages"])
    return {"stylegan_t": "style, 4--256 px", "dfgan": "fusion, 4--256 px"}[g]


def _stagewise(c):
    return c["train"].get("schedule") == "stagewise" and c["model"]["generator"] == "cascade"


def _schedule(c):
    if _stagewise(c):
        return "stage-wise, " + " / ".join(str(e) for e in c["train"]["stage_epochs"]) + " ep."
    return f"joint, {c['train']['epochs']} ep."


def _objective(c):
    l = c["loss"]
    if c["model"]["generator"] == "dfgan":
        return "hinge + MA-GP"
    if _stagewise(c):
        return " $\rightarrow$ ".join("+".join(t for t in ph if t == "adv" or l.get(f"lambda_{t}", 0) > 0)
                                         for ph in c["train"]["stage_losses"])
    terms = ["adv"] + [n for n, k in (("KL", "lambda_kl"), ("DAMSM", "lambda_damsm"), ("mem", "lambda_mem"),
                                      ("perc", "lambda_perc"), ("style", "lambda_style")) if l[k] > 0]
    return ", ".join(terms)


def hparams(a):
    cfgs = {}
    for mo in a.models:
        f = Path(a.runs) / a.dataset / mo / "seed_0" / "config.yaml"
        cfgs[mo] = yaml.safe_load(f.read_text(encoding="utf-8"))
    split = json.loads((Path(a.data) / a.dataset / "split.json").read_text(encoding="utf-8"))["counts"]
    lam = lambda k: (lambda c: f"{c['loss'][k]:g}")  # noqa: E731
    rows = [("Training data", lambda c: f"{split['train']:,} images (same for all)"),
            ("Resolution", lambda c: f"{c['data']['image_size']} px"),
            ("Captions", lambda c: "random caption per visit"),
            ("Generator", _stages), ("Objective", _objective), ("Training schedule", _schedule),
            ("Batch size", lambda c: c["train"]["batch_size"]),
            ("Learning rate (G / D)", lambda c: f"{c['train']['lr_g']:g} / {c['train']['lr_d']:g}"),
            (r"Adam $(\beta_1, \beta_2)$", lambda c: f"({c['train']['betas'][0]:g}, {c['train']['betas'][1]:g})"),
            ("EMA decay", lambda c: f"{c['train']['ema_beta']:g}"),
            ("Mixed precision", lambda c: c["train"]["amp"]),
            ("Checkpoint selection", lambda c: f"best val. FID (every {c['train']['eval_every']} ep.)"),
            (r"$\lambda_{\mathrm{adv}}$", lam("lambda_adv")), (r"$\lambda_{\mathrm{KL}}$", lam("lambda_kl")),
            (r"$\lambda_{\mathrm{DAMSM}}$", lam("lambda_damsm")), (r"$\lambda_{\mathrm{mem}}$", lam("lambda_mem")),
            (r"$\lambda_{\mathrm{perc}}$", lam("lambda_perc")), (r"$\lambda_{\mathrm{style}}$", lam("lambda_style")),
            ("Sentence feature", lambda c: "mean of word features" if c["text"].get("sentence") == "mean"
             else "last Bi-LSTM state")]
    lines = [r"\begin{tabular}{@{}l" + "c" * len(cfgs) + "@{}}", r"\toprule",
             "Setting & " + " & ".join(model_name(m) for m in cfgs) + r" \\", r"\midrule"]
    for label, fn in rows:
        lines.append(label + " & " + " & ".join(str(fn(c)) for c in cfgs.values()) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    d = next(iter(cfgs.values()))["damsm"]
    cap = (f"Training configuration of all models on {dataset_name(a.dataset)}. All models share the frozen "
           f"DAMSM encoders (pre-trained on the same training split; $\\gamma_1, \\gamma_2, \\gamma_3 = "
           f"{d['gamma1']:g}, {d['gamma2']:g}, {d['gamma3']:g}$), the data, the number of epochs, the batch size "
           "and the checkpoint-selection rule. Baselines use their original objectives.")
    write_lines(table_env(lines, cap, f"tab:hparams_{a.dataset}", wide=True, note=BASELINE_NOTE),
                a.out or f"results/table_hparams_{a.dataset}.tex")


# ------------------------------------------------------------------- hparams-all
def hparams_all(a):
    """Every hyperparameter of every run in one table (CSV + Markdown): one row per
    parameter, one column per run; parameters that differ between runs are listed first."""
    from gamf.config import flatten
    runs = Path(a.runs)
    cols = {}
    for f in sorted(runs.glob("*/*/seed_*/config.yaml")):
        name = "/".join(f.parts[-4:-1])
        cols[name] = dict(flatten(yaml.safe_load(f.read_text(encoding="utf-8"))))
    for f in sorted(runs.glob("*/damsm/config.yaml")):
        cols["/".join(f.parts[-3:-1])] = dict(flatten(yaml.safe_load(f.read_text(encoding="utf-8"))))
    if not cols:
        raise SystemExit(f"no runs found under {runs}")
    keys = sorted({k for c in cols.values() for k in c})
    differs = lambda k: len({str(c.get(k, "")) for c in cols.values()}) > 1  # noqa: E731
    keys = [k for k in keys if differs(k)] + [k for k in keys if not differs(k)]
    out = Path(a.out or "results/hyperparameters_all.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["parameter", "differs_between_runs"] + list(cols))
        for k in keys:
            w.writerow([k, differs(k)] + [c.get(k, "") for c in cols.values()])
    md = ["| Parameter | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    md += [f"| `{k}`{' *' if differs(k) else ''} | " + " | ".join(str(c.get(k, "")) for c in cols.values()) + " |"
           for k in keys]
    out.with_suffix(".md").write_text("\n".join(md) + "\n\n\\* differs between runs\n", encoding="utf-8")
    print(f"wrote {out} and {out.with_suffix('.md')} ({len(cols)} runs, {len(keys)} parameters)")


# ---------------------------------------------------------------------- protocol
def protocol(a):
    m = json.loads(Path(a.metrics_file).read_text(encoding="utf-8"))
    n_gen, n_ref, spc = m.get("n_images", "--"), m.get("n_reference", "--"), m.get("samples_per_caption", "--")
    tf = m.get("torch_fidelity", "")
    rows = [("FID, KID, IS", f"torch-fidelity {tf}, Inception-v3", f"{n_ref:,} real test images",
             f"{n_gen:,} ({spc}/caption)"),
            ("SSIM", r"Gaussian window $11$, $\sigma = 1.5$, RGB", "source image of the caption", f"{n_gen:,}"),
            ("LPIPS", m.get("lpips_impl", "LPIPS v0.1, AlexNet").strip(), "source image of the caption", f"{n_gen:,}"),
            ("LPIPS-div", "pairwise LPIPS, same caption", "--", f"{m.get('n_captions', 0):,} captions"),
            ("CLIP", m.get("clip_model", "CLIP").split("/")[-1] + ", cosine similarity", "input caption", f"{n_gen:,}"),
            ("CLIP-R@1", m.get("clip_model", "CLIP").split("/")[-1],
             f"1 true + {m.get('r_precision_candidates', 100) - 1} random captions", f"{n_gen:,}"),
            ("R-precision", "frozen DAMSM encoders",
             f"1 true + {m.get('r_precision_candidates', 100) - 1} random captions", f"{n_gen:,}")]
    lines = [r"\begin{tabular}{@{}llll@{}}", r"\toprule",
             r"Metric & Implementation & Reference & Generated images \\", r"\midrule"]
    lines += [" & ".join(r) + r" \\" for r in rows] + [r"\bottomrule", r"\end{tabular}"]
    cap = ("Evaluation protocol. All metrics are computed on the held-out test split at $256 \\times 256$ px, "
           "from images generated with the exponential-moving-average weights of the checkpoint with the lowest "
           f"validation FID (generation seed {m.get('eval_seed', '--')}). Real images are resized (shorter "
           "side 256 px) and centre-cropped.")
    write_lines(table_env(lines, cap, "tab:protocol", wide=True), a.out or "results/table_protocol.tex")


# ----------------------------------------------------------------------- overall
def overall(a):
    """Per seed, average each metric over the datasets; then mean +/- std over seeds."""
    runs = Path(a.runs)
    datasets = a.datasets or ["cub", "flowers", "coco"]
    table = {}
    for mo in a.models:
        per_seed = {}
        for ds in datasets:
            for f in (runs / ds / mo).glob("seed_*/metrics.json"):
                per_seed.setdefault(f.parent.name, []).append(json.loads(f.read_text(encoding="utf-8")))
        per_seed = {s: v for s, v in per_seed.items() if len(v) == len(datasets)}
        for metric in a.metrics:
            vals = [np.mean([r[metric] for r in v]) for v in per_seed.values() if all(metric in r for r in v)]
            if vals:
                table[(mo, metric)] = summarize(vals)
    metrics = [m for m in a.metrics if any((mo, m) in table for mo in a.models)]
    lines = [r"\begin{tabular}{@{}l" + "c" * len(metrics) + "@{}}", r"\toprule",
             "Model & " + " & ".join(LABELS[m] for m in metrics) + r" \\", r"\midrule"]
    best = {}
    for m in metrics:
        vals = {mo: table[(mo, m)]["mean"] for mo in a.models if (mo, m) in table}
        top = round((max if METRICS[m][0] else min)(vals.values()), METRICS[m][1])
        best[m] = {mo for mo, v in vals.items() if round(v, METRICS[m][1]) == top}
    for mo in a.models:
        cells = []
        for m in metrics:
            txt = _fmt(table.get((mo, m)), METRICS[m][1])
            if mo in best[m] and len(best[m]) < len(a.models):
                txt = r"\textbf{" + txt + "}"
            cells.append(txt)
        lines.append(model_name(mo) + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    n = max((s["n"] for s in table.values()), default=0)
    cap = ("Overall results on the held-out test splits, averaged over "
           + ", ".join(dataset_name(d) for d in datasets) + r". For each training seed the metric is averaged "
           f"over the datasets; values are mean{PM}standard deviation over {n} seeds; best values in bold.")
    write_lines(table_env(lines, cap, "tab:overall", wide=len(metrics) > 5, note=BASELINE_NOTE),
                a.out or "results/table_overall.tex")


# --------------------------------------------------------------------- stability
def _curve(path):
    """Validation curve of the final output: for stage-wise runs only the last phase."""
    with open(path, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["val_FID"] not in ("", "nan")]
    if rows and "phase" in rows[-1]:
        rows = [r for r in rows if r["phase"] == rows[-1]["phase"]]
    return np.array([float(r["epoch"]) for r in rows]), np.array([float(r["val_FID"]) for r in rows])


def stability(a):
    runs = Path(a.runs)
    datasets = a.datasets or ["cub", "flowers", "coco"]
    lines = [r"\begin{tabular}{@{}llccccc@{}}", r"\toprule",
             r"Dataset & Model & Runs & Best val. FID & Best epoch & Late fluct. & Diverged \\", r"\midrule"]
    for ds in datasets:
        first = True
        for mo in a.models:
            seeds = sorted((runs / ds / mo).glob("seed_*/val_curve.csv"))
            if not seeds:
                continue
            best, best_ep, fluct, diverged = [], [], [], 0
            for f in seeds:
                ep, fid = _curve(f)
                if not len(fid):
                    continue
                i = int(np.argmin(fid))
                best.append(fid[i]); best_ep.append(ep[i])
                late = fid[int(len(fid) * 0.75):]
                fluct.append(float(np.std(late)) if len(late) > 1 else float("nan"))
                diverged += int(fid[-1] > a.divergence_ratio * fid[i])
            if not best:
                continue
            name = dataset_name(ds) if first else ""
            first = False
            lines.append(f"{name} & {model_name(mo)} & {len(best)} & {_fmt(summarize(best), 2)} & "
                         f"{_fmt(summarize(best_ep), 0)} & {np.nanmean(fluct):.2f} & {diverged}/{len(best)} \\\\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    cap = ("Training stability. Validation FID of the final $256 \times 256$ output was evaluated periodically (for "
           "stage-wise training, during the final phase). Best val. FID and best "
           f"epoch: mean{PM}standard deviation over independent runs (seeds). Late fluct.: standard deviation of "
           "validation FID over the last 25\\% of evaluations, averaged over runs. Diverged: runs whose final "
           f"validation FID exceeds {a.divergence_ratio:g} times their best value.")
    write_lines(table_env(lines, cap, "tab:stability", wide=True, note=BASELINE_NOTE),
                a.out or "results/table_stability.tex")


# -------------------------------------------------------------------------- cost
def cost(a):
    data = json.loads(Path(a.cost).read_text(encoding="utf-8"))
    hw = data["hardware"]
    lines = [r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
             r"Model & G (M) & D (M) & GFLOPs/img & ms/img & Training (h) & Peak memory (GB) \\", r"\midrule"]
    for r in data["rows"]:
        th = (f"{r['train_hours_mean']:.1f}" if "train_hours_mean" in r else "--")
        pm = (f"{r['peak_train_mem_gb']:.1f}" if r.get("peak_train_mem_gb") else "--")
        lines.append(f"{model_name(r['model'])} & {r['params_G_M']:.2f} & {r['params_D_M']:.2f} & "
                     f"{r['GFLOPs_per_image']:.1f} & {r['latency_ms_per_image_b16']:.2f} & {th} & {pm} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    cap = (f"Computational cost. Parameters of the generator (G) and discriminator(s) (D); GFLOPs and inference "
           f"time per $256 \\times 256$ image (batch of 16, median of 50 runs); training time averaged over seeds "
           f"(identical epochs and batch size). Hardware: {hw.get('gpu', 'CPU')} ({hw.get('gpu_memory_gb', '--')} GB), "
           f"{hw.get('cpu', '')}; PyTorch {hw.get('torch', '')}, CUDA {hw.get('cuda', '--')}.")
    write_lines(table_env(lines, cap, "tab:cost", wide=True, note=BASELINE_NOTE), a.out or "results/table_cost.tex")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("splits"); s.add_argument("--datasets", nargs="+", required=True)
    s = sub.add_parser("hparams"); s.add_argument("--dataset", required=True); s.add_argument("--models", nargs="+", required=True)
    s = sub.add_parser("protocol"); s.add_argument("--metrics-file", required=True)
    for name in ("overall", "stability"):
        s = sub.add_parser(name)
        s.add_argument("--models", nargs="+", required=True)
        s.add_argument("--datasets", nargs="*")
        s.add_argument("--metrics", nargs="+", default=list(METRICS))
        s.add_argument("--divergence-ratio", type=float, default=1.5)
    s = sub.add_parser("cost"); s.add_argument("--cost", default="results/cost.json")
    sub.add_parser("hparams-all")
    for s in sub.choices.values():
        s.add_argument("--runs", default="runs")
        s.add_argument("--data", default="data")
        s.add_argument("--out")
    a = p.parse_args()
    {"splits": splits, "hparams": hparams, "protocol": protocol, "overall": overall,
     "stability": stability, "cost": cost, "hparams-all": hparams_all}[a.cmd](a)


if __name__ == "__main__":
    main()
