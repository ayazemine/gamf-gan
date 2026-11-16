"""Analysis of the gated word-attention and dynamic-memory branches of trained GAMF-GAN generators.

1. Learned gate values (g_a per attention module, g_m of the memory module).
2. Relative update of each branch on the test split: ||out - h|| / ||h|| per sample, i.e. how strongly the
   branch changes the feature map it is applied to (comparable across branches, unlike the raw gates).
3. Word-attention maps of the 64 px attention module for randomly drawn test captions (figure).

    python scripts/branch_analysis.py --runs runs --out results/branch_analysis
"""
import argparse
import json
import random
import statistics as st
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gamf.analysis.figstyle import apply_style, save_figure  # noqa: E402
from gamf.config import damsm_path, load_config  # noqa: E402
from gamf.data.dataset import T2IDataset  # noqa: E402
from gamf.data.text import tokenize  # noqa: E402
from gamf.models import build_generator  # noqa: E402
from gamf.models.text_encoder import load_damsm  # noqa: E402

RES = {"2": 32, "3": 64, "4": 128}          # block index -> resolution of the attention / memory modules
STOP = {"a", "an", "the", "this", "that", "and", "with", "has", "have", "is", "are", "of", "its", "it", "on",
        "in", "to", "as", "at", "very", "some", "there"}


def load(run, device):
    cfg = load_config([run / "config.yaml"])
    text_enc, _ = load_damsm(damsm_path(cfg), device)
    dim = cfg["text"]["hidden_dim"]
    G = build_generator(cfg, dim, dim).to(device).eval()
    G.load_state_dict(torch.load(run / "ckpt" / "best.pt", map_location=device, weights_only=False)["G_ema"])
    return cfg, text_enc, G


@torch.no_grad()
def relative_updates(G, text_enc, ds, device, n=256, seed=2026, bs=32):
    """Mean over n test captions of ||branch output - input|| / ||input|| for every gated branch."""
    stats, hooks = {}, []

    def hook(name):
        def f(mod, inp, out):
            h = inp[0].float()
            r = (out.float() - h).flatten(1).norm(dim=1) / h.flatten(1).norm(dim=1)
            stats.setdefault(name, []).extend(r.tolist())
        return f

    for k, m in G.attn.items():
        hooks.append(m.register_forward_hook(hook(f"attn_{RES[k]}")))
    for k, m in G.mem.items():
        hooks.append(m.register_forward_hook(hook(f"mem_{RES[k]}")))
    recs = random.Random(seed).sample(ds.records, min(n, len(ds.records)))
    gen = torch.Generator().manual_seed(seed)
    for i in range(0, len(recs), bs):
        chunk = recs[i:i + bs]
        enc = [ds.encode_caption(r["captions"][r["eval_caption_idx"]]) for r in chunk]
        caps = torch.stack([e[0] for e in enc]).to(device)
        words, sent = text_enc(caps, torch.tensor([e[1] for e in enc]))
        G(torch.randn(len(chunk), G.fc.in_features, generator=gen).to(device), sent, words, caps == 0)
    for h in hooks:
        h.remove()
    return {k: float(np.mean(v)) for k, v in stats.items()}


@torch.no_grad()
def attention_maps(G, text_enc, ds, device, caption, seed, module="3"):
    """Generated image and the per-word attention maps of one attention module."""
    store = {}
    hk = G.attn[module].attn.register_forward_hook(lambda m, i, o: store.__setitem__("attn", o[1]))
    ids, length = ds.encode_caption(caption)
    caps = ids[None].to(device)
    words, sent = text_enc(caps, torch.tensor([length]))
    z = torch.randn(1, G.fc.in_features, generator=torch.Generator().manual_seed(seed)).to(device)
    img = G(z, sent, words, caps == 0)["imgs"][-1][0]
    hk.remove()
    a = store["attn"][0].float()                         # HW x T
    side = int(a.shape[0] ** 0.5)
    maps = a.T.reshape(-1, 1, side, side)                # T x 1 x side x side
    maps = F.interpolate(maps, size=img.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
    img = ((img.float().clamp(-1, 1) + 1) / 2).permute(1, 2, 0).cpu().numpy()
    toks = tokenize(caption)[:length]
    return img, toks, maps[:length].cpu().numpy()


def figure(rows, out, max_words=7):
    import matplotlib.pyplot as plt
    import textwrap
    apply_style()
    ncol = max_words + 1
    fig, axes = plt.subplots(len(rows), ncol, figsize=(7.16, 1.05 * len(rows) + 0.15), squeeze=False)
    for r, (cap, img, toks, maps) in enumerate(rows):
        sel = [i for i, t in enumerate(toks) if t not in STOP][:max_words]
        vmax = max(maps[i].max() for i in sel) + 1e-8   # common scale per caption: weak attention stays dark
        axes[r][0].imshow(img)
        axes[r][0].set_title(textwrap.fill(cap, 30), fontsize=4.8, loc="left")
        for c in range(1, ncol):
            ax = axes[r][c]
            if c - 1 < len(sel):
                i = sel[c - 1]
                m = maps[i] / vmax
                ax.imshow(img)
                ax.imshow(m, cmap="jet", alpha=0.5, vmin=0, vmax=1)
                ax.set_title(toks[i], fontsize=6.5)
        for ax in axes[r]:
            ax.set_xticks([]); ax.set_yticks([]); [s.set_visible(False) for s in ax.spines.values()]
    fig.tight_layout(pad=0.2, w_pad=0.1, h_pad=0.3)
    save_figure(fig, out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="runs")
    p.add_argument("--model", default="gamf")
    p.add_argument("--out", default="results/branch_analysis")
    p.add_argument("--pick-seed", type=int, default=2026)
    a = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    result, rows = {}, []
    for ds_name in ("cub", "flowers", "coco"):
        for s in range(3):
            run = Path(a.runs) / ds_name / a.model / f"seed_{s}"
            cfg, text_enc, G = load(run, device)
            gates = {f"attn_{RES[k]}": float(m.gate) for k, m in G.attn.items()}
            gates.update({f"mem_{RES[k]}": float(m.gate) for k, m in G.mem.items()})
            ds = T2IDataset(cfg, "test", train=False, allow_test=True)
            rel = relative_updates(G, text_enc, ds, device)
            result[f"{ds_name}/seed_{s}"] = {"gate": gates, "relative_update": rel}
            print(ds_name, s, {k: round(v, 3) for k, v in gates.items()}, {k: round(v, 3) for k, v in rel.items()})
            if s == 0:   # attention maps: two random test captions per dataset, generator of seed 0
                for rec in random.Random(a.pick_seed).sample(ds.records, 2):
                    cap = rec["captions"][rec["eval_caption_idx"]]
                    rows.append((cap, *attention_maps(G, text_enc, ds, device, cap, a.pick_seed)))
    summary = {}
    for ds_name in ("cub", "flowers", "coco"):
        runs = [v for k, v in result.items() if k.startswith(ds_name)]
        for kind in ("gate", "relative_update"):
            for key in runs[0][kind]:
                vals = [abs(r[kind][key]) if kind == "gate" else r[kind][key] for r in runs]
                summary[f"{ds_name}/{kind}/{key}"] = (st.mean(vals), st.stdev(vals))
    (out / "branch_analysis.json").write_text(json.dumps({"runs": result, "summary_mean_sd": summary}, indent=2))
    for k, (m, sd) in summary.items():
        print(f"{k:40s} {m:.3f} +- {sd:.3f}")
    figure(rows, out / "fig_attention_maps.svg")


if __name__ == "__main__":
    main()
