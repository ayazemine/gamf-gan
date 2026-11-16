"""Print the tensor shape and parameter count of every GAMF-GAN stage
(Markdown table for docs/ARCHITECTURE.md and the revised Fig. 4; R3-1).

    python -m gamf.analysis.shapes --config configs/base.yaml configs/models/gamf.yaml
"""
import argparse

import torch

from gamf.config import load_config
from gamf.models import build_discriminator, build_generator
from gamf.utils import count_params


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", nargs="+", required=True)
    p.add_argument("--batch", type=int, default=2)
    args = p.parse_args()
    cfg = load_config(args.config)
    m, d, T, b = cfg["model"], cfg["text"]["hidden_dim"], cfg["data"]["words_num"], args.batch
    G = build_generator(cfg, d, d).eval()
    D = build_discriminator(cfg, d)
    z, s = torch.randn(b, m["z_dim"]), torch.randn(b, d)
    e, pad = torch.randn(b, d, T), torch.zeros(b, T, dtype=torch.bool)
    with torch.no_grad():
        out = G(z, s, e, pad, return_features=True)
    if not hasattr(G, "types"):  # single-path generators (StyleGAN-T, DF-GAN)
        print(f"| {m['name']} | output {' / '.join(str(tuple(x.shape[1:])) for x in out['imgs'])} | "
              f"G {count_params(G):,} / D {count_params(D):,} |")
        return
    shp = lambda t: " x ".join(str(x) for x in t.shape[1:])  # noqa: E731
    print(f"| Item | Tensor (per sample) | Params |\n|---|---|---|")
    print(f"| z (noise) | {shp(z)} | |\n| s (sentence feature) | {shp(s)} | |\n| e (word features) | {shp(e)} | |")
    print(f"| c = CA(s) | {m['c_dim']} | {count_params(G.ca):,} |")
    print(f"| h0 = F0([z; c]) | {shp(out['feats'][0])} | {count_params(G.init):,} |")
    for i, (t, stage) in enumerate(zip(G.types, G.stages)):
        print(f"| h{i + 1} = stage {i + 1} ({t}) | {shp(out['feats'][i + 1])} | "
              f"{count_params(stage) + count_params(G.to_rgb[i]):,} |")
        print(f"| x{i + 1} (image) | {shp(out['imgs'][i])} | |")
    for i, net in enumerate(D.nets):
        print(f"| D_{out['imgs'][i].shape[-1]} | logits: 1 uncond. + 1 cond. | {count_params(net):,} |")
    print(f"| **Total G / D** | | {count_params(G):,} / {count_params(D):,} |")


if __name__ == "__main__":
    main()
