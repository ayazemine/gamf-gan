"""Generate test images from a trained run (EMA weights of the best checkpoint).

    python -m gamf.sample --run runs/cub/gamf/seed_0 [--split test] [--save-stages]
"""
import argparse
from pathlib import Path

import torch

from gamf.config import damsm_path, load_config
from gamf.data.dataset import T2IDataset
from gamf.generate import generate_split
from gamf.models import build_generator
from gamf.models.text_encoder import load_damsm
from gamf.utils import get_device, mark_done


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--ckpt", default="best.pt")
    p.add_argument("--save-stages", action="store_true",
                   help="also save the 64/128/256 px outputs of every stage")
    args = p.parse_args()
    run = Path(args.run)
    cfg = load_config([run / "config.yaml"])
    device = get_device()
    text_enc, _ = load_damsm(damsm_path(cfg), device)
    dim = cfg["text"]["hidden_dim"]
    G = build_generator(cfg, dim, dim).to(device)
    G.load_state_dict(torch.load(run / "ckpt" / args.ckpt, map_location=device,
                                 weights_only=False)["G_ema"])
    ds = T2IDataset(cfg, args.split, train=False, allow_test=True)
    e = cfg["eval"]
    n = generate_split(G, text_enc, ds, run / "samples" / args.split, cfg["model"]["z_dim"], device,
                       seed=e["seed"], samples_per_caption=e["samples_per_caption"],
                       captions=e["captions"], batch_size=e["batch_size"],
                       stage_dir=run / "stages" / args.split if args.save_stages else None)
    mark_done(run / f"samples_{args.split}_done.json", images=n, ckpt=args.ckpt)
    print(f"{n} images written to {run / 'samples' / args.split}")


if __name__ == "__main__":
    main()
