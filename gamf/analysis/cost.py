"""Computational cost of each model (R2-6, R3-6).

Parameters, GFLOPs per generated image (torch FlopCounterMode), inference
latency (median of timed runs, batch 1 and batch 16), peak inference memory,
and -- if runs exist -- training hours averaged over seeds.

    python -m gamf.analysis.cost --config configs/base.yaml configs/datasets/cub.yaml \
        --models configs/models/attngan.yaml configs/models/dmgan.yaml \
                 configs/models/stylegan_t.yaml configs/models/gamf.yaml
"""
import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from torch.utils.flop_counter import FlopCounterMode

from gamf.config import load_config
from gamf.models import build_discriminator, build_generator
from gamf.utils import count_params, get_device, hardware_info, write_json


def _inputs(cfg, b, device, T=18):
    d = cfg["text"]["hidden_dim"]
    words = torch.randn(b, d, T, device=device)
    pad = torch.zeros(b, T, dtype=torch.bool, device=device)
    return torch.randn(b, cfg["model"]["z_dim"], device=device), torch.randn(b, d, device=device), words, pad


@torch.no_grad()
def latency_ms(G, inputs, device, repeats=50, warmup=10):
    for _ in range(warmup):
        G(*inputs)
    times = []
    for _ in range(repeats):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter()
        G(*inputs)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t) * 1000)
    return statistics.median(times)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", nargs="+", required=True)
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--runs", default="runs")
    p.add_argument("--out", default="results/cost.json")
    args = p.parse_args()
    device = get_device()
    rows = []
    for mpath in args.models:
        cfg = load_config(list(args.config) + [mpath])
        d = cfg["text"]["hidden_dim"]
        # no parameter needs gradients here; also avoids a FlopCounterMode module-tracker assertion
        G = build_generator(cfg, d, d).to(device).eval().requires_grad_(False)
        D = build_discriminator(cfg, d)
        with FlopCounterMode(display=False) as fc:
            with torch.no_grad():
                G(*_inputs(cfg, 1, device))
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        lat1 = latency_ms(G, _inputs(cfg, 1, device), device)
        lat16 = latency_ms(G, _inputs(cfg, 16, device), device)
        row = {"model": cfg["model"]["name"], "params_G_M": count_params(G) / 1e6,
               "params_D_M": count_params(D) / 1e6,
               "GFLOPs_per_image": fc.get_total_flops() / 1e9,
               "latency_ms_b1": lat1, "latency_ms_per_image_b16": lat16 / 16,
               "peak_infer_mem_gb_b16": (torch.cuda.max_memory_allocated() / 1024 ** 3
                                         if device.type == "cuda" else None)}
        stats = [json.loads(f.read_text()) for f in
                 (Path(args.runs) / cfg["data"]["name"] / cfg["model"]["name"]).glob("seed_*/train_stats.json")]
        if stats:
            hrs = [s["train_hours"] for s in stats]
            row.update(train_hours_mean=statistics.mean(hrs), train_seeds=len(hrs),
                       epochs=stats[0]["epochs"], batch_size=stats[0]["batch_size"],
                       peak_train_mem_gb=max(s["peak_gpu_mem_gb"] or 0 for s in stats))
        rows.append(row)
        print(row)
    write_json({"hardware": hardware_info(), "rows": rows}, args.out)


if __name__ == "__main__":
    main()
