"""Deterministic image generation for a split (used by training-time validation
and by ``gamf.sample``).

File names follow ``<image_id>_c<caption_idx>_s<sample>.png`` so that any
model -- including external ones such as DF-GAN or GALIP run on
``captions_<split>.csv`` -- can be evaluated by ``gamf.evaluate``.
"""
from pathlib import Path

import torch
from PIL import Image

from gamf.utils import save_image_atomic


def to_pil(x):
    x = ((x.detach().float().clamp(-1, 1) + 1) * 127.5).round().byte()
    return Image.fromarray(x.permute(1, 2, 0).cpu().numpy())


@torch.no_grad()
def generate_split(G, text_enc, dataset, out_dir, z_dim, device, seed,
                   samples_per_caption=1, captions="eval", batch_size=32, stage_dir=None, upto=None):
    """captions='eval': the fixed evaluation caption of each image;
    captions='all': every caption of every image.

    Resumable: all batches are recomputed in the same order (so every image gets the
    same latent and noise as in an uninterrupted run), but images already on disk are
    not rewritten; images are written atomically."""
    G.eval()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if stage_dir:
        Path(stage_dir).mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)  # also seeds the per-pixel noise of style layers
    gen = torch.Generator().manual_seed(seed)

    jobs = []
    for rec in dataset.records:
        idxs = [rec["eval_caption_idx"]] if captions == "eval" else range(len(rec["captions"]))
        for ci in idxs:
            for k in range(samples_per_caption):
                jobs.append((rec["image_id"], ci, k, rec["captions"][ci]))

    for start in range(0, len(jobs), batch_size):
        chunk = jobs[start:start + batch_size]
        enc = [dataset.encode_caption(j[3]) for j in chunk]
        caps = torch.stack([e[0] for e in enc]).to(device)
        lens = torch.tensor([e[1] for e in enc])
        words, sent = text_enc(caps, lens)
        z = torch.randn(len(chunk), z_dim, generator=gen).to(device)
        out = G(z, sent, words, caps == 0, upto=upto)
        for j, (iid, ci, k, _) in enumerate(chunk):
            name = f"{iid}_c{ci}_s{k}.png"
            if not (out_dir / name).exists():
                save_image_atomic(to_pil(out["imgs"][-1][j]), out_dir / name)
            if stage_dir:
                for s, img in enumerate(out["imgs"]):
                    sp = Path(stage_dir) / f"{iid}_c{ci}_s{k}_stage{s + 1}.png"
                    if not sp.exists():
                        save_image_atomic(to_pil(img[j]), sp)
    return len(jobs)
