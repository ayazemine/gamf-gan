"""Evaluation metrics with fully specified implementations (R2-4, R3-4, R3-5).

* FID / KID / IS : torch-fidelity (Inception-v3 pool3 / logits), generated set
                   vs. the held-out real reference set (real_256/<split>).
* SSIM           : Wang et al. (2004): Gaussian window 11, sigma 1.5, K1=0.01,
                   K2=0.03, RGB in [0,1], mean over channels and pixels.
                   Reference = the real image whose caption was used.
* LPIPS          : lpips package v0.1 (AlexNet by default), inputs in [-1,1].
                   Paired with the caption's source image like SSIM.
* LPIPS-div      : mean pairwise LPIPS between samples of the same caption
                   (diversity; higher = more diverse).
* CLIP score     : cosine similarity of CLIP image/text embeddings (model id
                   stored in the output); CLIPScore = 100 * max(cos, 0)
                   (Hessel et al., 2021) is reported as well.
* R-precision    : top-1 retrieval of the true caption among 1 + (N-1) random
                   captions of other test images, with CLIP (CLIP-R) and with the
                   frozen DAMSM encoders (AttnGAN protocol).
"""
import itertools
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


# --------------------------------------------------------------------- FID etc.
def fidelity_metrics(gen_dir, real_dir, device, kid_subset_size=1000):
    import torch_fidelity
    n_gen = len(list(Path(gen_dir).glob("*.png")))
    n_real = len(list(Path(real_dir).glob("*.png")))
    m = torch_fidelity.calculate_metrics(
        input1=str(gen_dir), input2=str(real_dir), cuda=device.type == "cuda",
        fid=True, kid=True, isc=True, verbose=False,
        kid_subset_size=min(kid_subset_size, n_gen, n_real))
    return {"FID": m["frechet_inception_distance"],
            "KID": m["kernel_inception_distance_mean"],
            "KID_std": m["kernel_inception_distance_std"],
            "IS": m["inception_score_mean"], "IS_std": m["inception_score_std"],
            "n_generated": n_gen, "n_real": n_real,
            "torch_fidelity": torch_fidelity.__version__}


# ------------------------------------------------------------------------- SSIM
def _gaussian_window(size=11, sigma=1.5, channels=3):
    x = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-x ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    w = (g[:, None] @ g[None, :])[None, None]
    return w.expand(channels, 1, size, size).contiguous()


def ssim(x, y, data_range=1.0, size=11, sigma=1.5, k1=0.01, k2=0.03):
    """Per-image SSIM; x, y: B x 3 x H x W in [0, data_range]."""
    c = x.size(1)
    w = _gaussian_window(size, sigma, c).to(x)
    conv = lambda t: F.conv2d(t, w, groups=c)  # noqa: E731  (valid padding)
    mu_x, mu_y = conv(x), conv(y)
    sxx = conv(x * x) - mu_x ** 2
    syy = conv(y * y) - mu_y ** 2
    sxy = conv(x * y) - mu_x * mu_y
    c1, c2 = (k1 * data_range) ** 2, (k2 * data_range) ** 2
    s = ((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (sxx + syy + c2))
    return s.mean(dim=(1, 2, 3))


# ------------------------------------------------------------------------ LPIPS
class LPIPSMetric:
    def __init__(self, device, net="alex"):
        import lpips
        self.model = lpips.LPIPS(net=net, version="0.1", verbose=False).to(device).eval()
        self.version = f"lpips {lpips.__version__ if hasattr(lpips, '__version__') else ''} {net} v0.1"

    @torch.no_grad()
    def __call__(self, x, y):  # [-1, 1]
        return self.model(x, y).flatten()


def lpips_diversity(groups, load, metric, device):
    """groups: {key: [file, ...]} with >= 2 samples per caption."""
    vals = []
    for files in groups.values():
        if len(files) < 2:
            continue
        imgs = torch.stack([load(f) for f in files]).to(device)
        for a, b in itertools.combinations(range(len(files)), 2):
            vals.append(metric(imgs[a:a + 1], imgs[b:b + 1]).item())
    return float(np.mean(vals)) if vals else float("nan")


# ------------------------------------------------------------------------- CLIP
def _embeds(out):
    # transformers < 5 returns the projected embeddings; >= 5 returns a model output
    # whose pooler_output holds them
    return (out if torch.is_tensor(out) else out.pooler_output).float()


class CLIPMetric:
    def __init__(self, device, model_id="openai/clip-vit-base-patch32"):
        from transformers import CLIPModel, CLIPProcessor
        self.model = CLIPModel.from_pretrained(model_id).to(device).eval()
        self.proc = CLIPProcessor.from_pretrained(model_id)
        self.device, self.model_id = device, model_id

    @torch.no_grad()
    def image_features(self, pil_images):
        inp = self.proc(images=pil_images, return_tensors="pt").to(self.device)
        return F.normalize(_embeds(self.model.get_image_features(**inp)), dim=-1)

    @torch.no_grad()
    def text_features(self, texts):
        inp = self.proc(text=texts, return_tensors="pt", padding=True, truncation=True,
                        max_length=77).to(self.device)
        return F.normalize(_embeds(self.model.get_text_features(**inp)), dim=-1)


def r_precision(img_feats, txt_feats, true_idx, n_candidates, seed):
    """img_feats: N x D, txt_feats: M x D (unique captions), true_idx: N.
    Candidates for image n: its caption + (n_candidates-1) random captions
    drawn from captions of *other* images."""
    rng = random.Random(seed)
    m = txt_feats.size(0)
    hits = []
    for n in range(img_feats.size(0)):
        t = int(true_idx[n])
        others = rng.sample([k for k in range(m) if k != t], min(n_candidates - 1, m - 1))
        cand = torch.tensor([t] + others, device=txt_feats.device)
        scores = txt_feats[cand] @ img_feats[n]
        hits.append(float(scores.argmax().item() == 0))
    return 100.0 * float(np.mean(hits))


def group_by_caption(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[(r["image_id"], r["caption_idx"])].append(r["file"])
    return groups
