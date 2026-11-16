"""Stage discriminators D_64, D_128, D_256 (AttnGAN layout).

Each produces an unconditional logit (real vs. fake) and a conditional logit
(image matches sentence s) from a shared 4x4 image code.
"""
import math

import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm


def _wrap(layer, norm):
    return spectral_norm(layer) if norm == "spectral" else layer


class StageDiscriminator(nn.Module):
    def __init__(self, res, ndf, sent_dim, norm="spectral"):
        super().__init__()
        layers, cin = [], 3
        n_down = int(math.log2(res // 4))
        for i in range(n_down):
            cout = ndf * min(2 ** i, 8)
            layers.append(_wrap(nn.Conv2d(cin, cout, 4, 2, 1, bias=False), norm))
            if norm == "batch" and i > 0:
                layers.append(nn.BatchNorm2d(cout))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            cin = cout
        self.encode = nn.Sequential(*layers)
        self.uncond = _wrap(nn.Conv2d(cin, 1, 4, 4), norm)
        self.joint = nn.Sequential(_wrap(nn.Conv2d(cin + sent_dim, cin, 3, 1, 1), norm),
                                   nn.LeakyReLU(0.2, inplace=True))
        self.cond = _wrap(nn.Conv2d(cin, 1, 4, 4), norm)

    def forward(self, x, sent):
        f = self.encode(x)
        u = self.uncond(f).flatten()
        s = sent[:, :, None, None].expand(-1, -1, f.size(2), f.size(3))
        c = self.cond(self.joint(torch.cat([f, s.to(f.dtype)], dim=1))).flatten()
        return u, c


class MultiScaleDiscriminator(nn.Module):
    def __init__(self, ndf, sent_dim, norm="spectral", resolutions=(64, 128, 256)):
        super().__init__()
        self.nets = nn.ModuleList([StageDiscriminator(r, ndf, sent_dim, norm)
                                   for r in resolutions])

    def __getitem__(self, i):
        return self.nets[i]

    def __len__(self):
        return len(self.nets)
