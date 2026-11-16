"""DF-GAN (Tao et al., CVPR 2022), re-implemented as a recent baseline (R2-5).

Single-stage generator with deep text-image fusion blocks and a one-way
conditional discriminator trained with the hinge loss and the matching-aware
zero-centred gradient penalty (MA-GP). Trained under the same data, epochs,
batch size and evaluation protocol as all other models.

Generator: z -> FC -> 8ngf x 4 x 4 -> 6 x [UpBlock: up x2, DFBlock, conv, DFBlock, conv,
residual] -> 256 x 256 RGB. Every DFBlock applies two text-conditioned channel-wise
affine transformations, gamma(c) * h + beta(c), with c = [z; s].
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Affine(nn.Module):
    def __init__(self, cond_dim, c):
        super().__init__()
        self.gamma = nn.Sequential(nn.Linear(cond_dim, c), nn.ReLU(inplace=True), nn.Linear(c, c))
        self.beta = nn.Sequential(nn.Linear(cond_dim, c), nn.ReLU(inplace=True), nn.Linear(c, c))
        nn.init.zeros_(self.gamma[2].weight)
        nn.init.ones_(self.gamma[2].bias)
        nn.init.zeros_(self.beta[2].weight)
        nn.init.zeros_(self.beta[2].bias)

    def forward(self, x, y):
        return self.gamma(y)[..., None, None] * x + self.beta(y)[..., None, None]


class DFBlock(nn.Module):
    def __init__(self, cond_dim, c):
        super().__init__()
        self.a1, self.a2 = Affine(cond_dim, c), Affine(cond_dim, c)

    def forward(self, x, y):
        h = F.leaky_relu(self.a1(x, y), 0.2)
        return F.leaky_relu(self.a2(h, y), 0.2)


class GBlock(nn.Module):
    def __init__(self, cond_dim, cin, cout):
        super().__init__()
        self.f1, self.f2 = DFBlock(cond_dim, cin), DFBlock(cond_dim, cout)
        self.c1, self.c2 = nn.Conv2d(cin, cout, 3, 1, 1), nn.Conv2d(cout, cout, 3, 1, 1)
        self.sc = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, y):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        h = self.c2(self.f2(self.c1(self.f1(x, y)), y))
        return self.sc(x) + h


class DFGenerator(nn.Module):
    MULT = (8, 8, 8, 8, 4, 2, 1)

    def __init__(self, m, word_dim, sent_dim):
        super().__init__()
        ngf, z_dim = m["dfgan_ngf"], m["z_dim"]
        cond = z_dim + sent_dim
        self.c0 = ngf * self.MULT[0]
        self.fc = nn.Linear(z_dim, self.c0 * 4 * 4)
        self.blocks = nn.ModuleList([GBlock(cond, ngf * a, ngf * b)
                                     for a, b in zip(self.MULT[:-1], self.MULT[1:])])
        self.to_rgb = nn.Sequential(nn.LeakyReLU(0.2), nn.Conv2d(ngf, 3, 3, 1, 1), nn.Tanh())

    def forward(self, z, sent, words=None, pad_mask=None, return_features=False, upto=None):
        y = torch.cat([z, sent], dim=1)
        h = self.fc(z).view(z.size(0), self.c0, 4, 4)
        for block in self.blocks:
            h = block(h, y)
        return {"imgs": [self.to_rgb(h)], "mu": None, "logvar": None}


class DBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.res = nn.Sequential(nn.Conv2d(cin, cout, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, inplace=True),
                                 nn.Conv2d(cout, cout, 3, 1, 1, bias=False), nn.LeakyReLU(0.2, inplace=True))
        self.sc = nn.Conv2d(cin, cout, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return F.avg_pool2d(self.sc(x), 2) + self.gamma * self.res(x)


class DFDiscriminator(nn.Module):
    """One-way conditional discriminator: D(x, s) -> one logit per sample."""

    def __init__(self, ndf, sent_dim, res=256):
        super().__init__()
        mult = [1, 2, 4, 8, 16, 16, 16]
        n = int(math.log2(res // 4))
        self.conv_img = nn.Conv2d(3, ndf, 3, 1, 1)
        self.blocks = nn.Sequential(*[DBlock(ndf * mult[i], ndf * mult[i + 1]) for i in range(n)])
        c = ndf * mult[n]
        self.head = nn.Sequential(nn.Conv2d(c + sent_dim, ndf * 2, 3, 1, 1, bias=False),
                                  nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(ndf * 2, 1, 4, 1, 0, bias=False))

    def forward(self, x, sent):
        h = self.blocks(self.conv_img(x))
        s = sent[:, :, None, None].expand(-1, -1, h.size(2), h.size(3)).to(h.dtype)
        return self.head(torch.cat([h, s], dim=1)).flatten()
