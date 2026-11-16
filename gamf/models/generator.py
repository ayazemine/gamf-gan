"""Cascade generators of the baselines (and of the obsolete stage-wise model).
GAMF-GAN itself is the single-stage generator in ``models/hybrid.py``.

Cascade (Eqs. (17)-(19) of the manuscript)
------------------------------------------
    c, mu, logvar = CA(s)                                   conditioning augmentation
    h0 = F0([z; c])                     B x ngf x 64 x 64
    h1, x1 = G_Attn(h0, e)              B x ngf x 64 x 64,   x1: B x 3 x 64 x 64
    h2, x2 = G_DM(h1, e)                B x ngf x 128 x 128, x2: B x 3 x 128 x 128
    h3, x3 = G_Style(h2, x2, w)         B x Cs x 256 x 256,  x3: B x 3 x 256 x 256
    w = f_map([z; s])                   text-conditioned StyleGAN mapping network

e: word features (B x D x T), s: sentence feature (B x D) from the frozen DAMSM
text encoder. Feature maps (not only images) are passed between stages; the
style stage receives h2 in place of StyleGAN's learned constant input and adds
a residual RGB refinement to the up-sampled x2 (skip architecture).

``model.stages`` selects the component of each stage, which yields the
baselines and the ablation variants from the same code:

    stage 1 (64 px, same resolution):  none | attn | mem | plain
    stage 2 (64 -> 128):               attn | mem  | plain
    stage 3 (128 -> 256):              attn | mem  | plain | style

    stage-wise (old)   [attn, mem,  style]
    AttnGAN (re-impl.) [attn, attn, attn ]
    DM-GAN  (re-impl.) [none, mem,  mem  ]
    w/o attention      [plain, mem, style]   etc.

``plain`` is a capacity-comparable residual refinement without text conditioning.
"""
import math

import torch
import torch.nn as nn

from gamf.models.layers import (ConditioningAugmentation, GLU, ResBlock, ToRGB,
                                 join_block, pixel_norm, up_block, upsample2x)


# ----------------------------------------------------------------------------- F0
class InitStage(nn.Module):
    """F0: [z; c] -> h0 (B x ngf x 64 x 64)."""

    def __init__(self, in_dim, ngf):
        super().__init__()
        c = ngf * 16
        self.c = c
        self.fc = nn.Sequential(nn.Linear(in_dim, c * 4 * 4 * 2, bias=False),
                                nn.BatchNorm1d(c * 4 * 4 * 2), GLU())
        self.up = nn.Sequential(up_block(c, c // 2), up_block(c // 2, c // 4),
                                up_block(c // 4, c // 8), up_block(c // 8, c // 16))

    def forward(self, zc):
        return self.up(self.fc(zc).view(-1, self.c, 4, 4))


# ------------------------------------------------------------------ AttnGAN part
class WordAttention(nn.Module):
    """Word-level attention of AttnGAN: each pixel attends over the words.

    e' = U e (1x1 projection to the image-feature space),
    beta_{j,i} = softmax_i(h_j^T e'_i),   ctx_j = sum_i beta_{j,i} e'_i.
    """

    def __init__(self, c, word_dim):
        super().__init__()
        self.proj = nn.Conv1d(word_dim, c, 1, bias=False)

    def forward(self, h, words, pad_mask):
        b, c, hh, ww = h.shape
        e = self.proj(words)                                     # B x C x T
        q = h.flatten(2).transpose(1, 2)                         # B x HW x C
        attn = torch.bmm(q, e).masked_fill(pad_mask[:, None, :], float("-inf"))
        attn = attn.softmax(dim=-1)                              # B x HW x T
        ctx = torch.bmm(e, attn.transpose(1, 2))                 # B x C x HW
        return ctx.view(b, c, hh, ww), attn


class AttnStage(nn.Module):
    def __init__(self, c, word_dim, upsample):
        super().__init__()
        self.attn = WordAttention(c, word_dim)
        self.res = nn.Sequential(ResBlock(c * 2), ResBlock(c * 2))
        self.out = up_block(c * 2, c) if upsample else join_block(c * 2, c)

    def forward(self, h, words, pad_mask):
        ctx, _ = self.attn(h, words, pad_mask)
        return self.out(self.res(torch.cat([h, ctx], dim=1)))


# ------------------------------------------------------------------- DM-GAN part
class DynamicMemory(nn.Module):
    """Dynamic memory of DM-GAN (Zhu et al., CVPR 2019).

    Memory writing:   g_i^w = sigma(A w_i + B h_bar)
                      m_i   = M_w(w_i) * g_i^w + M_r(h_bar) * (1 - g_i^w)
    Key addressing:   alpha_{j,i} = softmax_i(h_j^T phi_K(m_i))
    Value reading:    o_j = sum_i alpha_{j,i} phi_V(m_i)
    Response gate:    g_j = sigma(W_r [o_j; h_j] + b_r)
                      h'_j = o_j * g_j + h_j * (1 - g_j)
    """

    def __init__(self, c, word_dim, mem_dim=None):
        super().__init__()
        m = mem_dim or c
        self.gate_word = nn.Conv1d(word_dim, 1, 1, bias=False)
        self.gate_img = nn.Linear(c, 1)
        self.mem_word = nn.Conv1d(word_dim, m, 1, bias=False)
        self.mem_img = nn.Linear(c, m, bias=False)
        self.key = nn.Conv1d(m, c, 1, bias=False)
        self.value = nn.Conv1d(m, c, 1, bias=False)
        self.response = nn.Conv2d(2 * c, 1, 1)

    def forward(self, h, words, pad_mask):
        h_bar = h.mean(dim=(2, 3))                                            # B x C
        g_w = torch.sigmoid(self.gate_word(words) + self.gate_img(h_bar)[:, :, None])
        mem = self.mem_word(words) * g_w + self.mem_img(h_bar)[:, :, None] * (1 - g_w)
        k, v = self.key(mem), self.value(mem)                                  # B x C x T
        q = h.flatten(2).transpose(1, 2)                                       # B x HW x C
        alpha = torch.bmm(q, k).masked_fill(pad_mask[:, None, :], float("-inf")).softmax(-1)
        o = torch.bmm(v, alpha.transpose(1, 2)).view_as(h)
        g = torch.sigmoid(self.response(torch.cat([o, h], dim=1)))
        return o * g + h * (1 - g)


class MemStage(nn.Module):
    def __init__(self, c, word_dim, upsample):
        super().__init__()
        self.memory = DynamicMemory(c, word_dim)
        self.res = nn.Sequential(ResBlock(c), ResBlock(c))
        self.out = up_block(c, c) if upsample else join_block(c, c)

    def forward(self, h, words, pad_mask):
        return self.out(self.res(self.memory(h, words, pad_mask)))


class PlainStage(nn.Module):
    """Ablation stage: residual refinement without any text conditioning."""

    def __init__(self, c, upsample):
        super().__init__()
        self.res = nn.Sequential(ResBlock(c), ResBlock(c))
        self.out = up_block(c, c) if upsample else join_block(c, c)

    def forward(self, h, words=None, pad_mask=None):
        return self.out(self.res(h))


class IdentityStage(nn.Module):
    def forward(self, h, words=None, pad_mask=None):
        return h


# ----------------------------------------------------------------- StyleGAN part
class MappingNetwork(nn.Module):
    """Text-conditioned mapping network: w = f([norm(z); norm(E s)])."""

    def __init__(self, z_dim, sent_dim, w_dim, n_layers):
        super().__init__()
        self.embed = nn.Linear(sent_dim, z_dim)
        layers, d = [], 2 * z_dim
        for _ in range(n_layers):
            layers += [nn.Linear(d, w_dim), nn.LeakyReLU(0.2)]
            d = w_dim
        self.net = nn.Sequential(*layers)

    def forward(self, z, sent):
        return self.net(torch.cat([pixel_norm(z), pixel_norm(self.embed(sent))], dim=1))


class AdaIN(nn.Module):
    """AdaIN(x, y) = y_s * (x - mu(x)) / sigma(x) + y_b,  (y_s, y_b) = A(w)."""

    def __init__(self, c, w_dim):
        super().__init__()
        self.norm = nn.InstanceNorm2d(c)
        self.affine = nn.Linear(w_dim, 2 * c)
        nn.init.zeros_(self.affine.weight)
        with torch.no_grad():
            self.affine.bias[:c].fill_(1.0)
            self.affine.bias[c:].zero_()

    def forward(self, x, w):
        ys, yb = self.affine(w).chunk(2, dim=1)
        return ys[..., None, None] * self.norm(x) + yb[..., None, None]


class NoiseInjection(nn.Module):
    """x + b * n,  n ~ N(0, I) per pixel, b learned per channel."""

    def __init__(self, c):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, x):
        noise = torch.randn(x.size(0), 1, x.size(2), x.size(3), device=x.device, dtype=x.dtype)
        return x + self.weight * noise


class StyleLayer(nn.Module):
    """conv3x3 -> noise -> LeakyReLU -> AdaIN(w) (StyleGAN synthesis layer)."""

    def __init__(self, cin, cout, w_dim):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, 1, 1)
        self.noise = NoiseInjection(cout)
        self.act = nn.LeakyReLU(0.2)
        self.adain = AdaIN(cout, w_dim)

    def forward(self, x, w):
        return self.adain(self.act(self.noise(self.conv(x))), w)


class StyleStage(nn.Module):
    """G_Style: (h2, x2, w) -> (h3, x3) at twice the resolution.

    h2 replaces StyleGAN's constant input (1x1 projection to Cs channels); two
    style layers refine it at the input resolution, two more after bilinear
    upsampling. RGB outputs follow StyleGAN2's skip architecture, starting from
    the image of the previous stage: x3 = up(x2 + rgb_lo(.)) + rgb_hi(.).
    """

    def __init__(self, c_in, c_style, w_dim, z_dim, sent_dim, mapping_layers):
        super().__init__()
        self.mapping = MappingNetwork(z_dim, sent_dim, w_dim, mapping_layers)
        self.inp = nn.Conv2d(c_in, c_style, 1)
        self.lo = nn.ModuleList([StyleLayer(c_style, c_style, w_dim) for _ in range(2)])
        self.hi = nn.ModuleList([StyleLayer(c_style, c_style, w_dim) for _ in range(2)])
        self.rgb_lo = nn.Conv2d(c_style, 3, 1)
        self.rgb_hi = nn.Conv2d(c_style, 3, 1)

    def forward(self, h, x_prev, z, sent):
        w = self.mapping(z, sent)
        y = self.inp(h)
        for layer in self.lo:
            y = layer(y, w)
        rgb = x_prev + self.rgb_lo(y)
        y = upsample2x(y)
        for layer in self.hi:
            y = layer(y, w)
        return y, upsample2x(rgb) + self.rgb_hi(y)


# --------------------------------------------------------------------- generators
STAGE_CHOICES = (("none", "attn", "mem", "plain"),
                 ("attn", "mem", "plain"),
                 ("attn", "mem", "plain", "style"))


class CascadeGenerator(nn.Module):
    def __init__(self, m, word_dim, sent_dim):
        super().__init__()
        types = list(m["stages"])
        if len(types) != 3 or any(t not in ok for t, ok in zip(types, STAGE_CHOICES)):
            raise ValueError(f"model.stages={types} is not valid; choices: {STAGE_CHOICES}")
        ngf = m["ngf"]
        self.types = types
        self.ca = ConditioningAugmentation(sent_dim, m["c_dim"])
        self.init = InitStage(m["z_dim"] + m["c_dim"], ngf)
        self.stages, self.to_rgb = nn.ModuleList(), nn.ModuleList()
        for i, t in enumerate(types):
            up = i > 0
            if t == "none":
                stage = IdentityStage()
            elif t == "attn":
                stage = AttnStage(ngf, word_dim, up)
            elif t == "mem":
                stage = MemStage(ngf, word_dim, up)
            elif t == "plain":
                stage = PlainStage(ngf, up)
            else:
                stage = StyleStage(ngf, m["style_channels"], m["w_dim"], m["z_dim"],
                                   sent_dim, m["mapping_layers"])
            self.stages.append(stage)
            self.to_rgb.append(nn.Identity() if t == "style" else ToRGB(ngf))

    def forward(self, z, sent, words, pad_mask, return_features=False, upto=None):
        """upto=k runs stages 0..k only (stage-wise training of stage k).

        ``mem_pairs[i]`` holds (F_orig, F_mem) of a dynamic-memory stage i: the
        feature map entering the memory module and the memory-refined one,
        used by the optional memory loss L_mem = ||F_orig - F_mem||^2.
        """
        c, mu, logvar = self.ca(sent)
        h = self.init(torch.cat([z, c], dim=1))
        imgs, feats, x, mem_pairs = [], [h], None, {}
        last = len(self.types) - 1 if upto is None else upto
        for i, (t, stage, to_rgb) in enumerate(zip(self.types, self.stages, self.to_rgb)):
            if i > last:
                break
            if t == "style":
                h, x = stage(h, x, z, sent)
            elif t == "mem":
                h_mem = stage.memory(h, words, pad_mask)
                mem_pairs[i] = (h, h_mem)
                h = stage.out(stage.res(h_mem))
                x = to_rgb(h)
            else:
                h = stage(h, words, pad_mask)
                x = to_rgb(h)
            imgs.append(x)
            feats.append(h)
        out = {"imgs": imgs, "mu": mu, "logvar": logvar, "mem_pairs": mem_pairs}
        if return_features:
            out["feats"] = feats
        return out


class TextStyleGenerator(nn.Module):
    """Text-conditional StyleGAN baseline (reviewer R3-5).

    Standard StyleGAN is unconditional; here the sentence feature enters the
    mapping network, w = f([z; s]), and everything else follows StyleGAN:
    learned 4x4 constant, AdaIN style layers with noise injection from 4 to
    256 px, skip RGB outputs. Images at 64/128/256 px feed the same three
    discriminators as the cascade models.
    """

    CHANNELS = {4: 256, 8: 256, 16: 256, 32: 256, 64: 128, 128: 64, 256: 32}

    def __init__(self, m, word_dim, sent_dim):
        super().__init__()
        ch = self.CHANNELS
        w_dim = m["w_dim"]
        self.mapping = MappingNetwork(m["z_dim"], sent_dim, w_dim, m["mapping_layers"])
        self.const = nn.Parameter(torch.randn(1, ch[4], 4, 4))
        self.res = [2 ** k for k in range(2, int(math.log2(256)) + 1)]
        self.blocks, self.rgbs = nn.ModuleList(), nn.ModuleList()
        prev = ch[4]
        for r in self.res:
            self.blocks.append(nn.ModuleList([StyleLayer(prev, ch[r], w_dim),
                                              StyleLayer(ch[r], ch[r], w_dim)]))
            self.rgbs.append(nn.Conv2d(ch[r], 3, 1))
            prev = ch[r]

    def forward(self, z, sent, words=None, pad_mask=None, return_features=False, upto=None):
        w = self.mapping(z, sent)
        x = self.const.expand(z.size(0), -1, -1, -1)
        rgb, imgs = None, []
        for r, block, to_rgb in zip(self.res, self.blocks, self.rgbs):
            if r > 4:
                x = upsample2x(x)
            for layer in block:
                x = layer(x, w)
            y = to_rgb(x)
            rgb = y if rgb is None else upsample2x(rgb) + y
            if r in (64, 128, 256):
                imgs.append(rgb)
        return {"imgs": imgs, "mu": None, "logvar": None}
