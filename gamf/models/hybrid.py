"""Proposed single-stage hybrid text-to-image GAN.

Combines AttnGAN word attention and DM-GAN dynamic memory inside one jointly trained
DF-GAN-style generator, with the DAMSM loss active for the whole run, and a discriminator
that adds a frozen CLIP-ViT feature branch to the DF-GAN discriminator.

Generator: z -> FC -> 8ngf x 4 x 4 -> 6 x GBlock (DF-GAN deep fusion, up x2) -> 256 px RGB.
After the 32, 64 and 128 px blocks a word-attention fusion (AttnGAN) is added, after the
64 px block a dynamic-memory refinement (DM-GAN). Each is a residual branch with a learned
gate initialised at 0, so training starts from the plain DF-GAN generator.

Discriminator: DF-GAN one-way discriminator, optionally plus a branch on frozen CLIP-ViT
features (as in GALIP / Projected GAN). The CLIP model used here (ViT-B/16) differs from the
one used for the CLIPScore / CLIP-R metrics (ViT-B/32). Logits of both branches are summed;
the matching-aware gradient penalty is applied to each branch (image input for the DF
branch, CLIP-feature input for the CLIP branch).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from gamf.models.dfgan import DFDiscriminator, GBlock
from gamf.models.generator import DynamicMemory, WordAttention


class WordFusion(nn.Module):
    """h + g * conv([h; ctx]), ctx = AttnGAN word context, g initialised at 0."""

    def __init__(self, c, word_dim):
        super().__init__()
        self.attn = WordAttention(c, word_dim)
        self.fuse = nn.Sequential(nn.Conv2d(2 * c, c, 3, 1, 1), nn.LeakyReLU(0.2, inplace=True),
                                  nn.Conv2d(c, c, 3, 1, 1))
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, h, words, pad_mask):
        ctx, _ = self.attn(h, words.to(h.dtype), pad_mask)
        return h + self.gate * self.fuse(torch.cat([h, ctx], dim=1))


class MemoryFusion(nn.Module):
    """h + g * (DM(h) - h), DM = DM-GAN dynamic memory, g initialised at 0."""

    def __init__(self, c, word_dim):
        super().__init__()
        self.memory = DynamicMemory(c, word_dim)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, h, words, pad_mask):
        return h + self.gate * (self.memory(h, words.to(h.dtype), pad_mask) - h)


class HybridGenerator(nn.Module):
    MULT = (8, 8, 8, 8, 4, 2, 1)   # block outputs: 8, 16, 32, 64, 128, 256 px

    def __init__(self, m, word_dim, sent_dim):
        super().__init__()
        ngf, z_dim = m["dfgan_ngf"], m["z_dim"]
        cond = z_dim + sent_dim
        self.c0 = ngf * self.MULT[0]
        self.fc = nn.Linear(z_dim, self.c0 * 4 * 4)
        self.blocks = nn.ModuleList([GBlock(cond, ngf * a, ngf * b)
                                     for a, b in zip(self.MULT[:-1], self.MULT[1:])])
        res = [8 * 2 ** i for i in range(len(self.blocks))]
        attn_res, mem_res = set(m.get("hybrid_attn_res", [32, 64, 128])), set(m.get("hybrid_mem_res", [64]))
        self.attn = nn.ModuleDict({str(i): WordFusion(ngf * self.MULT[i + 1], word_dim)
                                   for i, r in enumerate(res) if r in attn_res})
        self.mem = nn.ModuleDict({str(i): MemoryFusion(ngf * self.MULT[i + 1], word_dim)
                                  for i, r in enumerate(res) if r in mem_res})
        self.to_rgb = nn.Sequential(nn.LeakyReLU(0.2), nn.Conv2d(ngf, 3, 3, 1, 1), nn.Tanh())

    def forward(self, z, sent, words=None, pad_mask=None, return_features=False, upto=None):
        y = torch.cat([z, sent], dim=1)
        h = self.fc(z).view(z.size(0), self.c0, 4, 4)
        for i, block in enumerate(self.blocks):
            h = block(h, y)
            if str(i) in self.attn:
                h = self.attn[str(i)](h, words, pad_mask)
            if str(i) in self.mem:
                h = self.mem[str(i)](h, words, pad_mask)
        return {"imgs": [self.to_rgb(h)], "mu": None, "logvar": None}


class CLIPBranch(nn.Module):
    """Frozen CLIP-ViT feature extractor + trainable conditional head."""

    MEAN = (0.48145466, 0.4578275, 0.40821073)
    STD = (0.26862954, 0.26130258, 0.27577711)

    def __init__(self, model_id, layers, sent_dim, width=256):
        super().__init__()
        from transformers import CLIPVisionModel
        vit = CLIPVisionModel.from_pretrained(model_id, attn_implementation="eager")
        vit.eval().requires_grad_(False)
        self.vit = [vit]                        # not a submodule: frozen, not in state_dict
        self.layers = list(layers)
        self.size = vit.config.image_size
        self.grid = self.size // vit.config.patch_size
        d = vit.config.hidden_size * len(self.layers)
        self.register_buffer("mean", torch.tensor(self.MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(self.STD).view(1, 3, 1, 1), persistent=False)
        self.img = nn.Sequential(nn.Conv2d(d, 2 * width, 1), nn.LeakyReLU(0.2, inplace=True),
                                 nn.Conv2d(2 * width, width, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True))
        self.joint = nn.Sequential(nn.Conv2d(width + sent_dim, width, 3, 1, 1), nn.LeakyReLU(0.2, inplace=True),
                                   nn.Conv2d(width, 1, (self.grid + 1) // 2, 1, 0))

    def _apply(self, fn, *args, **kwargs):      # keep the frozen ViT on the same device
        super()._apply(fn, *args, **kwargs)
        self.vit[0]._apply(fn, *args, **kwargs)
        return self

    def features(self, x):
        x = F.interpolate((x.float() + 1) / 2, size=(self.size, self.size), mode="bicubic",
                          align_corners=False, antialias=True)
        x = (x - self.mean) / self.std
        with torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=x.is_cuda):
            hs = self.vit[0](pixel_values=x, output_hidden_states=True).hidden_states
        b = x.size(0)
        f = [hs[i][:, 1:].transpose(1, 2).reshape(b, -1, self.grid, self.grid) for i in self.layers]
        return torch.cat(f, dim=1).float()

    def head(self, f, sent):
        h = self.img(f.to(self.img[0].weight.dtype))
        s = sent[:, :, None, None].expand(-1, -1, h.size(2), h.size(3)).to(h.dtype)
        return self.joint(torch.cat([h, s], dim=1)).flatten()

    def forward(self, x, sent):
        return self.head(self.features(x), sent)


class HybridDiscriminator(nn.Module):
    def __init__(self, m, sent_dim):
        super().__init__()
        self.df = DFDiscriminator(m["dfgan_ndf"], sent_dim)
        clip_id = m.get("hybrid_clip")
        self.clip = (CLIPBranch(clip_id, m.get("hybrid_clip_layers", [4, 8, 12]), sent_dim)
                     if clip_id else None)

    def forward(self, x, sent):
        out = self.df(x, sent)
        if self.clip is not None:
            out = out + self.clip(x, sent)
        return out

    def ma_gp(self, real, sent, k=2.0, p=6.0, real_feats=None):
        """Matching-aware zero-centred gradient penalty, per branch."""
        from gamf.losses import ma_gp
        gp = ma_gp(self.df, real, sent, k, p)
        if self.clip is not None:
            if real_feats is None:
                with torch.no_grad():
                    real_feats = self.clip.features(real)
            gp = gp + ma_gp(self.clip.head, real_feats, sent, k, p)
        return gp

    def d_step(self, real, fake, sent, dtype):
        """Discriminator hinge loss (DF-GAN, with mismatched pairs) and MA-GP. The frozen CLIP
        features of the real and the (detached) fake batch are computed once and shared by the
        real / mismatched / fake terms and the penalty."""
        from gamf.losses import dfgan_d_loss
        with torch.autocast(real.device.type, dtype=dtype, enabled=dtype is not None):
            loss = dfgan_d_loss(self.df, real, fake, sent)
        gp = self.ma_gp(real, sent) if self.clip is None else None
        if self.clip is not None:
            with torch.no_grad():
                fr, ff = self.clip.features(real), self.clip.features(fake)
            head = self.clip.head
            loss = loss + (F.relu(1 - head(fr, sent).float()).mean()
                           + (F.relu(1 + head(ff, sent).float()).mean()
                              + F.relu(1 + head(fr, sent.roll(1, dims=0)).float()).mean()) / 2)
            gp = self.ma_gp(real, sent, real_feats=fr)
        return loss, gp
