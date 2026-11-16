"""All loss terms of Eq. (20), written out explicitly (reviewer R3-2).

L_total = lambda_adv  L_adv          (sum over the three stage discriminators)
        + lambda_kl   L_KL           (conditioning augmentation; AttnGAN uses 1)
        + lambda_damsm L_DAMSM(x3)   (word + sentence matching on the final image)
        + lambda_mem  L_mem          (||phi(x_real) - phi(x_2)||^2, frozen DAMSM image features; optional, not used in the study)
        + lambda_perc L_perc         (VGG-16 feature distance to the paired real image)
        + lambda_style L_style       (Gram-matrix distance to the paired real image)

Module update table
-------------------
* Generator (CA, F0, G_Attn, G_DM, G_Style incl. mapping): all terms above.
* Discriminators: L_D (below) only.
* DAMSM text/image encoders, Inception-v3, VGG-16: frozen, never updated.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------- adversarial
def _bce(logits, target):
    return F.binary_cross_entropy_with_logits(logits.float(), torch.full_like(logits.float(), target))


def d_stage_loss(d, real, fake, wrong, sent):
    """AttnGAN discriminator loss for one stage.

    L_D = 1/2 [BCE(D_u(x),1) + BCE(D_c(x,s),1)]
        + 1/3 [BCE(D_u(G),0) + BCE(D_c(G,s),0) + BCE(D_c(x_wrong,s),0)]
    x_wrong is a real image paired with a non-matching sentence.
    """
    ru, rc = d(real, sent)
    fu, fc = d(fake, sent)
    _, wc = d(wrong, sent)
    return (_bce(ru, 1.0) + _bce(rc, 1.0)) / 2 + (_bce(fu, 0.0) + _bce(fc, 0.0) + _bce(wc, 0.0)) / 3


def g_stage_loss(d, fake, sent):
    """Non-saturating generator loss: BCE(D_u(G),1) + BCE(D_c(G,s),1)."""
    fu, fc = d(fake, sent)
    return _bce(fu, 1.0) + _bce(fc, 1.0)


def r1_penalty(d, real, sent):
    real = real.detach().requires_grad_(True)
    u, c = d(real, sent)
    grad, = torch.autograd.grad((u.sum() + c.sum()).float(), real, create_graph=True)
    return grad.pow(2).flatten(1).sum(1).mean()


def kl_loss(mu, logvar):
    """KL(N(mu, sigma^2) || N(0, I)) of conditioning augmentation."""
    if mu is None:
        return torch.zeros(())
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


def memory_feature_loss(f_orig, f_mem):
    """L_mem = ||F_orig - F_mem||^2 (optional auxiliary loss), as a mean squared error.

    Default (``loss.mem_type: real_feature``): F_orig = phi(x_real), F_mem = phi(x_2), where
    phi are the region features of the frozen DAMSM image encoder (Inception-v3 Mixed_6e,
    17 x 17 x 256), x_real is the source image of the caption at 128 px and x_2 the output
    of the dynamic-memory stage. The memory-refined image is thus pulled towards the
    feature representation of the original image, i.e. towards correct details.
    """
    return F.mse_loss(f_mem.float(), f_orig.float().detach())


# --------------------------------------------------------------------------- DAMSM
def _contrastive(sims, class_ids):
    """Symmetric cross-entropy over a B x B image-caption score matrix.

    Off-diagonal pairs from the same class are masked (AttnGAN), since their
    captions may legitimately describe the image.
    """
    b = sims.size(0)
    if class_ids is not None:
        same = class_ids[:, None] == class_ids[None, :]
        same &= ~torch.eye(b, dtype=torch.bool, device=sims.device)
        sims = sims.masked_fill(same, float("-inf"))
    labels = torch.arange(b, device=sims.device)
    return F.cross_entropy(sims, labels) + F.cross_entropy(sims.t(), labels)


def word_scores(img_local, words, cap_lens, gamma1, gamma2):
    """R(Q_j, D_i) of AttnGAN for all images j and captions i -> B x B."""
    b = img_local.size(0)
    context = img_local.flatten(2)                                  # B x D x N
    cols = []
    for i in range(b):
        n_words = int(cap_lens[i])
        w = words[i:i + 1, :, :n_words].expand(b, -1, -1)           # B x D x L
        attn = torch.bmm(context.transpose(1, 2), w).softmax(dim=2)  # normalise over words
        attn = (attn * gamma1).softmax(dim=1)                       # then over regions
        region_ctx = torch.bmm(context, attn)                       # B x D x L
        rel = F.cosine_similarity(w, region_ctx, dim=1)             # B x L
        cols.append(torch.logsumexp(rel * gamma2, dim=1))           # log sum_i exp(g2 R)
    return torch.stack(cols, dim=1)


def damsm_word_loss(img_local, words, cap_lens, class_ids, g1, g2, g3):
    return _contrastive(word_scores(img_local.float(), words.float(), cap_lens, g1, g2) * g3,
                        class_ids)


def damsm_sent_loss(img_global, sent, class_ids, g3):
    sims = F.normalize(img_global.float(), dim=1) @ F.normalize(sent.float(), dim=1).t()
    return _contrastive(sims * g3, class_ids)


# --------------------------------------------------------------- perceptual/style
VGG_LAYERS = {"relu1_2": 3, "relu2_2": 8, "relu3_3": 15, "relu4_3": 22}


class VGGLoss(nn.Module):
    """Perceptual (feature MSE) and style (Gram MSE) losses on frozen VGG-16.

    phi_l: ReLU activations of ImageNet VGG-16 at the listed layers; images are
    mapped from [-1, 1] to ImageNet normalisation. Gram matrices are divided by
    C*H*W.
    """

    def __init__(self, perc_layers, style_layers, pretrained=True):
        super().__init__()
        from torchvision.models import VGG16_Weights, vgg16
        self.perc_layers, self.style_layers = list(perc_layers), list(style_layers)
        last = max(VGG_LAYERS[n] for n in self.perc_layers + self.style_layers)
        weights = VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        self.features = vgg16(weights=weights).features[:last + 1].eval()
        for p in self.features.parameters():
            p.requires_grad_(False)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.idx = {VGG_LAYERS[n]: n for n in set(self.perc_layers + self.style_layers)}

    def train(self, mode=True):
        super().train(mode)
        self.features.eval()
        return self

    def _feats(self, x):
        x = ((x.float() + 1) / 2 - self.mean) / self.std
        out = {}
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.idx:
                out[self.idx[i]] = x
        return out

    @staticmethod
    def gram(f):
        b, c, h, w = f.shape
        f = f.flatten(2)
        return f @ f.transpose(1, 2) / (c * h * w)

    def forward(self, fake, real):
        ff = self._feats(fake)
        with torch.no_grad():
            fr = self._feats(real)
        perc = sum(F.mse_loss(ff[n], fr[n]) for n in self.perc_layers)
        style = sum(F.mse_loss(self.gram(ff[n]), self.gram(fr[n])) for n in self.style_layers)
        return perc, style


# ------------------------------------------------------------------ DF-GAN losses
def dfgan_d_loss(d, real, fake, sent):
    """Hinge loss with mismatched pairs (Tao et al., 2022):
    L_D = E[max(0, 1 - D(x, s))] + 1/2 E[max(0, 1 + D(G, s))] + 1/2 E[max(0, 1 + D(x, s_mis))]."""
    real_out = d(real, sent).float()
    fake_out = d(fake, sent).float()
    mis_out = d(real, sent.roll(1, dims=0)).float()
    return F.relu(1 - real_out).mean() + (F.relu(1 + fake_out).mean() + F.relu(1 + mis_out).mean()) / 2


def dfgan_g_loss(d, fake, sent):
    return -d(fake, sent).float().mean()


def ma_gp(d, real, sent, k=2.0, p=6.0):
    """Matching-aware zero-centred gradient penalty on (real image, matching sentence):
    k * E[ (||grad_x D||^2 + ||grad_s D||^2)^(p/2) ]."""
    real = real.detach().float().requires_grad_(True)
    sent = sent.detach().float().requires_grad_(True)
    out = d(real, sent)
    gx, gs = torch.autograd.grad(out.sum(), (real, sent), create_graph=True)
    norm = torch.cat([gx.flatten(1), gs.flatten(1)], dim=1).pow(2).sum(1).sqrt()
    return k * norm.pow(p).mean()
