"""Building blocks shared by the generator stages."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GLU(nn.Module):
    def forward(self, x):
        a, b = x.chunk(2, dim=1)
        return a * torch.sigmoid(b)


def conv3x3(cin, cout, bias=False):
    return nn.Conv2d(cin, cout, 3, 1, 1, bias=bias)


def up_block(cin, cout):
    """Nearest x2 upsampling -> 3x3 conv -> BN -> GLU (AttnGAN upBlock)."""
    return nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"),
                         conv3x3(cin, cout * 2), nn.BatchNorm2d(cout * 2), GLU())


def join_block(cin, cout):
    """Same-resolution fusion: 3x3 conv -> BN -> GLU."""
    return nn.Sequential(conv3x3(cin, cout * 2), nn.BatchNorm2d(cout * 2), GLU())


class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.block = nn.Sequential(conv3x3(c, c * 2), nn.BatchNorm2d(c * 2), GLU(),
                                   conv3x3(c, c), nn.BatchNorm2d(c))

    def forward(self, x):
        return x + self.block(x)


class ToRGB(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = conv3x3(c, 3, bias=True)

    def forward(self, h):
        return torch.tanh(self.conv(h))


class ConditioningAugmentation(nn.Module):
    """s -> c ~ N(mu(s), diag(sigma(s)^2)) (StackGAN); KL term regularises it."""

    def __init__(self, sent_dim, c_dim):
        super().__init__()
        self.fc = nn.Linear(sent_dim, c_dim * 4)
        self.glu = GLU()
        self.c_dim = c_dim

    def forward(self, sent):
        mu, logvar = self.glu(self.fc(sent)).chunk(2, dim=1)
        c = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return c, mu, logvar


def pixel_norm(x, eps=1e-8):
    return x * torch.rsqrt(x.pow(2).mean(dim=1, keepdim=True) + eps)


def upsample2x(x):
    return F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
