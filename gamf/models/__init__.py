from gamf.models.dfgan import DFDiscriminator, DFGenerator
from gamf.models.discriminator import MultiScaleDiscriminator
from gamf.models.generator import CascadeGenerator, TextStyleGenerator
from gamf.models.hybrid import HybridDiscriminator, HybridGenerator

GENERATORS = {"cascade": CascadeGenerator, "stylegan_t": TextStyleGenerator, "dfgan": DFGenerator,
              "hybrid": HybridGenerator}


def build_generator(cfg, word_dim, sent_dim):
    m = cfg["model"]
    if m["generator"] not in GENERATORS:
        raise ValueError(f"unknown model.generator {m['generator']!r}")
    return GENERATORS[m["generator"]](m, word_dim, sent_dim)


def build_discriminator(cfg, sent_dim):
    m = cfg["model"]
    if m["generator"] == "dfgan":
        return DFDiscriminator(m["dfgan_ndf"], sent_dim)
    if m["generator"] == "hybrid":
        return HybridDiscriminator(m, sent_dim)
    return MultiScaleDiscriminator(m["ndf"], sent_dim, m["d_norm"])
