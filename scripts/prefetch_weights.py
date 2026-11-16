"""Download all pretrained weights once, before jobs start in parallel.

Parallel jobs that all try to download the same file into the shared cache at the same
moment can corrupt it; this script fills the caches sequentially: Inception-v3 and
VGG-16 (torchvision), the FID/KID/IS feature extractor (torch-fidelity), LPIPS and CLIP.

    python scripts/prefetch_weights.py [--marker runs/prefetch_done.json]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import yaml  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--marker")
    a = p.parse_args()
    cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs" / "base.yaml").read_text(encoding="utf-8"))
    done = {}

    from torchvision.models import Inception_V3_Weights, VGG16_Weights, inception_v3, vgg16
    inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True)
    vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    done["torchvision"] = "inception_v3, vgg16"

    import torch_fidelity
    from torch_fidelity.utils import create_feature_extractor
    create_feature_extractor("inception-v3-compat", ["2048", "logits_unbiased"], cuda=False)
    done["torch_fidelity"] = torch_fidelity.__version__

    import lpips
    lpips.LPIPS(net=cfg["eval"]["lpips_net"], version="0.1", verbose=False)
    done["lpips"] = cfg["eval"]["lpips_net"]

    from transformers import CLIPModel, CLIPProcessor
    CLIPModel.from_pretrained(cfg["eval"]["clip_model"])
    CLIPProcessor.from_pretrained(cfg["eval"]["clip_model"])
    done["clip"] = cfg["eval"]["clip_model"]

    print("pretrained weights cached:", done)
    if a.marker:
        Path(a.marker).parent.mkdir(parents=True, exist_ok=True)
        Path(a.marker).write_text(json.dumps({"done": True, **done}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
