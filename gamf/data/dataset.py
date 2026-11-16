"""Text-image dataset over the subsets written by ``gamf.data.prepare``."""
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from gamf.config import dataset_dir
from gamf.data.text import encode, tokenize
from gamf.utils import read_json

SCALES = (64, 128, 256)


class T2IDataset(Dataset):
    """Returns real images at 64/128/256 px, one encoded caption and metadata.

    Training: a random caption of the image is drawn every time the image is
    visited. Evaluation: the fixed ``eval_caption_idx`` stored in split.json.
    """

    def __init__(self, cfg, split, train=False, scales=SCALES, allow_test=False):
        # Leakage guard: training code never passes allow_test, so the held-out test split
        # can only be opened by the sampling/evaluation scripts after training.
        if split == "test" and (train or not allow_test):
            raise PermissionError("the test split is reserved for final evaluation (allow_test=True, train=False)")
        root = dataset_dir(cfg)
        meta = read_json(root / "split.json")
        self.raw_root = Path(meta["raw_root"])
        self.records = meta["splits"][split]
        self.word2idx = read_json(root / "vocab.json")["word2idx"]
        self.words_num = cfg["data"]["words_num"]
        self.train = train
        self.scales = tuple(scales)
        classes = sorted({r["class_id"] for recs in meta["splits"].values()
                          for r in recs if r["class_id"] is not None})
        self.class_to_idx = {c: i for i, c in enumerate(classes)}

        size = cfg["data"]["image_size"]
        aug = cfg["data"]["augment"]
        if train:
            ops = [T.Resize(int(size * 76 / 64)), T.RandomCrop(size)]
            if aug.get("hflip", True):
                ops.append(T.RandomHorizontalFlip())
            if aug.get("rotate_deg", 0):
                ops.append(T.RandomRotation(aug["rotate_deg"]))
            if aug.get("color_jitter"):
                ops.append(T.ColorJitter(*aug["color_jitter"]))  # brightness, contrast, saturation
        else:
            ops = [T.Resize(size), T.CenterCrop(size)]
        ops += [T.ToTensor(), T.Normalize([0.5] * 3, [0.5] * 3)]
        self.transform = T.Compose(ops)

    def __len__(self):
        return len(self.records)

    def caption_index(self, i):
        rec = self.records[i]
        if self.train:
            return random.randrange(len(rec["captions"]))
        return rec["eval_caption_idx"]

    def encode_caption(self, text):
        ids, length = encode(tokenize(text), self.word2idx, self.words_num, self.train)
        return torch.tensor(ids), length

    def __getitem__(self, i):
        rec = self.records[i]
        ci = self.caption_index(i)
        ids, length = self.encode_caption(rec["captions"][ci])
        with Image.open(self.raw_root / rec["path"]) as im:
            img = self.transform(im.convert("RGB"))
        imgs = [img if s == img.shape[-1] else
                F.interpolate(img[None], size=(s, s), mode="bilinear",
                              antialias=True, align_corners=False)[0]
                for s in self.scales]
        cls = rec["class_id"]
        return {
            "imgs": imgs,
            "caption": ids,
            "cap_len": length,
            # COCO has no classes: every image is its own class, so no pair is masked.
            "class_idx": self.class_to_idx[cls] if cls is not None else 1_000_000 + i,
            "index": i,
            "caption_idx": ci,
        }
