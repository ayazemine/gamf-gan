"""Caption preprocessing: lower-casing, punctuation removal, word tokenisation."""
import random
import re

PAD, UNK = 0, 1
_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(caption):
    return _TOKEN.findall(caption.lower())


def encode(tokens, word2idx, words_num, train):
    """Map tokens to ids and fit them into ``words_num`` slots.

    Longer captions are shortened as in AttnGAN: during training a random,
    order-preserving subset of ``words_num`` words is kept; at evaluation the
    first ``words_num`` words are kept, so evaluation is deterministic.
    """
    ids = [word2idx.get(t, UNK) for t in tokens] or [UNK]
    if len(ids) > words_num:
        if train:
            keep = sorted(random.sample(range(len(ids)), words_num))
            ids = [ids[i] for i in keep]
        else:
            ids = ids[:words_num]
    length = len(ids)
    return ids + [PAD] * (words_num - length), length
