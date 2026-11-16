"""DAMSM text and image encoders (Xu et al., AttnGAN, CVPR 2018).

Both encoders are pre-trained once per dataset with ``gamf.train_damsm`` and
are kept FROZEN while any generator is trained, so every model (GAMF-GAN,
baselines and ablations) is conditioned on and evaluated with identical
text features.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class RNNEncoder(nn.Module):
    """Bi-LSTM: caption -> word features e (B x D x T) and sentence feature s (B x D)."""

    def __init__(self, vocab_size, emb_dim=300, hidden_dim=256, dropout=0.5, sentence="mean"):
        super().__init__()
        # sentence="mean": s = (1/n) sum_i e_i over the n caption words;
        # sentence="last": concatenated final hidden states of both directions (AttnGAN).
        self.sentence = sentence
        self.embed = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.drop = nn.Dropout(dropout)
        self.rnn = nn.LSTM(emb_dim, hidden_dim // 2, batch_first=True, bidirectional=True)
        self.out_dim = hidden_dim

    def forward(self, captions, cap_lens):
        emb = self.drop(self.embed(captions))
        packed = pack_padded_sequence(emb, cap_lens.cpu(), batch_first=True,
                                      enforce_sorted=False)
        out, (h, _) = self.rnn(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=captions.size(1))
        words = out.transpose(1, 2).contiguous()
        if self.sentence == "mean":
            valid = (torch.arange(captions.size(1), device=captions.device)[None, :]
                     < cap_lens.to(captions.device)[:, None]).to(words.dtype)
            sent = (words * valid[:, None, :]).sum(2) / valid.sum(1, keepdim=True).clamp(min=1)
        else:
            sent = torch.cat([h[0], h[1]], dim=1)
        return words, sent


class CNNEncoder(nn.Module):
    """Inception-v3 (ImageNet weights, frozen) with trainable projections.

    Returns local region features (B x D x 17 x 17, from Mixed_6e) and a global
    image feature (B x D, from the pooled Mixed_7c output).
    """

    def __init__(self, out_dim=256, pretrained=True):
        super().__init__()
        from torchvision.models import Inception_V3_Weights, inception_v3
        weights = Inception_V3_Weights.IMAGENET1K_V1 if pretrained else None
        net = inception_v3(weights=weights, aux_logits=True, init_weights=False)
        for p in net.parameters():
            p.requires_grad_(False)
        self.net = net.eval()
        self.local_proj = nn.Conv2d(768, out_dim, 1)
        self.global_proj = nn.Linear(2048, out_dim)

    def train(self, mode=True):
        super().train(mode)
        self.net.eval()  # the backbone always stays in inference mode
        return self

    def forward(self, x):
        # x in [-1, 1]; the torchvision port of the Google weights expects [-1, 1]
        # (its forward() would apply transform_input, which we bypass here).
        x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
        n = self.net
        for layer in (n.Conv2d_1a_3x3, n.Conv2d_2a_3x3, n.Conv2d_2b_3x3, n.maxpool1,
                      n.Conv2d_3b_1x1, n.Conv2d_4a_3x3, n.maxpool2, n.Mixed_5b,
                      n.Mixed_5c, n.Mixed_5d, n.Mixed_6a, n.Mixed_6b, n.Mixed_6c,
                      n.Mixed_6d, n.Mixed_6e):
            x = layer(x)
        local = x
        x = n.Mixed_7c(n.Mixed_7b(n.Mixed_7a(x)))
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.local_proj(local), self.global_proj(x)


def load_damsm(path, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    t = ckpt["text_cfg"]
    text_enc = RNNEncoder(ckpt["vocab_size"], t["embedding_dim"], t["hidden_dim"], t["dropout"],
                          t.get("sentence", "last"))
    # the checkpoint holds the full Inception backbone, so no weight download is needed
    img_enc = CNNEncoder(t["hidden_dim"], pretrained=False)
    text_enc.load_state_dict(ckpt["text_encoder"])
    img_enc.load_state_dict(ckpt["image_encoder"])
    for m in (text_enc, img_enc):
        m.to(device).eval()
        for p in m.parameters():
            p.requires_grad_(False)
    return text_enc, img_enc
