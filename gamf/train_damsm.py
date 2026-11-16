"""Pre-train the DAMSM text/image encoders once per dataset (train split only).

The checkpoint with the lowest validation DAMSM loss is kept and shared,
frozen, by every generator trained on that dataset.

    python -m gamf.train_damsm --config configs/base.yaml configs/datasets/cub.yaml
"""
import argparse
import csv
import time

import torch
from torch.utils.data import DataLoader

from gamf.config import add_config_args, damsm_path, load_config, save_config, save_hparam_table
from gamf.data.dataset import T2IDataset
from gamf.data.leakage import require_clean
from gamf.losses import damsm_sent_loss, damsm_word_loss
from gamf.models.text_encoder import CNNEncoder, RNNEncoder
from gamf.utils import (atomic_torch_save, get_device, load_checkpoint, mark_done, rng_state,
                         seed_everything, set_rng_state)


def run_epoch(text_enc, img_enc, loader, dcfg, device, opt=None):
    train = opt is not None
    text_enc.train(train)
    img_enc.train(train)
    total, n = 0.0, 0
    for batch in loader:
        img = batch["imgs"][0].to(device)
        caps, lens = batch["caption"].to(device), batch["cap_len"]
        cls = batch["class_idx"].to(device)
        with torch.set_grad_enabled(train):
            local, glob = img_enc(img)
            words, sent = text_enc(caps, lens)
            loss = (damsm_word_loss(local, words, lens, cls, dcfg["gamma1"], dcfg["gamma2"], dcfg["gamma3"])
                    + damsm_sent_loss(glob, sent, cls, dcfg["gamma3"]))
        if train:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(text_enc.parameters(), dcfg["rnn_grad_clip"])
            opt.step()
        total += loss.item() * img.size(0)
        n += img.size(0)
    return total / max(n, 1)


def main():
    args = add_config_args(argparse.ArgumentParser(description=__doc__)).parse_args()
    cfg = load_config(args.config, args.set)
    seed_everything(cfg["seed"])
    require_clean(cfg)  # refuse to train unless the split passed the leakage audit
    device = get_device()
    dcfg, tcfg = cfg["damsm"], cfg["text"]
    out = damsm_path(cfg)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out.parent / "config.yaml")
    save_hparam_table({k: cfg[k] for k in ("seed", "data", "text", "damsm")}, out.parent)

    workers = cfg["data"]["num_workers"]
    train_ds = T2IDataset(cfg, "train", train=True, scales=(cfg["data"]["image_size"],))
    val_ds = T2IDataset(cfg, "val", train=False, scales=(cfg["data"]["image_size"],))
    # drop_last: the contrastive loss needs a full batch of negatives
    train_dl = DataLoader(train_ds, dcfg["batch_size"], shuffle=True, num_workers=workers,
                          drop_last=True, pin_memory=True, persistent_workers=workers > 0)
    val_dl = DataLoader(val_ds, dcfg["batch_size"], shuffle=False, num_workers=workers,
                        drop_last=True)

    text_enc = RNNEncoder(len(train_ds.word2idx), tcfg["embedding_dim"],
                          tcfg["hidden_dim"], tcfg["dropout"], tcfg["sentence"]).to(device)
    img_enc = CNNEncoder(tcfg["hidden_dim"], pretrained=dcfg["pretrained_backbone"]).to(device)
    params = list(text_enc.parameters()) + [p for p in img_enc.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=dcfg["lr"], betas=(0.5, 0.999))

    # Crash safety: an atomic per-epoch checkpoint (models, optimiser, RNG, best value) lets
    # an interrupted run continue with the next epoch; completion is marked by damsm_done.json.
    done, last = out.parent / "damsm_done.json", out.parent / "damsm_last.pt"
    if done.exists():
        print(f"DAMSM already trained ({out})")
        return
    best, start = float("inf"), 1
    ck = load_checkpoint(last, device)
    if ck is not None:
        text_enc.load_state_dict(ck["text_encoder"]); img_enc.load_state_dict(ck["image_encoder"])
        opt.load_state_dict(ck["opt"]); set_rng_state(ck["rng"])
        best, start = ck["best"], ck["epoch"] + 1
        print(f"resuming DAMSM at epoch {start}")
    log_path = out.parent / "damsm_log.csv"
    if ck is not None and log_path.exists():  # drop rows written after the checkpoint
        with open(log_path, encoding="utf-8", newline="") as f:
            rows = list(csv.reader(f))
        rows = [rows[0]] + [r for r in rows[1:] if r and int(r[0]) < start]
    else:
        rows = [["epoch", "train_loss", "val_loss", "seconds"]]
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        log = csv.writer(f)
        log.writerows(rows)
        for epoch in range(start, dcfg["epochs"] + 1):
            t0 = time.time()
            tr = run_epoch(text_enc, img_enc, train_dl, dcfg, device, opt)
            va = run_epoch(text_enc, img_enc, val_dl, dcfg, device)
            log.writerow([epoch, f"{tr:.4f}", f"{va:.4f}", f"{time.time() - t0:.1f}"])
            f.flush()
            if va < best:
                best = va
                atomic_torch_save({"text_encoder": text_enc.state_dict(),
                                   "image_encoder": img_enc.state_dict(),
                                   "vocab_size": len(train_ds.word2idx), "text_cfg": tcfg,
                                   "epoch": epoch, "val_loss": va}, out)
            atomic_torch_save({"text_encoder": text_enc.state_dict(), "image_encoder": img_enc.state_dict(),
                               "opt": opt.state_dict(), "rng": rng_state(), "epoch": epoch, "best": best}, last)
            print(f"[DAMSM] epoch {epoch} train {tr:.3f} val {va:.3f} best {best:.3f}", flush=True)
    mark_done(done, best_val_loss=best, epochs=dcfg["epochs"])


if __name__ == "__main__":
    main()
