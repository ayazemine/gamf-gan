"""Train GAMF-GAN, a re-implemented baseline, or an ablation variant.

    python -m gamf.train --config configs/base.yaml configs/datasets/cub.yaml \
        configs/models/gamf.yaml --set seed=0

Training schedules (``train.schedule``)
---------------------------------------
* ``stagewise`` (obsolete stage-wise model, not GAMF-GAN): three consecutive phases.
  Phase k trains only generator stage k (phase 1 also CA and F0) and discriminator
  D_k on the output x_k; earlier stages are frozen and the later ones are not run.
  The loss terms of each phase are given by ``train.stage_losses``
  (e.g. [adv, DAMSM] -> [adv, memory] -> [adv, perceptual, style]; the KL term of
  the conditioning augmentation is listed in phase 1). Phase lengths: ``train.stage_epochs``.
* ``joint``: all stages and discriminators trained together end to end with all
  loss terms (original objective of AttnGAN / DM-GAN; single-stage models).

Outputs in runs/<dataset>/<model>/seed_<k>/:
    config.yaml       merged configuration (exact protocol of the run)
    train_log.csv     per-iteration loss terms and gradient norms (stability, R3-6)
    val_curve.csv     validation FID/KID every eval_every epochs (convergence figure)
    train_stats.json  wall-clock time, peak GPU memory, hardware (R3-6, R2-6)
    ckpt/last.pt, ckpt/best.pt  (best = lowest validation FID in the final phase;
                                 the test set is never used)

Crash safety: checkpoints are written atomically (the previous one is kept as
last.pt.prev) every ``train.ckpt_minutes`` and at the end of each epoch, including the
optimiser, AMP-scaler and RNG states. Rerunning the same command resumes from the exact
batch reached (deterministic per-epoch shuffling); log rows written after that
checkpoint are dropped. A finished run is marked by train_stats.json; --fresh restarts.
"""
import argparse
import copy
import csv
import math
import shutil
import time

import torch
from torch.utils.data import DataLoader

from gamf import losses as L
from gamf.config import (add_config_args, damsm_path, dataset_dir, load_config, run_dir, save_config,
                          save_hparam_table)
from gamf.data.dataset import T2IDataset
from gamf.data.leakage import require_clean
from gamf.generate import generate_split
from gamf.models import build_discriminator, build_generator
from gamf.models.text_encoder import load_damsm
from gamf.utils import (amp_dtype, atomic_torch_save, autocast, count_params, get_device, hardware_info,
                         load_checkpoint, rng_state, seed_everything, set_rng_state, write_json)

LOG_KEYS = ["phase", "d_loss", "g_adv", "kl", "damsm", "mem", "perc", "style", "r1", "g_total",
            "grad_g", "grad_d"]
ALL_TERMS = ["adv", "kl", "damsm", "mem", "perc", "style"]


# ------------------------------------------------------------------ phase plan
def phase_plan(cfg):
    """List of (phase index or None, epochs, loss terms)."""
    t = cfg["train"]
    if t["schedule"] == "stagewise" and cfg["model"]["generator"] == "cascade":
        if len(t["stage_epochs"]) != 3:
            raise ValueError("train.stage_epochs must list three phases")
        return [(k, e, list(t["stage_losses"][k])) for k, e in enumerate(t["stage_epochs"])]
    return [(None, t["epochs"], ALL_TERMS)]


def phase_modules(G, D, k):
    """Generator and discriminator modules trained in phase k (None = all)."""
    if k is None:
        return [G], list(D.nets) if hasattr(D, "nets") else [D]
    g = [G.stages[k], G.to_rgb[k]] + ([G.ca, G.init] if k == 0 else [])
    return g, [D[k]]


def set_trainable(G, D, k):
    g_mods, d_mods = phase_modules(G, D, k)
    for net in (G, D):
        net.eval()
        for p in net.parameters():
            p.requires_grad_(False)
    for m in g_mods + d_mods:
        m.train()
        for p in m.parameters():
            p.requires_grad_(True)
    return [p for m in g_mods for p in m.parameters()], [p for m in d_mods for p in m.parameters()]


def make_optimizers(G, g_params, d_params, tcfg):
    g_ids = {id(p) for p in g_params}
    mapping = [p for n, p in G.named_parameters() if "mapping" in n and id(p) in g_ids]
    m_ids = {id(p) for p in mapping}
    groups = [{"params": [p for p in g_params if id(p) not in m_ids]}]
    if mapping:
        groups.append({"params": mapping, "lr": tcfg["lr_g"] * tcfg["mapping_lr_mul"]})
    opt_g = torch.optim.Adam(groups, lr=tcfg["lr_g"], betas=tuple(tcfg["betas"]))
    opt_d = torch.optim.Adam(d_params, lr=tcfg["lr_d"], betas=tuple(tcfg["betas"]))
    return opt_g, opt_d


@torch.no_grad()
def ema_update(ema, model, beta):
    for pe, pm in zip(ema.parameters(), model.parameters()):
        pe.lerp_(pm, 1 - beta)
    for be, bm in zip(ema.buffers(), model.buffers()):
        be.copy_(bm)


@torch.no_grad()
def sync_finished_stages(G, G_ema, k):
    """Before phase k, give G the EMA weights of the finished stages, so the new stage
    is trained on exactly the features that are used at test time."""
    if k is None or k == 0:
        return
    src = [G_ema.ca, G_ema.init] + [m for i in range(k) for m in (G_ema.stages[i], G_ema.to_rgb[i])]
    dst = [G.ca, G.init] + [m for i in range(k) for m in (G.stages[i], G.to_rgb[i])]
    for s, d in zip(src, dst):
        d.load_state_dict(s.state_dict())


def validate(G_ema, text_enc, val_ds, cfg, device, out_dir, upto):
    from gamf.metrics import fidelity_metrics
    tmp = out_dir / "_val_samples"
    shutil.rmtree(tmp, ignore_errors=True)
    generate_split(G_ema, text_enc, val_ds, tmp, cfg["model"]["z_dim"], device, seed=cfg["eval"]["seed"],
                   samples_per_caption=cfg["train"]["val_samples_per_caption"], upto=upto)
    m = fidelity_metrics(tmp, dataset_dir(cfg) / "real_256" / "val", device)
    shutil.rmtree(tmp, ignore_errors=True)
    return m


class ResumableSampler(torch.utils.data.Sampler):
    """Deterministic per-epoch shuffling (seed, epoch) that can start in the middle of an
    epoch, so an interrupted run continues with exactly the remaining batches."""

    def __init__(self, n, seed):
        self.n, self.seed, self.epoch, self.start = n, seed, 0, 0

    def set_epoch(self, epoch, start=0):
        self.epoch, self.start = epoch, start

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed * 100_003 + self.epoch)
        return iter(torch.randperm(self.n, generator=g)[self.start:].tolist())

    def __len__(self):
        return self.n - self.start


def _f(v):
    return float(v.detach()) if torch.is_tensor(v) else float(v)


def _truncate_csv(path, max_iter, col=0):
    """Drop log rows written after the checkpoint we resume from (no duplicated rows)."""
    if not path.exists():
        return
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        return
    keep = [rows[0]] + [r for r in rows[1:] if len(r) > col and r[col].isdigit() and int(r[col]) <= max_iter]
    with open(path, "w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerows(keep)


# ------------------------------------------------------------------------- main
def main():
    p = add_config_args(argparse.ArgumentParser(description=__doc__,
                                                formatter_class=argparse.RawDescriptionHelpFormatter))
    p.add_argument("--fresh", action="store_true",
                   help="ignore existing checkpoints and start from scratch (default: resume)")
    p.add_argument("--resume", action="store_true", help="kept for compatibility; resuming is the default")
    args = p.parse_args()
    cfg = load_config(args.config, args.set)
    seed_everything(cfg["seed"])
    torch.backends.cudnn.benchmark = True
    device = get_device()
    tcfg, lcfg, dcfg = cfg["train"], cfg["loss"], cfg["damsm"]
    out = run_dir(cfg)
    (out / "ckpt").mkdir(parents=True, exist_ok=True)
    if (out / "train_stats.json").exists() and not args.fresh:
        print(f"{out} already finished (train_stats.json present)")
        return
    save_config(cfg, out / "config.yaml")
    dtype = amp_dtype(tcfg["amp"])
    _plan = phase_plan(cfg)
    save_hparam_table(cfg, out, extra={
        "derived.schedule": "stagewise" if _plan[0][0] is not None else "joint",
        "derived.phase_epochs": [e for _, e, _ in _plan],
        "derived.phase_losses": [t for _, _, t in _plan],
        "derived.total_epochs": sum(e for _, e, _ in _plan)})
    scaler = torch.amp.GradScaler(enabled=dtype == torch.float16)

    require_clean(cfg)  # refuse to train unless the split passed the leakage audit
    text_enc, img_enc = load_damsm(damsm_path(cfg), device)
    word_dim = sent_dim = cfg["text"]["hidden_dim"]
    G = build_generator(cfg, word_dim, sent_dim).to(device)
    D = build_discriminator(cfg, sent_dim).to(device)
    G_ema = copy.deepcopy(G).eval()
    for q in G_ema.parameters():
        q.requires_grad_(False)
    dfgan = cfg["model"]["generator"] in ("dfgan", "hybrid")  # hinge + MA-GP objective
    plan = phase_plan(cfg)
    total_epochs = sum(e for _, e, _ in plan)
    use_vgg = any(t in terms for _, _, terms in plan for t in ("perc", "style")) and \
        (lcfg["lambda_perc"] > 0 or lcfg["lambda_style"] > 0)
    vgg = (L.VGGLoss(lcfg["perc_layers"], lcfg["style_layers"], lcfg.get("vgg_pretrained", True)).to(device)
           if use_vgg else None)

    train_ds = T2IDataset(cfg, "train", train=True)
    val_ds = T2IDataset(cfg, "val", train=False)
    workers = cfg["data"]["num_workers"]
    bs = tcfg["batch_size"]
    sampler = ResumableSampler(len(train_ds), cfg["seed"])
    loader = DataLoader(train_ds, bs, sampler=sampler, drop_last=True, num_workers=workers,
                        pin_memory=True, persistent_workers=workers > 0)
    batches_per_epoch = len(train_ds) // bs

    # ---------------------------------------------------------------- resume state
    # 'epoch' is the epoch in progress, 'batch' the number of its batches already done.
    ck = None if args.fresh else load_checkpoint(out / "ckpt" / "last.pt", device)
    epoch, batch0, best_fid, it, elapsed, peak_mem = 1, 0, math.inf, 0, 0.0, 0.0
    if ck is not None:
        G.load_state_dict(ck["G"]); G_ema.load_state_dict(ck["G_ema"]); D.load_state_dict(ck["D"])
        epoch, batch0, best_fid, it = ck["epoch"], ck["batch"], ck["best_fid"], ck["iter"]
        elapsed, peak_mem = ck["elapsed"], ck.get("peak_mem_gb", 0.0)
        if batch0 >= batches_per_epoch:
            epoch, batch0 = epoch + 1, 0
        if "scaler" in ck:
            scaler.load_state_dict(ck["scaler"])
        set_rng_state(ck["rng"])
        print(f"resuming {out.name}: epoch {epoch}, batch {batch0}, iteration {it}")
        _truncate_csv(out / "train_log.csv", it)
        _truncate_csv(out / "val_curve.csv", it, col=1)

    new = ck is None
    log_f = open(out / "train_log.csv", "w" if new else "a", newline="")
    val_f = open(out / "val_curve.csv", "w" if new else "a", newline="")
    log, vlog = csv.writer(log_f), csv.writer(val_f)
    if new:
        log.writerow(["iter", "epoch"] + LOG_KEYS)
        vlog.writerow(["epoch", "iter", "phase", "val_FID", "val_KID", "hours"])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    g1, g2, g3 = dcfg["gamma1"], dcfg["gamma2"], dcfg["gamma3"]
    ckpt_every = tcfg.get("ckpt_minutes", 10) * 60

    def save_state(ep, nb, p_idx, opt_g, opt_d):
        nonlocal peak_mem
        if device.type == "cuda":
            peak_mem = max(peak_mem, torch.cuda.max_memory_allocated() / 1024 ** 3)
        atomic_torch_save({"G": G.state_dict(), "G_ema": G_ema.state_dict(), "D": D.state_dict(),
                           "opt_g": opt_g.state_dict(), "opt_d": opt_d.state_dict(),
                           "scaler": scaler.state_dict(), "rng": rng_state(), "epoch": ep, "batch": nb,
                           "phase_idx": p_idx, "iter": it, "elapsed": elapsed, "best_fid": best_fid,
                           "peak_mem_gb": peak_mem, "cfg": cfg}, out / "ckpt" / "last.pt")
        log_f.flush(); val_f.flush()

    epoch0 = 0
    for p_idx, (k, n_epochs, terms) in enumerate(plan):
        first, last_ep = epoch0 + 1, epoch0 + n_epochs
        epoch0 = last_ep
        if epoch > last_ep:
            continue
        final_phase = p_idx == len(plan) - 1
        phase_start = epoch == first and batch0 == 0
        if phase_start:
            sync_finished_stages(G, G_ema, k)
        g_params, d_params = set_trainable(G, D, k)
        opt_g, opt_d = make_optimizers(G, g_params, d_params, tcfg)
        if ck is not None and ck.get("phase_idx") == p_idx and not phase_start:
            opt_g.load_state_dict(ck["opt_g"]); opt_d.load_state_dict(ck["opt_d"])
        stages = [k] if k is not None else list(range(len(G.stages) if hasattr(G, "stages") else 3))
        phase_label = "joint" if k is None else f"stage{k + 1}"
        print(f"== phase {phase_label}: epochs {first}-{last_ep}, losses {terms}")

        while epoch <= last_ep:
            set_trainable(G, D, k)
            sampler.set_epoch(epoch, batch0 * bs)
            t_mark = time.time()
            last_save = time.time()
            nb = batch0
            for batch in loader:
                reals = [x.to(device, non_blocking=True) for x in batch["imgs"]]
                caps = batch["caption"].to(device)
                lens = batch["cap_len"]
                cls = batch["class_idx"].to(device)
                pad = caps == 0
                with torch.no_grad():
                    words, sent = text_enc(caps, lens)
                z = torch.randn(caps.size(0), cfg["model"]["z_dim"], device=device)
                with autocast(device, dtype):
                    out_g = G(z, sent, words, pad, upto=k)
                fakes = out_g["imgs"]
                top = len(fakes) - 1  # index (resolution) of the current output

                # ---- discriminator step
                r1 = torch.zeros((), device=device)
                if hasattr(D, "d_step"):   # hybrid: hinge per D branch + MA-GP, shared CLIP features
                    d_loss, r1 = D.d_step(reals[-1], fakes[-1].detach(), sent, dtype)
                    d_total = d_loss + r1
                elif dfgan:
                    with autocast(device, dtype):
                        d_loss = L.dfgan_d_loss(D, reals[-1], fakes[-1].detach(), sent)
                    r1 = L.ma_gp(D, reals[-1], sent)  # logged in the 'r1' column
                    d_total = d_loss + r1
                else:
                    with autocast(device, dtype):
                        d_loss = sum(L.d_stage_loss(D[i], reals[i], fakes[i].detach(),
                                                    reals[i].roll(1, dims=0), sent) for i in stages)
                    if lcfg["r1_gamma"] > 0:
                        r1 = sum(L.r1_penalty(D[i], reals[i], sent) for i in stages)
                    d_total = d_loss + lcfg["r1_gamma"] / 2 * r1
                opt_d.zero_grad(set_to_none=True)
                scaler.scale(d_total).backward()
                scaler.unscale_(opt_d)
                grad_d = torch.nn.utils.clip_grad_norm_(d_params, float("inf"))
                scaler.step(opt_d)

                # ---- generator step
                zero = torch.zeros((), device=device)
                with autocast(device, dtype):
                    g_adv = (L.dfgan_g_loss(D, fakes[-1], sent) if dfgan else
                             sum(L.g_stage_loss(D[i], fakes[i], sent) for i in stages))
                kl = (L.kl_loss(out_g["mu"], out_g["logvar"]).to(device)
                      if "kl" in terms and lcfg["lambda_kl"] > 0 else zero)
                damsm = zero
                if "damsm" in terms and lcfg["lambda_damsm"] > 0:
                    local, glob = img_enc(fakes[-1].float())
                    damsm = (L.damsm_word_loss(local, words, lens, cls, g1, g2, g3)
                             + L.damsm_sent_loss(glob, sent, cls, g3))
                mem = zero
                if "mem" in terms and lcfg["lambda_mem"] > 0 and 1 in out_g.get("mem_pairs", {}):
                    if lcfg["mem_type"] == "real_feature":   # F_orig = phi(x_real), F_mem = phi(x_2)
                        local2, _ = img_enc(fakes[1].float())
                        with torch.no_grad():
                            local_real, _ = img_enc(reals[1])
                        mem = L.memory_feature_loss(local_real, local2)
                    elif lcfg["mem_type"] == "module_feature":  # features entering vs leaving the memory
                        mem = L.memory_feature_loss(*out_g["mem_pairs"][1])
                    else:  # "damsm_word": word-level DAMSM on the memory-stage output x2
                        local2, _ = img_enc(fakes[1].float())
                        mem = L.damsm_word_loss(local2, words, lens, cls, g1, g2, g3)
                perc = style = zero
                if vgg is not None and ("perc" in terms or "style" in terms):
                    perc, style = vgg(fakes[-1], reals[top])
                    perc = perc if "perc" in terms else zero
                    style = style if "style" in terms else zero
                g_total = (lcfg["lambda_adv"] * g_adv + lcfg["lambda_kl"] * kl
                           + lcfg["lambda_damsm"] * damsm + lcfg["lambda_mem"] * mem
                           + lcfg["lambda_perc"] * perc + lcfg["lambda_style"] * style)
                opt_g.zero_grad(set_to_none=True)
                scaler.scale(g_total).backward()
                scaler.unscale_(opt_g)
                grad_g = torch.nn.utils.clip_grad_norm_(g_params, float("inf"))
                scaler.step(opt_g)
                scaler.update()
                ema_update(G_ema, G, tcfg["ema_beta"])

                it += 1
                nb += 1
                if it % tcfg["log_every"] == 0:
                    vals = [d_loss, g_adv, kl, damsm, mem, perc, style, r1, g_total, grad_g, grad_d]
                    log.writerow([it, epoch, phase_label] + [f"{_f(v):.5f}" for v in vals])
                # time-based mid-epoch checkpoint (power-failure safety)
                if time.time() - last_save >= ckpt_every and nb < batches_per_epoch:
                    elapsed += time.time() - t_mark
                    t_mark = last_save = time.time()
                    save_state(epoch, nb, p_idx, opt_g, opt_d)

            elapsed += time.time() - t_mark
            print(f"[{phase_label}] epoch {epoch}/{total_epochs}  {elapsed / 3600:.2f} h  "
                  f"D {_f(d_loss):.3f}  G {_f(g_total):.3f}", flush=True)
            if epoch % tcfg["eval_every"] == 0 or epoch == last_ep:
                try:
                    m = validate(G_ema, text_enc, val_ds, cfg, device, out, upto=k)
                except ImportError:
                    print("torch-fidelity not installed: validation skipped, best.pt = last.pt")
                    m = {"FID": math.nan, "KID": math.nan}
                vlog.writerow([epoch, it, phase_label, f"{m['FID']:.4f}", f"{m['KID']:.6f}",
                               f"{elapsed / 3600:.3f}"])
                # model selection only on the final output (last phase), never on test data
                if final_phase and not m["FID"] >= best_fid:  # also true for NaN -> keep latest
                    best_fid = m["FID"] if not math.isnan(m["FID"]) else best_fid
                    atomic_torch_save({"G_ema": G_ema.state_dict(), "epoch": epoch, "val_FID": m["FID"],
                                       "cfg": cfg}, out / "ckpt" / "best.pt")
            save_state(epoch, batches_per_epoch, p_idx, opt_g, opt_d)  # epoch complete
            epoch, batch0 = epoch + 1, 0

    if device.type == "cuda":
        peak_mem = max(peak_mem, torch.cuda.max_memory_allocated() / 1024 ** 3)
    stats = {
        "model": cfg["model"]["name"], "dataset": cfg["data"]["name"], "seed": cfg["seed"],
        "schedule": "stagewise" if plan[0][0] is not None else "joint",
        "phase_epochs": [e for _, e, _ in plan], "phase_losses": [t for _, _, t in plan],
        "epochs": total_epochs, "iterations": it, "batch_size": bs,
        "train_images": len(train_ds), "train_hours": elapsed / 3600,
        "best_val_FID": best_fid,
        "params_G": count_params(G), "params_D": count_params(D),
        "peak_gpu_mem_gb": peak_mem if device.type == "cuda" else None,
        "hardware": hardware_info(),
    }
    log_f.close(); val_f.close()
    write_json(stats, out / "train_stats.json")  # completion marker of the run


if __name__ == "__main__":
    main()
