"""Task 2.2 - LoRA fine-tuning of the Stable Diffusion 1.5 UNet (two runs, same data / steps / batch / LR).

Standard text-to-image LoRA recipe (as in the diffusers example `train_text_to_image_lora.py`): frozen VAE,
text encoder and UNet in half precision; trainable LoRA adapters on the attention projections
(to_q, to_k, to_v, to_out.0) kept in float32; epsilon-prediction MSE loss with min-SNR weighting; AdamW with
warm-up + cosine schedule; random horizontal flips. Every number of the 2.2 table is measured here.
"""
from __future__ import annotations

import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .config import P2Config


def device_and_dtype():
    if torch.cuda.is_available():
        return "cuda", (torch.bfloat16 if torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16)
    return "cpu", torch.float32


def _from_pretrained(cls, base, subfolder, **kw):
    """Prefer the fp16 weight files (half the download); fall back to the default files."""
    try:
        return cls.from_pretrained(base, subfolder=subfolder, variant="fp16", **kw)
    except (OSError, ValueError):
        return cls.from_pretrained(base, subfolder=subfolder, **kw)


def _load_batch(rows, idx, res, rng):
    pix = []
    for i in idx:
        im = Image.open(rows[i]["path"]).convert("RGB")
        if im.size != (res, res):
            im = im.resize((res, res), Image.BICUBIC)
        if rng.random() < 0.5:
            im = im.transpose(Image.FLIP_LEFT_RIGHT)
        a = torch.from_numpy(np.asarray(im, dtype=np.float32) / 127.5 - 1.0).permute(2, 0, 1)
        pix.append(a)
    return torch.stack(pix), [rows[i]["text"] for i in idx]


def train_lora(cfg: P2Config, name: str, rank: int, rows: list[dict], log=print, out_root: str | None = None) -> dict:
    from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
    from diffusers.optimization import get_scheduler
    from diffusers.training_utils import compute_snr
    from diffusers.utils import convert_state_dict_to_diffusers
    from peft import LoraConfig
    from peft.utils import get_peft_model_state_dict
    from transformers import CLIPTextModel, CLIPTokenizer

    dev, wdt = device_and_dtype()
    torch.manual_seed(cfg.seed); random.seed(cfg.seed); np.random.seed(cfg.seed)
    rng = random.Random(cfg.seed)
    base = cfg.base_model
    tok = CLIPTokenizer.from_pretrained(base, subfolder="tokenizer")
    te = _from_pretrained(CLIPTextModel, base, "text_encoder")
    vae = _from_pretrained(AutoencoderKL, base, "vae")
    unet = _from_pretrained(UNet2DConditionModel, base, "unet")
    noise_sched = DDPMScheduler.from_pretrained(base, subfolder="scheduler")
    for m in (te, vae, unet):
        m.requires_grad_(False)
    te.to(dev, dtype=wdt); unet.to(dev, dtype=wdt)
    vae.to(dev, dtype=torch.float32 if wdt == torch.float16 else wdt)  # fp16 VAE encoding can overflow

    unet.add_adapter(LoraConfig(r=rank, lora_alpha=rank, init_lora_weights="gaussian",
                                target_modules=["to_k", "to_q", "to_v", "to_out.0"]))
    params = [p for p in unet.parameters() if p.requires_grad]
    for p in params:  # trainable adapters in float32
        p.data = p.data.float()
    n_trainable = sum(p.numel() for p in params)
    n_total = sum(p.numel() for p in unet.parameters())
    if cfg.grad_checkpointing and dev == "cuda":
        unet.enable_gradient_checkpointing()
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=1e-2)
    lr_sched = get_scheduler("cosine", optimizer=opt, num_warmup_steps=cfg.warmup_steps, num_training_steps=cfg.max_steps)
    scaler = torch.amp.GradScaler("cuda") if (dev == "cuda" and wdt == torch.float16) else None

    log(f"[train {name}] rank={rank} trainable={n_trainable:,} / UNet {n_total:,} ({100 * n_trainable / n_total:.2f}%), "
        f"steps={cfg.max_steps} batch={cfg.batch_size} lr={cfg.lr} dtype={wdt} device={dev}")
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    order, pos = [], 0
    losses, t0 = [], time.time()
    unet.train()
    for step in range(1, cfg.max_steps + 1):
        if pos + cfg.batch_size > len(order):  # new epoch, reshuffled with the seeded RNG
            order = list(range(len(rows)))
            rng.shuffle(order)
            pos = 0
        idx = order[pos:pos + cfg.batch_size]
        pos += cfg.batch_size
        pix, caps = _load_batch(rows, idx, cfg.resolution, rng)
        with torch.no_grad():
            lat = vae.encode(pix.to(dev, dtype=vae.dtype)).latent_dist.sample() * vae.config.scaling_factor
            lat = lat.to(wdt)
            ids = tok(caps, padding="max_length", truncation=True, max_length=tok.model_max_length,
                      return_tensors="pt").input_ids.to(dev)
            enc = te(ids)[0]
        noise = torch.randn_like(lat)
        ts = torch.randint(0, noise_sched.config.num_train_timesteps, (lat.shape[0],), device=dev).long()
        noisy = noise_sched.add_noise(lat, noise, ts)
        with torch.autocast(device_type=dev, dtype=wdt, enabled=(dev == "cuda")):
            pred = unet(noisy, ts, enc).sample
        target = noise if noise_sched.config.prediction_type == "epsilon" else noise_sched.get_velocity(lat, noise, ts)
        snr = compute_snr(noise_sched, ts)
        w = torch.stack([snr, cfg.snr_gamma * torch.ones_like(snr)], dim=1).min(dim=1)[0] / snr
        loss = (F.mse_loss(pred.float(), target.float(), reduction="none").mean(dim=[1, 2, 3]) * w).mean()
        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
        lr_sched.step()
        opt.zero_grad(set_to_none=True)
        losses.append(float(loss.detach()))
        if step % 100 == 0 or step == cfg.max_steps:
            log(f"[train {name}] step {step}/{cfg.max_steps} loss(avg100)={np.mean(losses[-100:]):.4f} "
                f"lr={lr_sched.get_last_lr()[0]:.2e} {time.time() - t0:.0f}s")
    if dev == "cuda":
        torch.cuda.synchronize()
    secs = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1e9 if dev == "cuda" else float("nan")

    out = os.path.join(out_root or os.path.join(cfg.work_dir, "lora"), name)
    os.makedirs(out, exist_ok=True)
    unet = unet.to(torch.float32)
    StableDiffusionPipeline.save_lora_weights(
        save_directory=out, unet_lora_layers=convert_state_dict_to_diffusers(get_peft_model_state_dict(unet)),
        safe_serialization=True)
    stats = {"run": name, "method": f"LoRA rank {rank} (alpha {rank}) on UNet attention", "rank": rank,
             "trainable_params": n_trainable, "unet_params": n_total, "learning_rate": cfg.lr, "steps": cfg.max_steps,
             "batch_size": cfg.batch_size, "resolution": cfg.resolution, "epochs": round(cfg.max_steps * cfg.batch_size / len(rows), 2),
             "peak_gpu_memory_gb": round(peak, 2), "train_time_s": round(secs, 1), "gpu_hours": round(secs / 3600, 3),
             "final_loss_avg100": round(float(np.mean(losses[-100:])), 4), "loss_curve": [round(float(np.mean(losses[max(0, i - 25):i + 1])), 4)
                                                                                         for i in range(0, len(losses), 25)],
             "dtype": str(wdt), "device": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu", "lora_dir": out}
    with open(os.path.join(out, "train_stats.json"), "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=1)
    log(f"[train {name}] done in {secs / 60:.1f} min, peak GPU memory {peak:.2f} GB -> {out}")
    del unet, te, vae, opt
    if dev == "cuda":
        torch.cuda.empty_cache()
    return stats
