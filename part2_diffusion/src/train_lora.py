"""LoRA training that never keeps the VAE, text encoder and UNet on CUDA together."""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.utils import convert_state_dict_to_diffusers
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer
from prepare_data import prepared

ROOT = Path(__file__).resolve().parents[1]


def release():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def attach_lora(model, rank):
    model.requires_grad_(False)
    model.add_adapter(LoraConfig(r=rank, lora_alpha=rank, init_lora_weights="gaussian",
                                target_modules=["to_q", "to_k", "to_v", "to_out.0"]))
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    return [p for p in model.parameters() if p.requires_grad]


def prepare_cache(config):
    data = ROOT / "data/processed"
    if not prepared(data):
        raise RuntimeError("Run prepare_data.py first; incomplete folders are not accepted.")
    rows = [json.loads(line) for line in (data / "train/metadata.jsonl").read_text().splitlines()]
    signature = hashlib.sha256((json.dumps(config, sort_keys=True) +
        (data / "train/metadata.jsonl").read_text()).encode()).hexdigest()
    cache = data / "conditioning_cache.pt"
    meta = data / "conditioning_cache.json"
    if cache.exists() and meta.exists() and json.loads(meta.read_text())["signature"] == signature:
        return torch.load(cache, map_location="cpu", weights_only=True)
    tokenizer = CLIPTokenizer.from_pretrained(config["base_model"], subfolder="tokenizer",revision=config['base_revision'])
    text_encoder = CLIPTextModel.from_pretrained(config["base_model"], subfolder="text_encoder",
        torch_dtype=torch.float16,revision=config['base_revision'],variant=config['variant']).to("cuda").eval()
    hidden = []
    with torch.inference_mode():
        for offset in range(0, len(rows), 8):
            tokens = tokenizer([r["text"] for r in rows[offset:offset + 8]], padding="max_length",
                               truncation=True, max_length=tokenizer.model_max_length, return_tensors="pt")
            hidden.append(text_encoder(tokens.input_ids.to("cuda"))[0].cpu())
    del text_encoder
    release()
    vae = AutoencoderKL.from_pretrained(config["base_model"], subfolder="vae",revision=config['base_revision'],
                                       variant=config['variant']).to("cuda").eval()
    vae.enable_slicing()
    latents = []
    generator = torch.Generator(device="cuda").manual_seed(config["seed"])
    with torch.inference_mode():
        for i, row in enumerate(rows):
            with Image.open(data / "train" / row["file_name"]) as image:
                array = np.asarray(image.convert("RGB"), dtype=np.float32) / 127.5 - 1
            pixels = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to("cuda")
            z = vae.encode(pixels).latent_dist.sample(generator=generator) * vae.config.scaling_factor
            latents.append(z.half().cpu())
            if (i + 1) % 100 == 0:
                print(f"Encoded training image {i + 1}/{len(rows)}", flush=True)
    del vae
    release()
    result = {"latents": torch.cat(latents), "hidden": torch.cat(hidden)}
    assert all(torch.isfinite(v).all() for v in result.values())
    torch.save(result, cache)
    meta.write_text(json.dumps({"signature": signature, "training_images": len(rows)}, indent=2))
    return result


def train_updates(model, latents, hidden, noise_scheduler, output, config, rank, resume=False, stop_after=None):
    """Shared by the real run and the tiny-model optimizer/checkpoint integration test."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=config["learning_rate"])
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    generator = torch.Generator().manual_seed(config["seed"])
    model.train()
    if config.get("gradient_checkpointing"):
        model.enable_gradient_checkpointing()
    signature = {k: config[k] for k in ["seed", "max_train_steps", "gradient_accumulation_steps", "learning_rate"]}
    signature["rank"] = rank
    step, history, prior_seconds, prior_peak = 0, [], 0.0, 0.0
    checkpoints = sorted([p for p in output.glob("checkpoint-*") if (p / "complete.json").exists()],
                         key=lambda p: int(p.name.split("-")[-1]))
    if resume and checkpoints:
        state = torch.load(checkpoints[-1] / "training_state.pt", map_location="cpu", weights_only=True)
        if state["signature"] != signature:
            raise RuntimeError("Checkpoint settings differ. Use a new run directory; do not mix experiments.")
        set_peft_model_state_dict(model, state["adapter"])
        optimizer.load_state_dict(state["optimizer"])
        for value in optimizer.state.values():
            for key, tensor in value.items():
                if torch.is_tensor(tensor):
                    # AdamW's non-capturable step counter stays on CPU. Moving
                    # every state tensor to CUDA breaks some GPU resume paths.
                    value[key] = tensor.cpu() if key == 'step' else tensor.to(device)
        scaler.load_state_dict(state["scaler"])
        generator.set_state(state["generator"])
        step, history, prior_seconds = state["step"], state["history"], state["seconds"]
        prior_peak = state.get("peak_allocated_mib", 0.0)
        print(f"Resumed optimizer update {step}.", flush=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    started = time.perf_counter()
    updates = config["max_train_steps"]
    accumulation = config["gradient_accumulation_steps"]
    while step < updates:
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for _ in range(accumulation):
            indices = torch.randint(len(latents), (config["train_batch_size"],), generator=generator)
            z = latents[indices].to(device=device, dtype=dtype)
            condition = hidden[indices].to(device=device, dtype=dtype)
            noise = torch.randn(z.shape, generator=generator).to(device=device, dtype=dtype)
            timestep = torch.randint(noise_scheduler.config.num_train_timesteps, (len(z),), generator=generator).to(device)
            noisy = noise_scheduler.add_noise(z, noise, timestep)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                prediction = model(noisy, timestep, encoder_hidden_states=condition).sample
                target = noise if noise_scheduler.config.prediction_type == "epsilon" else noise_scheduler.get_velocity(z, noise, timestep)
                loss = F.mse_loss(prediction.float(), target.float())
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite loss at update {step}; last complete checkpoint is preserved.")
            total_loss += loss.item() / accumulation
            scaler.scale(loss / accumulation).backward()
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        if not torch.isfinite(norm):
            raise RuntimeError(f"Nonfinite gradients at update {step}; resume the last complete checkpoint.")
        scaler.step(optimizer)
        scaler.update()
        step += 1
        history.append({"step": step, "loss": total_loss})
        if step == 1 or step % 25 == 0 or step == updates:
            print(f"Update {step}/{updates}, noise MSE {total_loss:.6f}", flush=True)
        if step % config["checkpointing_steps"] == 0 or step == updates:
            if device.type == "cuda":
                torch.cuda.synchronize()
            folder = output / f"checkpoint-{step}"
            folder.mkdir(exist_ok=True)
            seconds = prior_seconds + time.perf_counter() - started
            adapter = {k: v.detach().cpu() for k, v in get_peft_model_state_dict(model).items()}
            torch.save({"adapter": adapter, "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                        "generator": generator.get_state(), "step": step, "history": history,
                        "signature": signature, "seconds": seconds,
                        "peak_allocated_mib": max(prior_peak, torch.cuda.max_memory_allocated()/2**20)
                        if device.type == "cuda" else 0.0}, folder / "training_state.pt")
            (folder / "complete.json").write_text(json.dumps({"step": step}))
            complete = sorted([p for p in output.glob("checkpoint-*") if (p / "complete.json").exists()],
                              key=lambda p: int(p.name.split("-")[-1]))
            for old in complete[:-config.get("checkpoint_limit", 2)]:
                # These are this run's resolved checkpoint directories, never a broad path.
                shutil.rmtree(old)
        if stop_after is not None and step >= stop_after:
            break
    if device.type == "cuda":
        torch.cuda.synchronize()
    seconds = prior_seconds + time.perf_counter() - started
    metrics = {"rank": rank, "steps": step, "final_loss": history[-1]["loss"],
               "total_parameters": sum(p.numel() for p in model.parameters()),
               "trainable_parameters": sum(p.numel() for p in parameters),
               "learning_rate": config["learning_rate"], "batch_size": config["train_batch_size"],
               "gradient_accumulation_steps": accumulation, "training_seconds": seconds,
               "effective_batch_size": config["train_batch_size"] * accumulation,
               "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
               "timing_scope": "Optimizer loop including checkpoint writes; excludes shared VAE/text caching and evaluation.",
               "gpu_hours": seconds / 3600 if device.type == "cuda" else None,
               "peak_allocated_mib": max(prior_peak, torch.cuda.max_memory_allocated() / 2**20) if device.type == "cuda" else None}
    (output / "loss_history.json").write_text(json.dumps(history, indent=2))
    (output / "training_metrics.json").write_text(json.dumps(metrics, indent=2))
    StableDiffusionPipeline.save_lora_weights(output,
        unet_lora_layers=convert_state_dict_to_diffusers(get_peft_model_state_dict(model)), safe_serialization=True)
    print(json.dumps(metrics, indent=2), flush=True)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--run", choices=["run1", "run2"], required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--resume-latest", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load((ROOT / args.config).read_text())
    if not torch.cuda.is_available():
        raise RuntimeError("Full diffusion training requires CUDA. Run the CPU smoke test instead on this machine.")
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    output = ROOT / config["output_root"] / args.run
    metadata = output / "run_config.json"
    if output.exists() and not metadata.exists() and any(output.iterdir()):
        # Preserve older trainer outputs instead of treating them as compatible.
        import datetime
        backup = output.with_name(output.name + '_legacy_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
        output.rename(backup)
        print(f'Preserved older run files at {backup}. A clean compatible run will start.', flush=True)
    fingerprint = {**config, "rank": args.rank}
    if metadata.exists() and json.loads(metadata.read_text()) != fingerprint:
        raise RuntimeError("This output folder contains a different configuration. Choose a new output_root.")
    output.mkdir(parents=True, exist_ok=True)
    metadata.write_text(json.dumps(fingerprint, indent=2))
    if (output / "training_metrics.json").exists() and (output / "pytorch_lora_weights.safetensors").exists():
        metrics = json.loads((output / "training_metrics.json").read_text())
        if metrics["steps"] == config["max_train_steps"]:
            print("Already completed; preserving the adapter and results.")
            print(json.dumps(metrics, indent=2))
            return
    cached = prepare_cache(config)
    scheduler = DDPMScheduler.from_pretrained(config["base_model"], subfolder="scheduler",revision=config['base_revision'])
    model = UNet2DConditionModel.from_pretrained(config["base_model"], subfolder="unet", torch_dtype=torch.float16,
        revision=config['base_revision'],variant=config['variant']).to("cuda")
    attach_lora(model, args.rank)
    train_updates(model, cached["latents"], cached["hidden"], scheduler, output, config, args.rank,
                  resume=args.resume_latest)


if __name__ == "__main__":
    main()
