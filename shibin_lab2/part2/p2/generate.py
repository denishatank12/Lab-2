"""Task 2.3 - generate 20 prompts x 2 seeds for the base model and both runs (same sampler settings).

NSFW filter (safety analysis): every image is checked by two filters after generation:
  * a dedicated NSFW image classifier (Falconsai/nsfw_image_detection, ViT) - this one decides: an image with
    p(nsfw) >= 0.5 is filtered (kept only blurred, for audit) and excluded from all metrics;
  * the Stable Diffusion safety checker - recorded for comparison only; on our first run it blacked out 9-14 of 40
    harmless anime portraits per fine-tuned model (a known false-positive pattern), which corrupted KID/CLIP.
All models also share the same negative prompt.
"""
from __future__ import annotations

import json
import os
import time

import torch

from .config import P2Config
from .train import device_and_dtype


NEGATIVE = "nsfw, nudity, lowres, blurry, watermark"
NSFW_MODEL = "Falconsai/nsfw_image_detection"


def load_pipe(cfg: P2Config, lora_dir: str | None = None):
    from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
    dev, wdt = device_and_dtype()
    dt = torch.float16 if dev == "cuda" else torch.float32
    kw = dict(torch_dtype=dt, safety_checker=None, requires_safety_checker=False)  # filters run separately below
    try:
        pipe = StableDiffusionPipeline.from_pretrained(cfg.base_model, variant="fp16", **kw)
    except (OSError, ValueError):
        pipe = StableDiffusionPipeline.from_pretrained(cfg.base_model, **kw)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    if lora_dir:
        pipe.load_lora_weights(lora_dir)
    pipe.set_progress_bar_config(disable=True)
    return pipe.to(dev)


class NSFWFilters:
    def __init__(self, cfg: P2Config, log=print):
        from transformers import pipeline
        dev = 0 if torch.cuda.is_available() else -1
        self.clf = pipeline("image-classification", model=NSFW_MODEL, device=dev)
        self.sd = self.fe = None
        try:
            from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
            from transformers import CLIPImageProcessor
            try:
                self.sd = StableDiffusionSafetyChecker.from_pretrained(cfg.base_model, subfolder="safety_checker", variant="fp16")
            except (OSError, ValueError):
                self.sd = StableDiffusionSafetyChecker.from_pretrained(cfg.base_model, subfolder="safety_checker")
            self.sd = self.sd.to("cuda" if dev == 0 else "cpu", dtype=torch.float32).eval()
            self.fe = CLIPImageProcessor.from_pretrained(cfg.base_model, subfolder="feature_extractor")
        except Exception as e:
            log(f"[nsfw] SD safety checker not available ({type(e).__name__}); using the classifier only")

    def __call__(self, image) -> tuple[float, bool | None]:
        import numpy as np
        scores = {d["label"].lower(): d["score"] for d in self.clf(image)}
        p = float(scores.get("nsfw", 0.0))
        sd = None
        if self.sd is not None:
            x = self.fe(images=[image], return_tensors="pt").pixel_values.to(self.sd.device)
            with torch.no_grad():
                _, flags = self.sd(images=np.asarray(image, dtype=np.float32)[None] / 255.0, clip_input=x)
            sd = bool(flags[0])
        return p, sd


def generate_all(cfg: P2Config, prompts: list[dict], models: dict, out_dir: str, log=print) -> list[dict]:
    """models: {"base": None, "run1_lora_r4": lora_dir, ...}. Returns one row per image."""
    from PIL import ImageFilter
    dev, _ = device_and_dtype()
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    filters = NSFWFilters(cfg, log)
    for model_name, lora in models.items():
        pipe = load_pipe(cfg, lora)
        t0 = time.time()
        for p in prompts:
            for seed in cfg.gen_seeds:
                g = torch.Generator(device="cpu").manual_seed(seed)
                img = pipe(p["prompt"], negative_prompt=NEGATIVE, num_inference_steps=cfg.steps,
                           guidance_scale=cfg.guidance, generator=g, height=cfg.resolution, width=cfg.resolution).images[0]
                p_nsfw, sd_flag = filters(img)
                filtered = p_nsfw >= 0.5
                path = os.path.join(out_dir, f"{model_name}__{p['id']}__s{seed}.png")
                (img.filter(ImageFilter.GaussianBlur(24)) if filtered else img).save(path)
                rows.append({"model": model_name, "lora": lora or "", "prompt_id": p["id"], "kind": p["kind"],
                             "prompt": p["prompt"], "negative_prompt": NEGATIVE, "seed": seed, "steps": cfg.steps,
                             "guidance": cfg.guidance, "sampler": "DPMSolverMultistep", "resolution": cfg.resolution,
                             "nsfw_prob": round(p_nsfw, 4), "nsfw_filtered": filtered, "sd_checker_flag": sd_flag,
                             "path": path})
        secs = time.time() - t0
        n = len(prompts) * len(cfg.gen_seeds)
        mine = [r for r in rows if r["model"] == model_name]
        log(f"[generate] {model_name}: {n} images in {secs:.0f}s ({secs / n:.1f} s/image); "
            f"filtered by the NSFW classifier: {sum(r['nsfw_filtered'] for r in mine)}, "
            f"SD safety checker would flag: {sum(bool(r['sd_checker_flag']) for r in mine)}")
        del pipe
        if dev == "cuda":
            torch.cuda.empty_cache()
    with open(os.path.join(out_dir, "metadata.jsonl"), "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps({k: (os.path.basename(v) if k == "path" else v) for k, v in r.items()}) + "\n")
    return rows
