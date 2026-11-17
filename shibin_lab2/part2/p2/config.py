"""Settings for Part 2 (fine-tuning Stable Diffusion 1.5 on Naruto BLIP captions)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field

SEED = 266


def _env(name, default):
    v = os.environ.get(name)
    if v is None:
        return default
    if isinstance(default, bool):
        return v.strip().lower() in {"1", "true", "yes"}
    return type(default)(v)


@dataclass
class P2Config:
    dataset: str = field(default_factory=lambda: _env("P2_DATASET", "lambdalabs/naruto-blip-captions"))
    base_model: str = field(default_factory=lambda: _env("P2_BASE", "stable-diffusion-v1-5/stable-diffusion-v1-5"))
    work_dir: str = field(default_factory=lambda: _env("P2_WORK", "artifacts_p2"))
    resolution: int = field(default_factory=lambda: _env("P2_RES", 512))  # smaller only for CPU tests
    test_fraction: float = 0.10
    seed: int = SEED
    min_side: int = 128  # images smaller than this count as corrupted / unusable
    dup_hamming: int = 4  # perceptual-hash distance at or below which two images are near-duplicates

    # training (both runs share data, steps, batch size and learning rate; only the adapter differs)
    max_steps: int = field(default_factory=lambda: _env("P2_STEPS", 1500))
    batch_size: int = field(default_factory=lambda: _env("P2_BATCH", 4))
    lr: float = field(default_factory=lambda: _env("P2_LR", 1e-4))
    warmup_steps: int = 100
    snr_gamma: float = 5.0  # min-SNR loss weighting
    grad_checkpointing: bool = True
    runs: tuple = (("run1_lora_r4", 4), ("run2_lora_r32", 32))  # (name, LoRA rank); alpha = rank

    # generation (same sampler settings for every model)
    steps: int = 30
    guidance: float = 7.5
    gen_seeds: tuple = (0, 1)

    # Hugging Face (optional; needs HF_TOKEN)
    hf_user: str = field(default_factory=lambda: _env("HF_USER", ""))
    hf_model_prefix: str = field(default_factory=lambda: _env("HF_MODEL_PREFIX", "sd15-naruto"))
    hf_images_repo: str = field(default_factory=lambda: _env("HF_IMAGES_REPO", "lab2-p2-generated-images"))
