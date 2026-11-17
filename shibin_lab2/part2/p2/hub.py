"""Push the trained LoRA adapters and the generated images to Hugging Face (C1.6, Task 2.2.3 / 2.3.3).

Needs a token with write access: `huggingface-cli login` once, or the HF_TOKEN environment variable
(on Kaggle: Add-ons -> Secrets). Without a token every push is skipped with a message.
"""
from __future__ import annotations

import os

from .config import P2Config


def whoami() -> str | None:
    try:
        from huggingface_hub import HfApi
        return HfApi().whoami()["name"]
    except Exception:
        return None


MODEL_CARD = """---
license: creativeml-openrail-m
base_model: {base}
tags: [stable-diffusion, lora, text-to-image, diffusers, naruto]
datasets: [{dataset}]
---
# {repo} - {method}

LoRA adapter for **{base}** fine-tuned on **{dataset}** (DATA 266 Lab 2, Part 2).

## Data
{n_train} training images (90 % split, seed 266) after removing {removed} unusable image(s); 512x512 centre crops;
BLIP captions as prompts. The test split ({n_test} images) was used only for evaluation.

## Training
| setting | value |
|---|---|
| method | {method} |
| trainable parameters | {trainable:,} |
| steps / batch size / resolution | {steps} / {batch} / {res} |
| learning rate | {lr} (AdamW, 100 warm-up steps, cosine), min-SNR gamma 5 |
| precision | {dtype} weights, float32 adapters |
| peak GPU memory | {peak} GB on {device} |
| training time | {minutes} min ({gpu_hours} GPU-hours) |

## Results (20 fixed prompts x 2 seeds, DPM-Solver 30 steps, guidance 7.5)
{results}

## Use
```python
from diffusers import StableDiffusionPipeline
pipe = StableDiffusionPipeline.from_pretrained("{base}").to("cuda")
pipe.load_lora_weights("{repo}")
pipe("a ninja with spiky blonde hair and a headband").images[0]
```

## Limitations and license
Naruto characters are copyrighted; the training images come from Narutopedia via the BLIP-captioned dataset,
whose card states no license. This adapter is for **non-commercial research and teaching only**. It inherits the
CreativeML OpenRAIL-M license and the biases of SD 1.5; the captions are short, mostly about one male character,
and repeat phrases ("a man with ..."), so prompts outside that pattern follow the style less reliably. Outputs were
screened with an NSFW image classifier (filtered images are blurred and excluded from the metrics); the SD safety
checker's flags are recorded for comparison.
"""


def push_lora(cfg: P2Config, stats: dict, results_md: str, data_info: dict, private: bool = True, log=print) -> str | None:
    from huggingface_hub import HfApi
    user = cfg.hf_user or whoami()
    if not user:
        log("[hub] no Hugging Face login/token - skipping model upload (run `huggingface-cli login` or set HF_TOKEN)")
        return None
    repo = f"{user}/{cfg.hf_model_prefix}-{stats['run'].replace('_', '-')}"
    api = HfApi()
    api.create_repo(repo, private=private, exist_ok=True)
    card = MODEL_CARD.format(base=cfg.base_model, dataset=cfg.dataset, repo=repo, method=stats["method"],
                             n_train=data_info.get("train", "?"), n_test=data_info.get("test", "?"),
                             removed=data_info.get("removed", "?"), trainable=stats["trainable_params"],
                             steps=stats["steps"], batch=stats["batch_size"], res=stats["resolution"], lr=stats["learning_rate"],
                             dtype=stats["dtype"], peak=stats["peak_gpu_memory_gb"], device=stats["device"],
                             minutes=round(stats["train_time_s"] / 60, 1), gpu_hours=stats["gpu_hours"], results=results_md)
    with open(os.path.join(stats["lora_dir"], "README.md"), "w", encoding="utf-8") as fh:
        fh.write(card)
    api.upload_folder(folder_path=stats["lora_dir"], repo_id=repo, commit_message="LoRA adapter + model card")
    log(f"[hub] pushed {repo}")
    return repo


def push_images(cfg: P2Config, rows: list[dict], private: bool = True, log=print) -> str | None:
    user = cfg.hf_user or whoami()
    if not user:
        log("[hub] no Hugging Face login/token - skipping image dataset upload")
        return None
    from datasets import Dataset, Image as HFImage
    repo = f"{user}/{cfg.hf_images_repo}"
    cols = ["model", "prompt_id", "kind", "prompt", "negative_prompt", "seed", "steps", "guidance", "sampler", "lora",
            "nsfw_prob", "nsfw_filtered", "sd_checker_flag"]
    data = {c: [r[c] for r in rows] for c in cols}
    data["image"] = [r["path"] for r in rows]
    Dataset.from_dict(data).cast_column("image", HFImage()).push_to_hub(repo, private=private)
    log(f"[hub] pushed {len(rows)} images to datasets/{repo}")
    return repo
