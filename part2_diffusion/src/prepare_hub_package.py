"""Prepare only real model and image artifacts for upload. No authentication or upload occurs here."""
from __future__ import annotations
import json
from pathlib import Path
import shutil
import yaml
from submission import audit

ROOT = Path(__file__).resolve().parents[1]


def prepare(root=ROOT):
    result = audit(root)
    for name in ["run1_cuda_training", "run2_cuda_training", "120_images_same_protocol", "kid_clip_lpips_for_all_models"]:
        if not result["checks"][name]:
            raise ValueError("Cannot package an incomplete experiment: " + name)
    cfg = yaml.safe_load((root / "config.yaml").read_text())
    output = root / cfg["output_root"]
    package = output / "hub_export"
    by_model = {r["model"]: r for r in result["evaluation"]["metrics"]}
    review = result["review"]
    for row in result["training"]:
        name = row["model"]
        folder = package / name
        folder.mkdir(parents=True, exist_ok=True)
        for filename in ["pytorch_lora_weights.safetensors", "training_metrics.json", "run_config.json"]:
            shutil.copy2(output / name / filename, folder / filename)
        score = by_model[name]
        card = f'''---
library_name: diffusers
base_model: {cfg['base_model']}
pipeline_tag: text-to-image
tags: [lora, stable-diffusion, research]
---
# DATA 266 Lab 2: {name}, rank {row['rank']}

This {'is the selected domain-style adapter' if review.get('final_run') == name else 'is the comparison adapter'} for a non-commercial course experiment.
The base is SD 1.5 at revision `{cfg['base_revision']}`. Only UNet attention LoRA weights were trained.

## Data and training
Naruto BLIP Captions, revision `1ed13e8ef280bd45e3bbac4cfa8bbd9d64ec9f89`;
1,098 training and 123 held-out images, seed 266, RGB 512 x 512.
Rank {row['rank']}; {row['trainable_parameters']:,} trainable parameters; AdamW LR {row['learning_rate']};
{row['steps']} updates; batch {row['batch_size']}, accumulation {row['gradient_accumulation_steps']}.
Measured optimizer-loop time: {row['training_seconds']:.2f} seconds / {row['gpu_hours']:.4f} GPU-hours;
peak allocated VRAM {row['peak_allocated_mib']:.2f} MiB. This excludes shared feature caching and evaluation.

## Results
KID {score['kid_mean']:.6f} (subset SD {score['kid_std']:.6f}); CLIP {score['clip_score_mean']:.4f};
LPIPS across seeds {score['lpips_seed_pair_mean_256px']:.6f}. KID uses 20 in-domain generations versus 123 test images.
All models use 20-step DDIM, CFG 7.5, eta 0, seeds 0 and 1. Filtered outputs remain counted.
Independent three-rater judgments are pending and no human win rate is claimed.

## Use and limitations
Load the pinned base with `StableDiffusionPipeline.from_pretrained`, then use `pipe.load_lora_weights`
with this repository. Keep the base safety checker enabled.
Anime coverage and automated caption errors limit generalization. Visible failures include color-attribute
binding and weapon-count errors; a black filtered sample is not proof of a false positive.

## Safety and rights
For non-commercial research only. Naruto characters and source imagery are copyrighted;
the dataset card has no explicit license field. No ownership or extra redistribution rights are claimed.
Comply with SD 1.5's CreativeML Open RAIL-M restrictions and verify permission for any other use.
The safety checker and CLIP nearest-neighbor checks are limited safeguards, not guarantees.
'''
        (folder / "README.md").write_text(card, encoding="utf-8")
    folder = package / "images"
    folder.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((output / "generations/manifest.json").read_text())
    for row in manifest:
        shutil.copy2(output / "generations" / row["file"], folder / row["file"])
    (folder / "metadata.jsonl").write_text("".join(json.dumps({"file_name": r["file"], **r}) + "\n" for r in manifest))
    (folder / "README.md").write_text("# DATA 266 Lab 2 controlled image comparisons\n\n"
        "120 generated images: base SD 1.5, rank-4 LoRA and rank-8 LoRA; twenty fixed prompts with seeds 0 and 1.\n"
        "Metadata records the model, prompt, category, seed, sampler settings, safety flag and protocol fingerprint.\n"
        "Black safety-filtered outputs are retained. No original training images or rater identities are uploaded.\n\n"
        "For non-commercial course research only. Source anime characters are copyrighted and the dataset card\n"
        "does not state an explicit license. Generated images do not grant rights to those characters or designs.\n")
    print("Prepared local upload packages at:", package)
    return package


if __name__ == "__main__":
    prepare()
