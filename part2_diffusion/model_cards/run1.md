---
library_name: diffusers
base_model: stable-diffusion-v1-5/stable-diffusion-v1-5
pipeline_tag: text-to-image
tags: [lora, stable-diffusion, research]
---
# DATA 266 Lab 2: run1, rank 4

This is the selected domain-style adapter for a non-commercial course experiment.
The base is SD 1.5 at revision `451f4fe16113bff5a5d2269ed5ad43b0592e9a14`. Only UNet attention LoRA weights were trained.

## Data and training
Naruto BLIP Captions, revision `1ed13e8ef280bd45e3bbac4cfa8bbd9d64ec9f89`;
1,098 training and 123 held-out images, seed 266, RGB 512 x 512.
Rank 4; 797,184 trainable parameters; AdamW LR 0.0001;
1200 updates; batch 1, accumulation 4.
Measured optimizer-loop time: 764.54 seconds / 0.2124 GPU-hours;
peak allocated VRAM 1949.88 MiB. This excludes shared feature caching and evaluation.

## Results
KID 0.068097 (subset SD 0.007952); CLIP 29.5355;
LPIPS across seeds 0.754682. KID uses 20 in-domain generations versus 123 test images.
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
