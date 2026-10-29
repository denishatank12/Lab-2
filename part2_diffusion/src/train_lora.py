"""GPU-safe wrapper around the pinned Diffusers LoRA trainer.

Using the official trainer keeps the training loop aligned with the published
Stable Diffusion implementation while this wrapper owns fixed seeds, two-run
configuration, checkpointing, and resume behavior.
"""
from __future__ import annotations
import argparse, os, subprocess, sys, urllib.request
from pathlib import Path
import yaml

TRAINER_URL = 'https://raw.githubusercontent.com/huggingface/diffusers/v0.31.0/examples/text_to_image/train_text_to_image_lora.py'

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--config', default='config.yaml'); ap.add_argument('--run', required=True, choices=['run1','run2']); ap.add_argument('--rank', type=int, required=True); ap.add_argument('--resume-latest', action='store_true')
    a = ap.parse_args(); root = Path(__file__).resolve().parents[1]; c = yaml.safe_load((root / a.config).read_text())
    trainer = root / 'tools' / 'train_text_to_image_lora.py'; trainer.parent.mkdir(exist_ok=True)
    if not trainer.exists(): urllib.request.urlretrieve(TRAINER_URL, trainer)
    data = root / 'data' / 'processed' / 'train'
    if not data.exists(): raise FileNotFoundError('Run prepare_data.py first')
    out = root / c['output_root'] / a.run; out.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(trainer), '--pretrained_model_name_or_path', c['base_model'], '--train_data_dir', str(data), '--image_column', 'image', '--caption_column', 'text', '--resolution', str(c['resolution']), '--random_flip', '--train_batch_size', str(c['train_batch_size']), '--gradient_accumulation_steps', str(c['gradient_accumulation_steps']), '--max_train_steps', str(c['max_train_steps']), '--learning_rate', str(c['learning_rate']), '--lr_scheduler', 'cosine', '--lr_warmup_steps', '100', '--checkpointing_steps', str(c['checkpointing_steps']), '--rank', str(a.rank), '--output_dir', str(out), '--seed', str(c['seed']), '--mixed_precision', c['mixed_precision'], '--gradient_checkpointing']
    if c.get('use_8bit_adam'): cmd += ['--use_8bit_adam']
    if a.resume_latest: cmd += ['--resume_from_checkpoint', 'latest']
    print('Launching:', ' '.join(cmd)); subprocess.run(cmd, check=True, cwd=root)
if __name__ == '__main__': main()
