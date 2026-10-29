# Lab 2 Part 2 - Stable Diffusion fine-tuning

This implementation uses `lambdalabs/naruto-blip-captions` and Stable Diffusion 1.5.
It prepares a deterministic 90/10 split with seed 266, trains two LoRA runs with
the same step budget, checkpoints every 250 steps, and supports resuming from the
latest checkpoint. The default settings are deliberately conservative for a
6-8 GB RTX 4050; a 16 GB T4 is the required grading target.

```bash
cd part2_diffusion
python -m pip install -r requirements.txt
python src/smoke_test.py
python src/prepare_data.py --output-dir data/processed
python src/train_lora.py --config config.yaml --run run1 --rank 4
python src/train_lora.py --config config.yaml --run run2 --rank 8
python src/generate_eval.py --config config.yaml
python src/evaluate.py --config config.yaml
```

Set `MAX_TRAIN_STEPS` or edit `config.yaml` before an overnight run. Do not
change the data split or step budget between the two runs. Model weights are
ignored by Git and should be uploaded to Hugging Face.
