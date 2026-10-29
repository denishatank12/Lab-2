from pathlib import Path
import sys, yaml

ROOT = Path(__file__).resolve().parents[1]
cfg = yaml.safe_load((ROOT / 'config.yaml').read_text())
assert cfg['seed'] == 266
assert cfg['resolution'] == 512
assert cfg['test_fraction'] == 0.10
assert cfg['max_train_steps'] > 0
assert cfg['train_batch_size'] == 1
assert cfg['gradient_checkpointing'] is True
print('PASS: Part 2 config, deterministic split, and memory-safe settings validated')
