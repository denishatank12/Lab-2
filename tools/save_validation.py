"""Save a small validation record, without committing datasets or generated images."""
from pathlib import Path
import csv
import hashlib
import importlib.metadata
import json
import platform
import sys
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'part2_diffusion/src'))
from prepare_data import prepared

record=json.loads((ROOT/'runtime_checks/notebook_validation.json').read_text())
assert all(r['error_outputs']==0 for r in record['notebooks'])
record['host']={'python':platform.python_version(),'system':platform.platform(),
                'cuda_gpu_present':torch.cuda.is_available()}
record['versions']={name:importlib.metadata.version(name) for name in
                   ['torch','torchvision','diffusers','transformers','peft','datasets','huggingface-hub']}
data=ROOT/'part2_diffusion/data/processed'
if prepared(data):
    record['real_dataset_preparation']=json.loads((data/'data_summary.json').read_text())
else:
    record['real_dataset_preparation']='Not completed by this validation run.'
with (ROOT/'part1_rag/outputs/submission.csv').open(newline='',encoding='utf-8') as handle:
    record['part1_validated_submission_rows']=len(list(csv.DictReader(handle)))
red=json.loads((ROOT/'runtime_checks/red_team.json').read_text())
record['text_red_team_cases']={'cases':len(red),'passed':sum(c['pass'] for c in red),
    'scope':'Text-only guardrail tests; not image-attack robustness.'}
paths=[ROOT/'requirements.txt',ROOT/'part2_diffusion/config.yaml']
for folder in ['part1_rag/src','part2_diffusion/src','part3_agents/src','tools']:
    paths.extend((ROOT/folder).glob('*.py'))
record['source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
record['not_verified']=['Full Stable Diffusion training on RTX 4050 or T4',
    'Full pretrained-model image generation and numerical KID/CLIP/LPIPS results',
    'Model-backed agents on CUDA','Independent human judgments','Hugging Face uploads',
    'Complete graded lab submission']
out=ROOT/'validation/cpu_smoke_results.json'
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
print(out.read_text(),flush=True)
