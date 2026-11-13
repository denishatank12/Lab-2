"""Atomic per-item checkpoints. Changed inputs never reuse old measured results."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import uuid
import platform
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_hash(path):
    path = Path(path)
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def source_hashes(paths):
    return {str(Path(p).relative_to(ROOT)): file_hash(p) for p in paths}


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temporary.open('w', encoding='utf-8') as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def experiment_sources(part):
    names = ['tools/vlm_runtime.py', 'tools/response_schemas.py', 'tools/run_checkpoint.py', 'requirements.txt']
    if part == 'part1_rag':
        names += ['part1_rag/src/multimodal.py', 'part1_rag/src/rag_pipeline.py',
                  'part1_rag/src/generate_submission.py', 'part1_rag/src/evaluate_rag.py', 'tools/rag_model.py']
    else:
        names += ['part3_agents/src/agent_system.py', 'part3_agents/src/model_runtime.py',
                  'part3_agents/src/run_scenarios.py', 'part3_agents/src/run_red_team.py',
                  'part3_agents/eval/scenarios.json', 'part3_agents/eval/red_team_cases.json',
                  'part2_diffusion/config.yaml']
    result=source_hashes(ROOT / name for name in names)
    if part == 'part1_rag':
        result['rag_runtime_options'] = {key: os.environ.get(key, '') for key in
            ['LAB2_RAG_PROFILE', 'LAB2_RAG_NATURAL_JSON']}
    if part!='part1_rag':
        registry=json.loads((ROOT/'part2_diffusion/evidence/hub_links.json').read_text())
        # Grader-access confirmations do not change model outputs. Only bind the selected adapter.
        result['selected_hub_adapter']={key:registry[key] for key in ['run1','run1_commit','run1_adapter_sha256']}
    return result


def verify_inputs(inputs):
    for key,row in inputs.items():
        if file_hash(row['path']) != row.get('file_sha256'):
            raise ValueError(f'Input {key} changed after its manifest was written. Recreate inputs before continuing.')
    return inputs


def runtime_identity():
    import torch
    identity=dict(torch=torch.__version__,python=platform.python_version(),host=platform.node(),cuda=torch.cuda.is_available())
    if identity['cuda']:
        identity['gpu']=torch.cuda.get_device_name(0)
        identity['total_memory']=torch.cuda.get_device_properties(0).total_memory
        try:
            identity['gpu_uuid']=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],
                text=True,timeout=10).strip()
        except (OSError,subprocess.SubprocessError):
            identity['gpu_uuid']='unavailable'
    return identity


def select_run(part):
    """A matching restart chooses the same run; different code/hardware does not."""
    signature=dict(sources=experiment_sources(part),runtime=runtime_identity())
    folder=ROOT/part/'outputs/v2/runs'/digest(signature)[:20]
    folder.mkdir(parents=True,exist_ok=True)
    atomic_json(ROOT/part/'outputs/v2/latest_run.json',dict(folder=str(folder),signature=signature))
    return folder


class ItemCheckpoint:
    def __init__(self, folder, protocol):
        protocol={**protocol,'runtime':runtime_identity()}
        self.folder = Path(folder) / digest(protocol)
        self.folder.mkdir(parents=True, exist_ok=True)
        marker = self.folder / 'protocol.json'
        if marker.exists() and json.loads(marker.read_text()) != protocol:
            raise ValueError('Checkpoint protocol mismatch. Existing results were not overwritten.')
        if not marker.exists():
            atomic_json(marker, protocol)

    def path(self, key):
        return self.folder / (digest(key) + '.json')

    def get(self, key):
        path = self.path(key)
        if not path.is_file():
            return None
        try:
            saved = json.loads(path.read_text())
            if saved.get('key') != key or saved.get('result_sha256') != digest(saved['result']):
                return None
            if any(file_hash(name) != value for name, value in saved.get('artifacts', {}).items()):
                return None
            return saved['result']
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def put(self, key, result, artifacts=()):
        files = {str(Path(path).resolve()): file_hash(path) for path in artifacts}
        if any(value is None for value in files.values()):
            raise FileNotFoundError('Cannot mark an item complete while its output artifact is missing.')
        atomic_json(self.path(key), dict(key=key, result=result, result_sha256=digest(result),
            artifacts=files, completed_utc=datetime.now(timezone.utc).isoformat()))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--select-run',required=True,choices=['part3_agents'])
    args=parser.parse_args()
    print('Resumable run folder:',select_run(args.select_run))
