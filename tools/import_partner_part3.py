"""Preserve a teammate's public executed snapshot with explicit attribution.

This imports evidence, not newly executed work by Denisha. No partner repository
is modified and no human ratings are manufactured.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main(repository):
    repo = Path(repository).resolve()
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    base = repo / 'GPU_LAB 2'
    original = base / 'notebooks/p3_agents.ipynb'
    raw = original.read_bytes()
    if re.search(rb'\bhf_[A-Za-z0-9]{20,}\b', raw):
        raise ValueError('Possible credential in the source notebook; do not copy it to another repository.')
    nb = json.loads(raw)
    target = ROOT / 'part3_agents/partner_snapshot'
    target.mkdir(parents=True, exist_ok=True)
    files, differences, configs, gpu_outputs = {}, [], [], []
    for cell in nb['cells']:
        source = ''.join(cell['source'])
        if source.startswith('%%writefile '):
            header, body = source.split('\n', 1)
            name = header[len('%%writefile '):].strip()
            rel = PurePosixPath(name)
            if rel.is_absolute() or '..' in rel.parts:
                raise ValueError('Unsafe embedded source path.')
            destination = target / 'source' / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(body, encoding='utf-8')
            files['source/' + name] = sha(body.encode())
            current = base / 'part3' / name
            if current.is_file() and current.read_text().strip() != body.strip():
                differences.append(name)
        for output in cell.get('outputs', []):
            text = ''.join(output.get('text', []))
            for line in text.splitlines():
                if line.startswith('BackendConfig('):
                    expr = ast.parse(line, mode='eval').body
                    configs.append({kw.arg: ast.literal_eval(kw.value) for kw in expr.keywords})
                if "'gpu':" in line and "'python':" in line:
                    gpu_outputs.append(ast.literal_eval(line))
    if len(configs) != 1:
        raise ValueError('Cannot uniquely recover the executed backend configuration.')
    shutil.copy2(original, target / 'Partner_Part3_Executed.ipynb')
    files['Partner_Part3_Executed.ipynb'] = sha(raw)
    for name in ['p3_results.json', 'p3_results_table.csv', 'p3_redteam.csv',
                 'p3_human_task_success.csv', 'p3_human_critic_agreement.csv']:
        source = base / 'report/results' / name
        destination = target / 'reported_results' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        files['reported_results/' + name] = sha(source.read_bytes())
    for name in ['test_agents_mock.py', 'test_eval_mock.py']:
        source = base / 'part3/tests' / name
        # References, not runnable tests for the older executed snapshot.
        destination = target / 'latest_source_tests' / (name + '.reference')
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        files['latest_source_tests/' + name + '.reference'] = sha(source.read_bytes())
    codes = [c for c in nb['cells'] if c['cell_type'] == 'code']
    errors = [o for c in codes for o in c.get('outputs', []) if o.get('output_type') == 'error']
    manifest = dict(
        origin='https://github.com/Shibin-1020/data266-6945', commit=commit,
        folder='GPU_LAB 2/part3', executed_notebook='GPU_LAB 2/notebooks/p3_agents.ipynb',
        attribution='Partner Shibin; imported unchanged for team evidence. Not Denisha\'s own baseline execution.',
        source_policy='Use code embedded in the executed notebook, not later edited modules.',
        latest_modules_different_from_executed_snapshot=differences,
        code_cells=len(codes), executed_code_cells=sum(c.get('execution_count') is not None for c in codes),
        saved_error_count=len(errors), execution_identity=gpu_outputs,
        executed_backend_config=configs[0], human_results_verified=False,
        notes=['Reported results are automatic proxies, not completed human evaluation.',
               'The supplied real-model run used an RTX 4090, not verified single-16-GB execution.',
               'Latest-source tests include later changes; they are not proof of this older executed snapshot.',
               'Original input image bytes and per-request output files/traces are not included in the partner Git tree.'],
        files=files)
    (target / 'PROVENANCE.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in manifest.items() if k != 'files'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('repository', type=Path)
    main(parser.parse_args().repository)
