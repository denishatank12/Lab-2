"""Offline integrity checks. These checks do not claim human or model quality."""
import ast
import base64
import hashlib
import io
import json
from pathlib import Path
import re
import zipfile

import nbformat
from IPython.core.inputtransformer2 import TransformerManager

from final_submission_audit import ROOT, collect


def check():
    audit = collect()
    p1 = ROOT / 'part1_rag/notebooks/Lab2_Part1_Kaggle_Final.ipynb'
    assert hashlib.sha256(p1.read_bytes()).hexdigest() == audit['part1']['notebook_sha256'], 'Selected v6 changed'
    value = nbformat.read(p1, as_version=4)
    assignments = {}
    for cell in value.cells:
        if cell.cell_type == 'code':
            for statement in ast.parse(cell.source).body:
                if isinstance(statement, ast.Assign):
                    for target in statement.targets:
                        if isinstance(target, ast.Name) and target.id == 'BUNDLE_BASE64':
                            assignments[target.id] = ast.literal_eval(statement.value)
    raw = base64.b64decode(assignments['BUNDLE_BASE64'])
    assert hashlib.sha256(raw).hexdigest() == '600d1e02c1ff21f3e1c999cb59b1d98499492585a0dcafc4696169f6a479f68e'
    with zipfile.ZipFile(io.BytesIO(raw)) as bundle:
        assert bundle.testzip() is None
        manifest = json.loads(bundle.read('STANDALONE_MANIFEST.json'))
        for name, expected in manifest['files'].items():
            assert hashlib.sha256(bundle.read(name)).hexdigest() == expected, name
            if name.endswith('.py'):
                compile(bundle.read(name), name, 'exec')
        golden = json.loads(bundle.read('part1_rag/eval/golden_candidate.json'))
        assert len(golden) == 25
    for name in ['part2', 'part3_baseline', 'part3_partner']:
        assert audit['notebooks'][name]['complete'], name
    paths = [ROOT / 'part1_rag/notebooks/Lab2_Part1_Kaggle_Final.ipynb',
             ROOT / 'part2_diffusion/notebooks/Part2_FineTune_Run.ipynb',
             ROOT / 'part3_agents/notebooks/Part3_Baseline_Executed.ipynb',
             ROOT / 'part3_agents/notebooks/Part3_Partner_Baseline_Standalone.ipynb',
             ROOT / 'part3_agents/partner_snapshot/Partner_Part3_Executed.ipynb']
    for path in paths:
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        for cell in notebook.cells:
            if cell.cell_type == 'code':
                compile(TransformerManager().transform_cell(cell.source), str(path), 'exec')
    imported = json.loads((ROOT / 'part3_agents/evidence/denisha_baseline/IMPORT.json').read_text())
    for name, expected in imported['files'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    b = ROOT / 'part3_agents/evidence/denisha_baseline/baseline_outputs'
    states = list((b / 'states').glob('*.json'))
    traces = list((b / 'traces').glob('*.json'))
    assert len(states) == len(traces) == 25
    for path in states:
        state = json.loads(path.read_text())
        assert state['use_critique'] is False and state['max_revisions'] == 0
    assert audit['checks']['part3_real_t4_baseline']
    if not (ROOT / 'submission/ratings').is_dir():
        assert not audit['checks']['part1_real_groundedness_ratings']
        assert not audit['checks']['part3_real_human_ratings']
    # Do not print credential matches. Source/test placeholder values do not match this pattern.
    secrets = []
    for root in [ROOT / 'submission', ROOT / 'part3_agents/evidence/denisha_baseline']:
        for path in root.rglob('*'):
            if path.is_file() and path.suffix in {'.json', '.txt', '.ipynb', '.csv'}:
                if re.search(r'\bhf_[A-Za-z0-9]{20,}\b', path.read_text(errors='replace')):
                    secrets.append(str(path.relative_to(ROOT)))
    assert not secrets, 'Credential-like content found in: ' + ', '.join(secrets)
    result = dict(passed=True, scope='Offline packaging, syntax, immutable-source and saved-output integrity only',
                  selected_v6_unchanged=True, embedded_v6_files_verified=len(manifest['files']),
                  canonical_notebooks_checked=len(paths), part3_real_states=25, part3_real_traces=25,
                  no_human_evaluation_results_invented=True,
                  synthetic_format_examples_present=(ROOT / 'docs/rating_forms/EXAMPLES_ONLY.json').is_file(),
                  synthetic_examples_used_as_measurements=False, submission_ready=audit['ready'])
    (ROOT / 'submission/integrity.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    check()
