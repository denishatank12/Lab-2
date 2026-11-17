"""Execute the actual one-file notebook with a fresh, clean CPU package profile.

Retains its temp folder and real logs for inspection. Never claims GPU/model
quality, adapter inference, human ratings, or a completed Part 3 submission.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import nbformat as nb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from overnight_runner import execute_saved


def main(directory=None, record_path=None):
    work = Path(directory).resolve() if directory else Path(tempfile.mkdtemp(prefix='lab2-p3-clean-smoke-'))
    work.mkdir(parents=True, exist_ok=True)
    source = ROOT / 'part3_agents/notebooks/Part3_Partner_Baseline_Standalone.ipynb'
    notebook = nb.read(source, as_version=4)
    notebook.cells[1].source = notebook.cells[1].source.replace('SMOKE_ONLY = False', 'SMOKE_ONLY = True', 1)
    # execute_saved starts the notebook in this empty working directory. Its
    # embedded code must supply everything else, including the isolated deps.
    local = work / 'Standalone_CPU_Smoke.ipynb'
    saved = work / 'Standalone_CPU_Smoke_Executed.ipynb'
    nb.write(notebook, local)
    print('Clean standalone smoke directory:', work, flush=True)
    try:
        execute_saved(local, saved, working_dir=work)
    except Exception as error:
        diagnostic = str(error)
        (work / 'SMOKE_FAILURE.txt').write_text(diagnostic)
        print('Smoke failed; complete saved notebook:', saved, flush=True)
        print('Diagnostic:', work / 'SMOKE_FAILURE.txt', flush=True)
        print(diagnostic[-2500:], flush=True)
        raise SystemExit(1) from None
    executed = nb.read(saved, as_version=4)
    codes = [c for c in executed.cells if c.cell_type == 'code']
    errors = [o for c in codes for o in c.outputs if o.output_type == 'error']
    runs = list(work.glob('Lab2_Part3_Baseline_*'))
    output = work / ('Lab2_Part3_Baseline_' + notebook.metadata.baseline_bundle_sha256[:16])
    if not output.is_dir():
        raise RuntimeError('The current source-verified smoke folder is missing.')
    environment = json.loads((output / 'runtime_environment.json').read_text())
    inner = nb.read(output / 'Baseline_SMOKE_Executed.ipynb', as_version=4)
    assert not errors
    assert all(c.execution_count is not None for c in codes)
    assert inner.metadata.execution.completed and inner.metadata.execution.mock_only
    assert not environment['foreign_site_paths']
    assert not (output / 'baseline_outputs/baseline_metrics.json').exists()
    result = dict(mock_only=True, full_standalone_cpu_smoke_passed=True,
        executed_code_cells=len(codes), saved_errors=len(errors), runtime=environment,
        executed_notebook=str(saved), result_zip_files=[str(p) for p in runs if p.suffix == '.zip'],
        not_verified=['CUDA compatibility', 'pretrained inference', 'human evaluation'])
    (work / 'CLEAN_SMOKE_RECORD.json').write_text(json.dumps(result, indent=2))
    if record_path:
        portable = {k: v for k, v in result.items() if k not in ['runtime', 'executed_notebook', 'result_zip_files']}
        portable.update(source_notebook_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            embedded_bundle_sha256=notebook.metadata.baseline_bundle_sha256,
            python=environment['python'], torch=environment['torch'], torchvision=environment['torchvision'],
            dependency_check=environment['dependency_check'], foreign_site_paths=environment['foreign_site_paths'],
            packages=environment['packages'],
            inner_executed_code_cells=sum(c.cell_type == 'code' and c.execution_count is not None for c in inner.cells))
        record_path = Path(record_path)
        record_path.parent.mkdir(parents=True, exist_ok=True)
        record_path.write_text(json.dumps(portable, indent=2) + '\n')
        print('Portable CPU-only evidence:', record_path, flush=True)
    print(json.dumps({k: v for k, v in result.items() if k != 'runtime'}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory')
    parser.add_argument('--record')
    arguments = parser.parse_args()
    main(arguments.directory, arguments.record)
