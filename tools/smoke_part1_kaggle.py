"""Execute the actual standalone notebook's CPU smoke in a fresh directory.

Uses the explicitly selected, already-installed interpreter. It does not test
CUDA, dependency installation, pretrained-model answers or leaderboard quality.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time

import nbformat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from overnight_runner import execute_saved
from run_checkpoint import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path,
        default=ROOT / 'part1_rag/notebooks/Lab2_Part1_Kaggle_Final.ipynb')
    parser.add_argument('--record', type=Path,
        default=ROOT / 'validation/part1_kaggle_final_smoke.json')
    args = parser.parse_args()
    source = args.source.resolve()
    folder = Path(tempfile.mkdtemp(prefix='lab2-part1-kaggle-final-'))
    target = folder / 'executed_outer.ipynb'
    os.environ.update(LAB2_SMOKE_ONLY='1', LAB2_DEVICE='cpu',
        LAB2_STANDALONE_PARENT=str(folder), LAB2_STANDALONE_TEST_PYTHON=sys.executable,
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    started = time.perf_counter()
    error = None
    try:
        execute_saved(source, target, working_dir=folder)
    except Exception as exc:
        error = str(exc)
        raise
    finally:
        saved = nbformat.read(target, as_version=4) if target.is_file() else None
        errors = [output for cell in saved.cells if cell.cell_type == 'code'
                  for output in cell.outputs if output.output_type == 'error'] if saved else []
        atomic_json(args.record, dict(
            scope='Clean-folder CPU smoke only. Model calls and quality metrics skipped; preinstalled interpreter used.',
            elapsed_seconds=time.perf_counter() - started, folder=str(folder),
            source_file_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            error=error, saved_errors=len(errors),
            execution=saved.metadata.get('overnight_execution') if saved else None))
        print(args.record.read_text())
    assert saved.metadata['overnight_execution']['completed'] and not errors
    assert all(cell.execution_count is not None for cell in saved.cells if cell.cell_type == 'code')


if __name__ == '__main__':
    main()
