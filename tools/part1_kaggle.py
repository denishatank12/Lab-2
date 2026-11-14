"""Run the isolated Part 1 notebook and always preserve available evidence."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from run_checkpoint import atomic_json


def find_questions(data, environ=None, input_root=Path('/kaggle/input')):
    environ = os.environ if environ is None else environ
    explicit = environ.get('QUESTIONS_PATH')
    if explicit:
        result = Path(explicit).expanduser().resolve()
        if not result.is_file():
            raise FileNotFoundError('QUESTIONS_PATH does not exist. No substitute question file was used.')
        return result
    candidates = sorted(input_root.rglob('Questions.csv')) if input_root.is_dir() else []
    if len(candidates) > 1:
        raise ValueError('Multiple attached Questions.csv files found. Set QUESTIONS_PATH explicitly; no file was guessed.')
    return candidates[0].resolve() if candidates else (Path(data) / 'Questions.csv').resolve()


def export(root=ROOT):
    """Pure export, including a partially executed notebook after an error."""
    from export_lab_results import allowed_result, redact_common_secrets, TEXT_SUFFIXES
    root = Path(root)
    target = root / 'output/exports'
    target.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + '_' + str(time.time_ns())[-9:]
    path = target / f'Lab2_Part1_Results_{stamp}.zip'
    members = {}
    for folder in ['part1_rag/outputs', 'part1_rag/eval', 'part1_rag/src', 'output/executed', 'runtime_checks']:
        for item in (root / folder).rglob('*'):
            if allowed_result(item, item.relative_to(root)):
                members[item.relative_to(root).as_posix()] = item
    for item in (root / 'tools').glob('*.py'):
        members[item.relative_to(root).as_posix()] = item
    for name in ['STANDALONE_MANIFEST.json', 'requirements.txt']:
        if (root / name).is_file():
            members[name] = root / name
    required = sum(item.stat().st_size for item in members.values())
    if shutil.disk_usage(target).free < required + 64 * 2**20:
        raise RuntimeError('Not enough free space to package results. Existing files were preserved.')
    manifest = dict(scope='Part 1 only. No Part 2/3 completion is claimed.', files=[],
                    pending=['Two real groundedness ratings', 'Team review of candidate golden references',
                             'Final report, team records, tag and competition submission'])
    temporary = path.with_suffix('.partial.zip')
    try:
        with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED, compresslevel=4) as bundle:
            for name, item in sorted(members.items()):
                data = item.read_bytes()
                if item.suffix.lower() in TEXT_SUFFIXES:
                    data, _ = redact_common_secrets(data)
                bundle.writestr(name, data)
                manifest['files'].append(dict(path=name, sha256=hashlib.sha256(data).hexdigest()))
            bundle.writestr('BUNDLE_MANIFEST.json', json.dumps(manifest, indent=2))
        with zipfile.ZipFile(temporary) as bundle:
            if bundle.testzip():
                raise ValueError('Result ZIP failed its integrity check.')
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    atomic_json(target / 'latest_part1_results.json', dict(archive=str(path), files=len(members)))
    return path


def run(root=ROOT, smoke=False):
    from overnight_runner import execute_saved
    from kaggle_runtime import publish_zip
    root = Path(root)
    record = dict(computational_checks_passed=False, smoke_only=smoke, submission_ready=False)
    started = time.perf_counter()
    if smoke:
        os.environ['LAB2_SMOKE_ONLY'] = '1'
    else:
        os.environ.pop('LAB2_SMOKE_ONLY', None)
    try:
        saved = execute_saved(root / 'part1_rag/notebooks/Part1_Kaggle_Experiment.ipynb',
            root / 'output/executed/part1_rag/Part1_Kaggle_Experiment.ipynb', working_dir=root)
        record['executed_notebook'] = str(saved)
        if not smoke:
            output = root / 'part1_rag/outputs/v2'
            with (output / 'submission.csv').open(newline='', encoding='utf-8') as handle:
                rows = list(csv.DictReader(handle))
            metrics = json.loads((output / 'evaluation/evaluation.json').read_text())
            record['checks'] = dict(nonempty_submission=bool(rows),
                both_systems_measured={m['system'] for m in metrics['results']} == {'dense_baseline', 'hybrid_multimodal'},
                no_saved_errors=not any(o.get('output_type') == 'error' for c in json.loads(saved.read_text())['cells']
                    if c['cell_type'] == 'code' for o in c.get('outputs', [])))
            record['computational_checks_passed'] = all(record['checks'].values())
            if not record['computational_checks_passed']:
                raise RuntimeError('Part 1 computation checks failed. Partial evidence was retained.')
            destination = root.parent / 'submission_part1_v3.csv'
            if destination.exists() and destination.read_bytes() != (output / 'submission.csv').read_bytes():
                destination = root.parent / ('submission_part1_v3_' + root.name.removeprefix('Lab2_Standalone_') + '.csv')
            if destination.exists() and destination.read_bytes() != (output / 'submission.csv').read_bytes():
                raise FileExistsError('A different submission exists at the destination and was preserved.')
            shutil.copy2(output / 'submission.csv', destination)
            record['competition_csv'] = str(destination)
            print('VALIDATED KAGGLE CSV:', destination, flush=True)
        print('Part 1 finished. Human review and the external competition submission remain separate.', flush=True)
    except Exception as exc:
        record['error'] = str(exc)
        raise
    finally:
        record['seconds'] = time.perf_counter() - started
        atomic_json(root / 'runtime_checks/part1_completion.json', record)
        # Never hide the original failure if export also encounters a disk error.
        try:
            archive = export(root)
            publish_zip(archive, working=root.parent, display_link=False)
        except Exception as exc:
            print('Export needs attention:', str(exc), 'Original results remain in', root, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['run', 'export'])
    parser.add_argument('--smoke-only', action='store_true')
    args = parser.parse_args()
    if args.action == 'run':
        run(smoke=args.smoke_only)
    else:
        from kaggle_runtime import publish_zip
        publish_zip(export(), working=ROOT.parent, display_link=False)
