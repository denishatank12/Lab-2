"""Resume a saved v5 evaluation, replacing ONLY its broken citation judge.

Original answering/evaluator files, CSV and receipts are not patched. Keeping
their original fingerprints lets their genuine answer checkpoints resume. The
new judge's source hash and recovery timing are recorded separately, not passed
off as the original code or a new within-budget full run.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_saved_files(root, receipt):
    root = Path(root).resolve()
    output = root / 'part1_rag/outputs/v2'
    if receipt.get('submission_sha256') != sha(output / 'submission.csv'):
        raise ValueError('Saved competition CSV changed. Existing files were not overwritten.')
    if receipt.get('questions_sha256') != sha(output / 'questions_used.csv'):
        raise ValueError('Saved questions changed. No answer regeneration was started.')
    files = receipt.get('sources')
    if not isinstance(files, dict) or not files:
        raise ValueError('Saved generation receipt has no source fingerprints.')
    checked = {}
    for name, expected in files.items():
        if name == 'rag_runtime_options':
            continue
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file() or sha(path) != expected:
            raise ValueError('Original source changed or is missing: ' + name)
        checked[name] = expected
    return checked


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def recover(root):
    root = Path(root).resolve()
    output = root / 'part1_rag/outputs/v2'
    receipt = json.loads((output / 'submission.provenance.json').read_text())
    original_files = validate_saved_files(root, receipt)
    options = receipt['sources'].get('rag_runtime_options', {})
    for key in ['LAB2_RAG_PROFILE', 'LAB2_RAG_NATURAL_JSON',
                'LAB2_RAG_EVIDENCE_REVIEW', 'LAB2_RAG_REVIEW_REQUIRED']:
        if options.get(key): os.environ[key] = options[key]
        else: os.environ.pop(key, None)
    os.environ.pop('LAB2_SMOKE_ONLY', None)
    sys.path[:0] = [str(root / 'tools'), str(root / 'part1_rag/src')]
    from run_checkpoint import atomic_json, runtime_identity, experiment_sources
    if receipt.get('sources') != experiment_sources('part1_rag'):
        raise ValueError('Original run fingerprints/options differ. Recovery will not regenerate old answers.')
    if receipt.get('runtime') != runtime_identity():
        raise ValueError('This is not the original runtime/GPU. Attach the saved results for an audited migration; '
                         'automatic recovery has not regenerated any answers.')
    # The patch is isolated from the original source tree/signatures. Explicit
    # provenance below identifies the replacement judge, including after resume.
    patch = Path(__file__).with_name('citation_judge.py')
    judge = load_module('_lab2_recovered_citation_judge', patch)
    evaluator = load_module('_lab2_original_saved_evaluator', root / 'part1_rag/src/evaluate_rag.py')
    evaluator.judge_citations = judge.judge_citations
    original_completion = json.loads((root / 'runtime_checks/part1_completion.json').read_text())
    record_path = root / 'runtime_checks/part1_evaluation_recovery.json'
    previous = json.loads(record_path.read_text()) if record_path.is_file() else {}
    record = dict(evaluation_completed=False, competition_answers_regenerated=False,
        original_csv_sha256=receipt['submission_sha256'], original_sources=original_files,
        citation_judge_protocol=judge.PROTOCOL, citation_judge_sha256=sha(patch),
        recovery_driver_sha256=sha(__file__), runtime=runtime_identity(),
        original_attempt_seconds=original_completion.get('seconds'),
        original_failed_notebook_preserved=True,
        limitations='Recovery is not a fresh error-free full run, a leaderboard score, human ratings or timing compliance.')
    evaluation = output / 'evaluation/evaluation.json'
    started = time.perf_counter()
    try:
        if (previous.get('evaluation_completed') is True and evaluation.is_file()
                and previous.get('evaluation_sha256') == sha(evaluation)
                and previous.get('citation_judge_sha256') == record['citation_judge_sha256']):
            print('Recovered evaluation already complete. No additional model calls.', flush=True)
        else:
            print('RESUMING GOLDEN EVALUATION ONLY. The 40 competition answers are not regenerated.', flush=True)
            print('Unfinished golden items still need GPU inference; saved golden answers resume unchanged.', flush=True)
            evaluator.evaluate(root / 'part1_rag/data', output / 'evaluation',
                               root / 'part1_rag/eval/golden_candidate.json')
        metrics = json.loads(evaluation.read_text())
        if {row['system'] for row in metrics['results']} != {'dense_baseline', 'hybrid_multimodal'}:
            raise ValueError('Both evaluation systems have not completed.')
        record.update(evaluation_completed=True, evaluation_sha256=sha(evaluation))
        print(json.dumps(metrics, indent=2), flush=True)
    except Exception as exc:
        record['error'] = str(exc)
        raise
    finally:
        record['this_recovery_seconds'] = time.perf_counter() - started
        record['recovery_seconds_total'] = previous.get('recovery_seconds_total', 0) + record['this_recovery_seconds']
        original_seconds = record['original_attempt_seconds']
        if isinstance(original_seconds, (int, float)):
            record['combined_attempt_seconds'] = original_seconds + record['recovery_seconds_total']
            record['within_90_minutes'] = record['combined_attempt_seconds'] <= 5400
        record['original_files_and_csv_unchanged'] = validate_saved_files(root, receipt) == original_files
        atomic_json(record_path, record)
        print('RECOVERY RECORD:', json.dumps(record, indent=2), flush=True)
        try:
            from part1_kaggle import export
            from kaggle_runtime import publish_zip
            publish_zip(export(root), working=root.parent, display_link=False)
        except Exception as exc:
            print('Export needs attention:', str(exc), 'Original files remain at', root, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    recover(parser.parse_args().root)
