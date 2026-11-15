"""Offline final-notebook packaging and export checks, never leaderboard scores."""
import ast
import base64
from contextlib import redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from kaggle_runtime import configure_runtime
from part1_kaggle import find_questions, export, publish_submission, run
from part1_diagnostics import failure_details
from run_checkpoint import experiment_sources, runtime_identity, file_hash


class Part1RuntimeTests(unittest.TestCase):
    def fixture_submission(self, root):
        output = root / 'part1_rag/outputs/v2'
        output.mkdir(parents=True)
        (output / 'submission.csv').write_text('id,conversation_id,question_id,answer,figure_references,citations,status\n'
            '1,C01,Q01,Fictional test answer,NONE,DOC1:p1,answered\n')
        (output / 'questions_used.csv').write_text('id,conversation_id,question_id,question\n1,C01,Q01,Fictional test question\n')
        (output / 'run_inputs.json').write_text(json.dumps(dict(smoke_only=False,
            questions_sha256=file_hash(output / 'questions_used.csv'))))
        (output / 'submission.provenance.json').write_text(json.dumps(dict(
            submission_sha256=file_hash(output / 'submission.csv'),
            questions_sha256=file_hash(output / 'questions_used.csv'),
            sources=experiment_sources('part1_rag'), runtime=runtime_identity(), rows=1)))
        validator = root / 'part1_rag/data/validate_submission.py'
        validator.parent.mkdir(parents=True)
        shutil.copy2(ROOT / 'part1_rag/data/validate_submission.py', validator)
        return output

    def test_completed_csv_is_published_without_waiting_for_evaluation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / 'working/Lab2_Standalone_fixture'
            output = self.fixture_submission(root)
            path = publish_submission(root)
            self.assertEqual(path.name, 'submission_part1_v6.csv')
            self.assertEqual(path.read_bytes(), (output / 'submission.csv').read_bytes())
            self.assertTrue((output / 'published_submission.json').is_file())

    def test_stale_or_changed_predictions_are_not_published(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / 'working/Lab2_Standalone_fixture'
            output = self.fixture_submission(root)
            with (output / 'submission.csv').open('a') as stream:
                stream.write('2,C02,Q02,Changed,NONE,NONE,not_found\n')
            with self.assertRaisesRegex(ValueError, 'hashes'):
                publish_submission(root)
            self.assertFalse((root.parent / 'submission_part1_v6.csv').exists())

    def test_late_evaluation_failure_retains_real_csv_but_is_not_called_complete(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / 'working/Lab2_Standalone_fixture'
            self.fixture_submission(root)
            with patch('overnight_runner.execute_saved', side_effect=RuntimeError('Fictional evaluation failure')), \
                 patch('part1_kaggle.export', return_value=root / 'fixture.zip'), \
                 patch('kaggle_runtime.publish_zip'), redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, 'Fictional evaluation failure'):
                    run(root)
            record = json.loads((root / 'runtime_checks/part1_completion.json').read_text())
            self.assertTrue(record['csv_generation_complete'])
            self.assertFalse(record['computational_checks_passed'])
            self.assertFalse(record['submission_ready'])
            self.assertTrue(Path(record['competition_csv']).is_file())

    def test_failure_diagnostics_reveal_inner_error_and_redact_credentials(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            saved = root / 'output/executed/part1_rag/Part1_Kaggle_Experiment.ipynb'
            saved.parent.mkdir(parents=True)
            saved.write_text(json.dumps(dict(cells=[dict(outputs=[dict(output_type='error',
                ename='ValueError', evalue='Fictional inner cause', traceback=['Fictional traceback'])])])) )
            log = root / 'runtime_checks/generate_submission.log'
            log.parent.mkdir()
            token = 'hf_' + 'z' * 30
            log.write_text('Fictional source log: ' + token)
            detail = failure_details(root)
            self.assertIn('Fictional inner cause', detail)
            self.assertIn('TOKEN REDACTED', detail)
            self.assertNotIn(token, detail)

    def test_public_part1_models_require_no_secret(self):
        with tempfile.TemporaryDirectory() as folder:
            working = Path(folder) / 'working'
            working.mkdir()
            def forbidden(name):
                self.fail('Part 1 must never request a private Part 2 token')
            env = {}
            value = configure_runtime(working, Path(folder) / 'scratch', environ=env,
                                      secret_loader=forbidden, require_private_artifacts=False)
            self.assertFalse(value['secret_loaded'])
            self.assertNotIn('HF_TOKEN', env)

    def test_attached_hidden_question_file_is_selected_unchanged(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            attached = root / 'input/competition/Questions.csv'
            attached.parent.mkdir(parents=True)
            content = b'id,conversation_id,question_id,question\n987,C99,Q88,Fictional hidden query\n'
            attached.write_bytes(content)
            self.assertEqual(find_questions(root / 'data', {}, root / 'input'), attached.resolve())
            self.assertEqual(attached.read_bytes(), content)

    def test_ambiguous_and_missing_explicit_question_files_fail_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ['a', 'b']:
                path = root / name / 'Questions.csv'
                path.parent.mkdir()
                path.write_text('FIXTURE')
            with self.assertRaises(ValueError):
                find_questions(root, {}, root)
            with self.assertRaises(FileNotFoundError):
                find_questions(root, {'QUESTIONS_PATH': str(root / 'missing.csv')}, root)

    def test_export_keeps_partial_notebook_and_redacts_credentials(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'output/executed/part1_rag/Part1_Kaggle_Experiment.ipynb'
            source.parent.mkdir(parents=True)
            source.write_text(json.dumps({'cells': [{'outputs': [{'output_type': 'error'}]}]}))
            log = root / 'runtime_checks/failure.log'
            log.parent.mkdir()
            log.write_text('Fictional token: hf_' + 'z' * 30)
            bundle = export(root)
            with zipfile.ZipFile(bundle) as archive:
                self.assertIn(source.relative_to(root).as_posix(), archive.namelist())
                self.assertIn('REDACTED_CREDENTIAL', archive.read('runtime_checks/failure.log').decode())
                self.assertIn('Part 1 only', json.loads(archive.read('BUNDLE_MANIFEST.json'))['scope'])
                self.assertIsNone(archive.testzip())

    def test_final_notebook_embeds_all_sources_and_course_inputs(self):
        import nbformat
        path = ROOT / 'part1_rag/notebooks/Lab2_Part1_Kaggle_Final.ipynb'
        notebook = nbformat.read(path, 4)
        nbformat.validate(notebook)
        codes = [c.source for c in notebook.cells if c.cell_type == 'code']
        for source in codes:
            compile(source, 'part1-final-cell', 'exec')
        self.assertIn('require_private_artifacts=False', codes[0])
        values = {node.targets[0].id: ast.literal_eval(node.value) for node in ast.parse(codes[0]).body
                  if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                  and node.targets[0].id in {'BUNDLE_SHA256', 'BUNDLE_BASE64'}}
        raw = base64.b64decode(values['BUNDLE_BASE64'])
        self.assertEqual(hashlib.sha256(raw).hexdigest(), values['BUNDLE_SHA256'])
        with zipfile.ZipFile(io.BytesIO(raw)) as bundle:
            manifest = json.loads(bundle.read('STANDALONE_MANIFEST.json'))
            for name, expected in manifest['files'].items():
                self.assertEqual(hashlib.sha256(bundle.read(name)).hexdigest(), expected)
            self.assertFalse(any(name.startswith('part2_diffusion/') for name in bundle.namelist()))
            self.assertFalse(any(name.endswith('.pdf') for name in bundle.namelist()))
            self.assertEqual(len([r for r in manifest['resources'] if r['path'].endswith('.pdf')]), 3)
            self.assertLess(path.stat().st_size, 950_000)
            child = json.loads(bundle.read('part1_rag/notebooks/Part1_Kaggle_Experiment.ipynb'))
            sources = [''.join(c['source']) if isinstance(c['source'], list) else c['source'] for c in child['cells']]
            for source in sources:
                if 'from pathlib' in source or 'run_script(' in source or 'if SMOKE_ONLY:' in source:
                    compile(source, 'child-cell', 'exec')
            self.assertIn('find_questions(DATA)', '\n'.join(sources))
            self.assertIn('validate_submission.py', '\n'.join(sources))
            self.assertIn('tools/test_evidence_review.py', '\n'.join(sources))
            self.assertIn('LAB2_RAG_EVIDENCE_REVIEW', '\n'.join(sources))
            self.assertIn('part1_rag/src/evidence_review.py', bundle.namelist())
            self.assertIn('submission_part1_v6.csv', bundle.read('tools/part1_kaggle.py').decode())
            self.assertIn('retry_schema=allowed_schema', bundle.read('part1_rag/src/multimodal.py').decode())
            self.assertIn('tools/part1_diagnostics.py', bundle.namelist())
            self.assertIn('tools/citation_judge.py', bundle.namelist())
            self.assertIn('READ-ONLY FAILURE DIAGNOSTICS', '\n'.join(codes))
            self.assertIn('publish_submission(ROOT)', '\n'.join(sources))

    def test_fresh_notebook_has_no_saved_session_or_recovery_dependency(self):
        import nbformat
        notebook = nbformat.read(ROOT / 'part1_rag/notebooks/Lab2_Part1_Kaggle_Final.ipynb', 4)
        codes = [c.source for c in notebook.cells if c.cell_type == 'code']
        self.assertNotIn('LAB2_PART1_SAVED_ROOT', '\n'.join(codes))
        self.assertNotIn('submission.provenance.json").is_file()', codes[0])
        self.assertIn('ROOT = unpack(', codes[0])
        self.assertIn('NOTEBOOK_STARTED = time.perf_counter()', codes[0])
        self.assertIn('os.environ["LAB2_RAG_EVIDENCE_REVIEW"] = "0"', codes[0])
        self.assertIn('notebook_wall_time.json', '\n'.join(codes))
        values = {node.targets[0].id:ast.literal_eval(node.value) for node in ast.parse(codes[0]).body
            if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name)
            and node.targets[0].id == 'BUNDLE_BASE64'}
        with zipfile.ZipFile(io.BytesIO(base64.b64decode(values['BUNDLE_BASE64']))) as bundle:
            self.assertFalse(any('recover_part1' in name for name in bundle.namelist()))
            child = json.loads(bundle.read('part1_rag/notebooks/Part1_Kaggle_Experiment.ipynb'))
            text = '\n'.join(''.join(c['source']) for c in child['cells'])
            self.assertIn('evidence_review=False', text)
            self.assertIn('REAL GPU CITATION-JUDGE PREFLIGHT PASSED', text)
            self.assertIn('tools/test_part1_pipeline.py', text)

    def test_missing_evaluation_display_reports_pending_instead_of_file_error(self):
        import nbformat
        notebook = nbformat.read(ROOT / 'part1_rag/notebooks/Lab2_Part1_Kaggle_Final.ipynb', 4)
        cell = next(c for c in notebook.cells if c.cell_type == 'code' and 'columns = [' in c.source)
        with tempfile.TemporaryDirectory() as folder:
            printed = io.StringIO()
            with redirect_stdout(printed):
                exec(cell.source, {'ROOT':Path(folder), 'SMOKE_ONLY':False})
            self.assertIn('Evaluation is incomplete', printed.getvalue())


if __name__ == '__main__':
    unittest.main(verbosity=2)
