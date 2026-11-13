"""Offline final-notebook packaging and export checks, never leaderboard scores."""
import ast
import base64
from contextlib import redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from kaggle_runtime import configure_runtime
from part1_kaggle import find_questions, export


class Part1RuntimeTests(unittest.TestCase):
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
        notebook = nbformat.read(ROOT / 'part1_rag/notebooks/Lab2_Part1_Kaggle_Final.ipynb', 4)
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
            self.assertEqual(len([name for name in bundle.namelist() if name.endswith('.pdf')]), 3)
            child = json.loads(bundle.read('part1_rag/notebooks/Part1_Kaggle_Experiment.ipynb'))
            sources = [''.join(c['source']) if isinstance(c['source'], list) else c['source'] for c in child['cells']]
            for source in sources:
                if 'from pathlib' in source or 'run_script(' in source or 'if SMOKE_ONLY:' in source:
                    compile(source, 'child-cell', 'exec')
            self.assertIn('find_questions(DATA)', '\n'.join(sources))
            self.assertIn('validate_submission.py', '\n'.join(sources))


if __name__ == '__main__':
    unittest.main(verbosity=2)
