"""Synthetic Kaggle portability tests, not pretrained-model GPU results."""
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
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from kaggle_runtime import configure_runtime, publish_zip


class ConfigurationTests(unittest.TestCase):
    def test_secret_is_not_returned_and_caches_stay_outside_saved_output(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            working = root / 'working'; working.mkdir()
            env = {'CUDA_VISIBLE_DEVICES': '0,1'}
            result = configure_runtime(working, root / 'temporary', env,
                                       secret_loader=lambda name: 'FICTIONAL_READ_SECRET')
            self.assertEqual(env['HF_TOKEN'], 'FICTIONAL_READ_SECRET')
            self.assertNotIn('FICTIONAL_READ_SECRET', json.dumps(result))
            self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '0')
            for key in ['HF_HOME', 'HF_HUB_CACHE', 'HF_DATASETS_CACHE', 'TORCH_HOME',
                        'UV_PYTHON_INSTALL_DIR', 'LAB2_ENV_DIR', 'LAB2_BOOTSTRAP_DIR']:
                self.assertNotIn(working, Path(env[key]).parents)
            self.assertEqual(env['LAB2_STANDALONE_PARENT'], str(working.resolve()))

    def test_missing_secret_fails_before_any_download_or_interactive_prompt(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            working = root / 'working'; working.mkdir()
            def missing(name):
                raise RuntimeError('FICTIONAL provider details should not leak')
            with self.assertRaisesRegex(RuntimeError, 'Add-ons > Secrets') as error:
                configure_runtime(working, root / 'not-created', {}, secret_loader=missing)
            self.assertNotIn('provider details', str(error.exception))
            self.assertFalse((root / 'not-created').exists())
            self.assertEqual(list(working.iterdir()), [])

    def test_existing_environment_secret_needs_no_provider_login(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            working = root / 'working'; working.mkdir()
            def unavailable(name):
                self.fail('Should reuse the existing environment secret')
            configure_runtime(working, root / 'scratch', {'HF_TOKEN': 'FIXTURE_ONLY'}, unavailable)

    def test_cache_inside_saved_output_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                configure_runtime(folder, Path(folder) / 'large-cache', {'HF_TOKEN': 'FIXTURE_ONLY'})

    def test_cpu_smoke_changes_no_host_paths_or_secrets(self):
        env = {}
        self.assertIn('CPU smoke', configure_runtime(environ=env, smoke_only=True)['mode'])
        self.assertEqual(env, {})

    def test_uv_bootstrap_respects_the_external_environment_location(self):
        import setup_runtime
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            bootstrap = root / 'temporary-bootstrap'
            with patch.dict(os.environ, {'LAB2_BOOTSTRAP_DIR': str(bootstrap)}), \
                 patch.object(setup_runtime.sys, 'version_info', (3, 13, 0)), \
                 patch.object(setup_runtime.subprocess, 'run') as run:
                setup_runtime.create_environment(root / 'venv')
            self.assertEqual(run.call_args_list[1].kwargs['env']['PYTHONPATH'], str(bootstrap / 'uv-0.5.31'))

    def test_installer_fixture_isolated_from_inherited_kaggle_paths(self):
        from test_part2_submission import TestRuntimeSafety
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            live = {'LAB2_ENV_DIR': str(root / 'real-environments'),
                    'LAB2_BOOTSTRAP_DIR': str(root / 'real-bootstrap')}
            with patch.dict(os.environ, live), redirect_stdout(io.StringIO()):
                result = unittest.TestResult()
                TestRuntimeSafety('test_setup_is_one_install_transaction_and_marks_only_success').run(result)
                self.assertTrue(result.wasSuccessful(), str(result.errors + result.failures))
                for name, value in live.items():
                    self.assertEqual(os.environ[name], value)
                    self.assertFalse(Path(value).exists())


class DeliveryTests(unittest.TestCase):
    def fixture(self, root):
        archive = root / 'run/output/results.zip'
        archive.parent.mkdir(parents=True)
        with zipfile.ZipFile(archive, 'w') as bundle:
            bundle.writestr('BUNDLE_MANIFEST.json', '{"fixture":true}')
        working = root / 'working'; working.mkdir()
        return archive, working

    def test_result_zip_is_validated_copied_and_safe_to_repeat(self):
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            archive, working = self.fixture(Path(folder))
            destination = publish_zip(archive, working, display_link=False)
            self.assertEqual(destination.read_bytes(), archive.read_bytes())
            self.assertEqual(publish_zip(archive, working, False), destination)

    def test_different_existing_output_is_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            archive, working = self.fixture(Path(folder))
            destination = working / archive.name
            destination.write_bytes(b'OLD RESULT FIXTURE')
            with self.assertRaises(FileExistsError):
                publish_zip(archive, working, False)
            self.assertEqual(destination.read_bytes(), b'OLD RESULT FIXTURE')

    def test_corrupt_zip_is_not_copied(self):
        with tempfile.TemporaryDirectory() as folder:
            archive, working = self.fixture(Path(folder))
            archive.write_bytes(b'CORRUPT FIXTURE')
            with self.assertRaises(zipfile.BadZipFile):
                publish_zip(archive, working, False)
            self.assertEqual(list(working.iterdir()), [])


class NotebookTests(unittest.TestCase):
    def test_packaged_kaggle_notebook_is_self_contained_and_noninteractive(self):
        import nbformat
        path = ROOT / 'part1_rag/notebooks/Lab2_Kaggle_Standalone.ipynb'
        if not path.exists():
            self.skipTest('Artifact test is external to the runtime source bundle.')
        n = nbformat.read(path, as_version=4)
        nbformat.validate(n)
        codes = [c.source for c in n.cells if c.cell_type == 'code']
        for source in codes:
            compile(source, 'kaggle-cell', 'exec')
        self.assertNotIn('getpass(', '\n'.join(codes))
        self.assertNotIn('from google.colab', '\n'.join(codes))
        self.assertIn('UserSecretsClient', codes[0])
        self.assertIn('publish_zip', codes[-1])
        self.assertIn('overnight_runner.py', '\n'.join(codes))
        self.assertIn('restore-part2', '\n'.join(codes))
        values = {node.targets[0].id: ast.literal_eval(node.value) for node in ast.parse(codes[0]).body
                  if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                  and node.targets[0].id in {'BUNDLE_BASE64', 'BUNDLE_SHA256'}}
        data = base64.b64decode(values['BUNDLE_BASE64'])
        self.assertEqual(hashlib.sha256(data).hexdigest(), values['BUNDLE_SHA256'])
        with zipfile.ZipFile(io.BytesIO(data)) as bundle:
            self.assertIn('tools/kaggle_runtime.py', bundle.namelist())
            runtime = bundle.read('tools/vlm_runtime.py').decode()
            self.assertIn('alphabet=tokenizer_data.tokenizer_alphabet', runtime)
            manifest = json.loads(bundle.read('STANDALONE_MANIFEST.json'))
            for name, expected in manifest['files'].items():
                self.assertEqual(hashlib.sha256(bundle.read(name)).hexdigest(), expected)


if __name__ == '__main__':
    unittest.main(verbosity=2)
