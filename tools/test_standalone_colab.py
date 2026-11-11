"""Clean-folder packaging and synthetic recovery tests, not model-quality evidence."""
from contextlib import redirect_stdout
import ast
import base64
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
import subprocess
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import standalone_colab as standalone


def archive_bytes(files, resources=None):
    manifest = dict(source_commit='FIXTURE_BASE_COMMIT', source_dirty_at_build=True,
        files={name: hashlib.sha256(value).hexdigest() for name, value in files.items()}, resources=resources or [])
    out = io.BytesIO()
    with zipfile.ZipFile(out, 'w') as archive:
        for name, value in files.items():
            archive.writestr(name, value)
        archive.writestr('STANDALONE_MANIFEST.json', json.dumps(manifest))
    data = out.getvalue()
    return data, hashlib.sha256(data).hexdigest()


class BundleTests(unittest.TestCase):
    def test_empty_folder_needs_no_repo_or_helper_files(self):
        data, digest = archive_bytes({'tools/fixture.py': b'# Synthetic code fixture\n'})
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            self.assertEqual(list(parent.iterdir()), [])
            root = standalone.unpack(data, digest, parent)
            self.assertEqual((root / 'tools/fixture.py').read_bytes(), b'# Synthetic code fixture\n')
            self.assertFalse((root / '.git').exists())
            self.assertEqual(standalone.unpack(data, digest, parent), root)

    def test_corrupted_payload_is_rejected_before_writes(self):
        data, digest = archive_bytes({'tools/fixture.py': b'# Fixture'})
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(ValueError):
                standalone.unpack(data + b'CHANGED', digest, Path(temporary))
            self.assertEqual(list(Path(temporary).iterdir()), [])

    def test_traversal_and_symlink_targets_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            data, digest = archive_bytes({'../escape.py': b'# Fixture'})
            with self.assertRaises(ValueError):
                standalone.unpack(data, digest, Path(temporary))
            self.assertFalse((Path(temporary).parent / 'escape.py').exists())

    def test_local_edits_are_not_overwritten(self):
        data, digest = archive_bytes({'tools/fixture.py': b'# Fixture'})
        with tempfile.TemporaryDirectory() as temporary:
            root = standalone.unpack(data, digest, Path(temporary))
            path = root / 'tools/fixture.py'; path.write_text('# LOCAL EDIT')
            with self.assertRaises(ValueError):
                standalone.unpack(data, digest, Path(temporary))
            self.assertEqual(path.read_text(), '# LOCAL EDIT')

    def test_actual_notebook_is_self_contained_and_has_no_clone_setup(self):
        import nbformat
        n = nbformat.read(ROOT / 'part1_rag/notebooks/Lab2_Colab_Standalone.ipynb', as_version=4)
        nbformat.validate(n)
        codes = [c.source for c in n.cells if c.cell_type == 'code']
        first = codes[0]
        self.assertNotIn('git", "clone', first)
        self.assertNotIn('git", "pull', first)
        self.assertNotIn('"4090" not in', first)
        tree = ast.parse(first)
        values = {node.targets[0].id: ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name) and node.targets[0].id in ['BUNDLE_BASE64', 'BUNDLE_SHA256']}
        data = base64.b64decode(values['BUNDLE_BASE64'])
        self.assertEqual(hashlib.sha256(data).hexdigest(), values['BUNDLE_SHA256'])
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            names = archive.namelist()
            for name in ['tools/standalone_colab.py', 'tools/setup_runtime.py', 'tools/response_schemas.py', 'tools/kernel_preflight.py',
                         'part1_rag/src/generate_submission.py', 'part3_agents/src/model_runtime.py']:
                self.assertIn(name, names)
            runner=archive.read('tools/overnight_runner.py').decode()
            self.assertIn('class LabKernelApp(IPKernelApp):',runner)
            self.assertIn('def load_config_file(self, *args, **kwargs):',runner)
            self.assertIn("[sys.executable,'-I','-c',KERNEL_BOOTSTRAP",runner)
            self.assertFalse(any(Path(name).suffix in ['.safetensors', '.pt', '.pdf', '.png'] for name in names))
            manifest = json.loads(archive.read('STANDALONE_MANIFEST.json'))
            self.assertTrue(all(not r['path'].endswith('.py') for r in manifest['resources']))
            for part, name in [('part1_rag', 'Part1_RAG_Run.ipynb'), ('part3_agents', 'Part3_Agents_Run.ipynb')]:
                child = json.loads(archive.read(f'{part}/notebooks/{name}'))
                setup = next(c for c in child['cells'] if c['cell_type'] == 'code')['source']
                setup = setup if isinstance(setup, str) else ''.join(setup)
                self.assertNotIn('git", "clone', setup)
                self.assertIn('LAB2_STANDALONE_ROOT', setup)
        for source in codes:
            compile(source, 'standalone-cell', 'exec')

    def test_recovery_cell_runs_before_tests_without_removing_saved_files(self):
        import nbformat
        n = nbformat.read(ROOT / 'part1_rag/notebooks/Lab2_Colab_Standalone.ipynb', as_version=4)
        codes = [c.source for c in n.cells if c.cell_type == 'code']
        recovery = next(i for i, source in enumerate(codes) if 'run_script("tools/kernel_preflight.py")' in source)
        checks = next(i for i, source in enumerate(codes) if 'for name in ["test_lab_completion.py"' in source)
        self.assertLess(recovery, checks)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            saved = root / 'saved_result.json'
            saved.write_text('{"preserved": true}')
            calls=[]
            namespace = {'ROOT': root, 'run_script': lambda *args: calls.append(args)}
            with redirect_stdout(io.StringIO()):
                exec(compile(codes[recovery], 'recovery-cell', 'exec'), namespace)
                exec(compile(codes[recovery], 'recovery-cell', 'exec'), namespace)
            self.assertEqual(calls,[('tools/kernel_preflight.py',)]*2)
            self.assertNotIn('os.environ',codes[recovery])
            self.assertEqual(saved.read_text(), '{"preserved": true}')


class ResourceTests(unittest.TestCase):
    def test_data_download_is_verified_and_resumed(self):
        data = b'FAKE COURSE INPUT FOR UNIT TEST ONLY'
        row = dict(path='part1_rag/data/fixture.csv', sha256=hashlib.sha256(data).hexdigest(), url='https://example.test/fixture.csv')
        calls = []
        def opener(*args, **kwargs):
            calls.append(1); return io.BytesIO(data)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'STANDALONE_MANIFEST.json').write_text(json.dumps(dict(resources=[row])))
            with redirect_stdout(io.StringIO()):
                standalone.download_resources(root, opener=opener)
                standalone.download_resources(root, opener=opener)
            self.assertEqual(len(calls), 1)
            self.assertEqual((root / row['path']).read_bytes(), data)

    def test_wrong_download_does_not_become_a_ready_input(self):
        row = dict(path='part1_rag/data/fixture.csv', sha256='0' * 64, url='https://example.test/fixture.csv')
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'STANDALONE_MANIFEST.json').write_text(json.dumps(dict(resources=[row])))
            with self.assertRaises(RuntimeError):
                standalone.download_resources(root, opener=lambda *a, **k: io.BytesIO(b'BAD FIXTURE'), sleeper=lambda _: None)
            self.assertFalse((root / row['path']).exists())
            self.assertFalse(list(root.rglob('*.downloading')))

    def test_runtime_does_not_download_python_code(self):
        row = dict(path='tools/forbidden.py', sha256='0' * 64, url='https://example.test/code.py')
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'STANDALONE_MANIFEST.json').write_text(json.dumps(dict(resources=[row])))
            with self.assertRaisesRegex(ValueError, 'embedded'):
                standalone.download_resources(root)


class InstallerTests(unittest.TestCase):
    def test_newer_kernel_bootstraps_python312_without_changing_live_packages(self):
        import setup_runtime
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / 'new-runtime'; target.mkdir()
            with patch.object(setup_runtime, 'ROOT', root), patch.object(setup_runtime.sys, 'version_info', (3, 13, 0)), \
                 patch.object(setup_runtime.subprocess, 'run') as run, patch.object(setup_runtime.venv, 'EnvBuilder') as builder:
                created = setup_runtime.create_environment(target)
            builder.assert_not_called()
            self.assertEqual(created.name, 'new-runtime-uv')
            install, create = run.call_args_list
            self.assertIn('--target', install.args[0])
            self.assertIn('--python', create.args[0])
            self.assertIn('3.12', create.args[0])
            self.assertEqual(create.kwargs['env']['PYTHONPATH'], str(root / '.bootstrap-tools/uv-0.5.31'))

    def test_missing_ensurepip_keeps_the_failed_folder_and_uses_a_new_one(self):
        import setup_runtime
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / 'partial'; target.mkdir()
            (target / 'preserve.txt').write_text('OLD INSTALL FIXTURE')
            with patch.object(setup_runtime, 'ROOT', root), patch.object(setup_runtime.sys, 'version_info', (3, 12, 0)), \
                 patch.object(setup_runtime.venv.EnvBuilder, 'create', side_effect=subprocess.CalledProcessError(1, ['ensurepip'])), \
                 patch.object(setup_runtime.subprocess, 'run'), redirect_stdout(io.StringIO()):
                created = setup_runtime.create_environment(target)
            self.assertNotEqual(created, target)
            self.assertEqual((target / 'preserve.txt').read_text(), 'OLD INSTALL FIXTURE')


class RecoveryTests(unittest.TestCase):
    def fixture(self, parent):
        from PIL import Image
        from test_rtx4090_workflow import part2_fixture
        source = parent / 'source'; source.mkdir()
        p2 = part2_fixture(source)
        train = p2 / 'data/processed/train'; train.mkdir(parents=True)
        Image.new('RGB', (512, 512), 'blue').save(train / 'fixture_training.png')
        rows = [dict(model=run, generated_file=f'{run}_{i:02d}_0.png',
            training_file='fixture_training.png', clip_cosine=.5) for run in ['run1', 'run2'] for i in range(10)]
        (p2 / 'outputs/memorization.json').write_text(json.dumps(rows))
        links = json.loads((p2 / 'evidence/hub_links.json').read_text())
        links.update(run2='https://huggingface.co/FIXTURE_ONLY/run2', run2_commit='FIXTURE_REVISION_2',
            images='https://huggingface.co/datasets/FIXTURE_ONLY/images', images_commit='FIXTURE_IMAGES_REVISION')
        root = parent / 'fresh'; (root / 'part2_diffusion/evidence').mkdir(parents=True)
        shutil.copy2(p2 / 'config.yaml', root / 'part2_diffusion/config.yaml')
        (root / 'part2_diffusion/evidence/hub_links.json').write_text(json.dumps(links))
        original = root / 'standalone_evidence/original_part2'
        record = dict(original_archive_sha256='FIXTURE_ONLY', saved_files={}, hub_files={})
        for name in ['outputs/evaluation.json', 'outputs/memorization.json', 'outputs/generations/manifest.json']:
            destination = original / name; destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p2 / name, destination)
            record['saved_files'][name] = standalone.sha256(destination)
        for run in ['run1', 'run2']:
            record['hub_files'][run] = {name: standalone.sha256(p2 / 'outputs' / run / name)
                for name in ['pytorch_lora_weights.safetensors', 'training_metrics.json', 'run_config.json']}
        record['hub_files']['images'] = {path.name: standalone.sha256(path)
            for path in (p2 / 'outputs/generations').glob('*.png') if not path.name.startswith('comparison')}
        (original / 'RESTORE_MANIFEST.json').write_text(json.dumps(record))
        target = root / 'part2_diffusion/data/processed/train'; target.mkdir(parents=True)
        shutil.copy2(train / 'fixture_training.png', target / 'fixture_training.png')
        def snapshot(repo_id, **kwargs):
            return str(p2 / 'outputs/generations') if kwargs.get('repo_type') == 'dataset' else str(p2 / 'outputs' / repo_id.rsplit('/', 1)[1])
        return root, p2, snapshot

    def test_complete_synthetic_recovery_needs_no_uploaded_files_or_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, p2, snapshot = self.fixture(Path(temporary))
            with redirect_stdout(io.StringIO()):
                result = standalone.restore_original_part2(root, snapshot=snapshot, token='UNIT_TEST_NOT_A_REAL_TOKEN')
            self.assertEqual(result['generated_images'], 120)
            self.assertFalse(result['trained_on_current_gpu'])
            self.assertFalse(result['metrics_recomputed'])
            self.assertFalse(result['training_repeated'])
            self.assertTrue((root / 'part2_diffusion/outputs/generations/comparison_07.png').exists())
            self.assertTrue((root / 'part2_diffusion/outputs/memorization_run2_2.png').exists())
            self.assertTrue((root / 'part3_agents/outputs/hub_adapter/hub_provenance.json').exists())
            self.assertEqual((root / 'part2_diffusion/outputs/run1/pytorch_lora_weights.safetensors').read_bytes(),
                             (p2 / 'outputs/run1/pytorch_lora_weights.safetensors').read_bytes())

    def test_tampered_hub_image_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, p2, snapshot = self.fixture(Path(temporary))
            (p2 / 'outputs/generations/base_00_0.png').write_bytes(b'TAMPERED FIXTURE')
            with redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, 'Hash mismatch'):
                standalone.restore_original_part2(root, snapshot=snapshot, token='UNIT_TEST_NOT_A_REAL_TOKEN')
            self.assertFalse((root / 'runtime_checks/standalone_part2_restoration.json').exists())

    def test_changed_local_output_is_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, _, snapshot = self.fixture(Path(temporary))
            path = root / 'part2_diffusion/outputs/run1/pytorch_lora_weights.safetensors'
            path.parent.mkdir(parents=True); path.write_bytes(b'USER_OWNED_FIXTURE')
            with redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, 'preserved'):
                standalone.restore_original_part2(root, snapshot=snapshot, token='UNIT_TEST_NOT_A_REAL_TOKEN')
            self.assertEqual(path.read_bytes(), b'USER_OWNED_FIXTURE')


if __name__ == '__main__':
    unittest.main(verbosity=2)
