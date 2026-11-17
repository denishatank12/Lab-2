"""Offline checks for attribution, baseline-only routing and standalone packaging."""
import ast
import base64
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import nbformat as nb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from build_partner_baseline_notebook import payload


class PartnerBaselineTests(unittest.TestCase):
    def notebook_helper(self, name):
        notebook = nb.read(ROOT / "part3_agents/notebooks/Part3_Partner_Baseline_Standalone.ipynb", as_version=4)
        tree = ast.parse(notebook.cells[1].source)
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        namespace = {}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "notebook-helper", "exec"), namespace)
        return namespace[name]

    def authentication_helper(self):
        return self.notebook_helper("configure_hf_token")

    def test_environment_creation_never_calls_ensurepip(self):
        import venv
        prepare = self.notebook_helper("prepare_python_environment")
        with tempfile.TemporaryDirectory() as temporary:
            env = Path(temporary) / "python_env"
            with patch.object(venv.EnvBuilder, "_setup_pip", side_effect=AssertionError("ensurepip must never run")):
                python = prepare(env)
                self.assertEqual(prepare(env), python)
            result = subprocess.run([str(python), '-I', '-c', 'import json,sys,importlib.util; '
                'print(json.dumps(dict(prefix=sys.prefix, version=list(sys.version_info[:2]), '
                'torch_visible=importlib.util.find_spec("torch") is not None, paths=sys.path)))'],
                check=True, text=True, capture_output=True, timeout=90)
            record = json.loads(result.stdout)
            self.assertEqual(Path(record['prefix']).resolve(), env.resolve())
            self.assertEqual(record['version'], list(sys.version_info[:2]))
            self.assertFalse(record['torch_visible'])
            self.assertFalse(any(str(ROOT / '.venv') in p for p in record['paths']))

    def test_environment_recovers_partial_folder_without_deleting_files(self):
        import venv
        prepare = self.notebook_helper("prepare_python_environment")
        with tempfile.TemporaryDirectory() as temporary:
            env = Path(temporary) / "partial_env"
            env.mkdir()
            sentinel = env / "preserve_this_file.txt"
            sentinel.write_text("fictional previous run marker")
            with patch.object(venv.EnvBuilder, "_setup_pip", side_effect=AssertionError("broken ensurepip")):
                prepare(env)
            self.assertTrue(sentinel.exists())
            self.assertIn('include-system-site-packages = false', (env / 'pyvenv.cfg').read_text().lower())

    def test_old_host_inheriting_environment_is_preserved_not_reused(self):
        import venv
        prepare = self.notebook_helper("prepare_python_environment")
        with tempfile.TemporaryDirectory() as temporary:
            env = Path(temporary) / 'old_environment'
            venv.EnvBuilder(with_pip=False, system_site_packages=True).create(env)
            before = (env / 'pyvenv.cfg').read_bytes()
            with self.assertRaisesRegex(RuntimeError, 'preserved'):
                prepare(env)
            self.assertEqual((env / 'pyvenv.cfg').read_bytes(), before)

    def test_gpu_dependency_profile_has_matching_pair_and_exact_pins(self):
        profile = self.notebook_helper('dependency_profile')(False)
        self.assertTrue(profile['cuda'])
        self.assertIn('torch==2.11.0+cu128', profile['pins'])
        self.assertIn('torchvision==0.26.0+cu128', profile['pins'])
        self.assertIn('bitsandbytes==0.50.2', profile['pins'])
        # tokenizers 0.23.1 requires hub<2; transformers/diffusers need >=1.32.
        self.assertIn('huggingface-hub==1.32.0', profile['pins'])
        self.assertTrue(all('==' in item for item in profile['pins']))
        self.assertFalse(any(item.split('==')[0] in ['google-colab', 'nbconvert', 'cudf-cu13', 'torchao'] for item in profile['pins']))

    def test_public_adapter_access_needs_no_secret_and_private_failure_is_clear(self):
        from urllib.error import HTTPError
        check = self.notebook_helper('check_adapter_access')
        info = {'sha': 'fictional-immutable-sha', 'private': False,
                'siblings': [{'rfilename': 'pytorch_lora_weights.safetensors'}]}
        def public(request, timeout):
            self.assertNotIn('Authorization', request.headers)
            return io.BytesIO(json.dumps(info).encode())
        result = check('Shibin22/sd15-naruto-run1-lora-r4', opener=public)
        self.assertFalse(result['authenticated'])
        for status in [401, 403, 404]:
            def denied(request, timeout):
                raise HTTPError(request.full_url, status, 'fictional access failure', {}, None)
            with self.assertRaisesRegex(RuntimeError, 'exact adapter ZIP'):
                check('Shibin22/sd15-naruto-run1-lora-r4', opener=denied)

    def test_public_mode_ignores_a_stale_token(self):
        with tempfile.TemporaryDirectory() as temporary:
            module = self.unpack(Path(temporary))
            with patch.dict(os.environ, {'LAB2_P3_USE_HF_TOKEN': '0', 'HF_TOKEN': 'fictional-stale-token'}):
                self.assertIs(module.authenticate(), False)

    def test_host_pip_installs_offline_wheel_into_pip_free_target(self):
        prepare = self.notebook_helper("prepare_python_environment")
        with tempfile.TemporaryDirectory() as temporary:
            env = Path(temporary) / "target"
            python = prepare(env)
            wheel = Path(temporary) / "lab2_env_probe-1.0-py3-none-any.whl"
            with zipfile.ZipFile(wheel, 'w') as bundle:
                files = {
                    'lab2_env_probe/__init__.py': 'VALUE = "fictional bootstrap check"\n',
                    'lab2_env_probe-1.0.dist-info/METADATA': 'Metadata-Version: 2.1\nName: lab2-env-probe\nVersion: 1.0\n',
                    'lab2_env_probe-1.0.dist-info/WHEEL': 'Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n'}
                files['lab2_env_probe-1.0.dist-info/RECORD'] = ''.join(name + ',,\n' for name in files) + 'lab2_env_probe-1.0.dist-info/RECORD,,\n'
                for name, contents in files.items():
                    bundle.writestr(name, contents)
            with patch.object(__import__('venv').EnvBuilder, '_setup_pip', side_effect=AssertionError('ensurepip forbidden')):
                result = subprocess.run([sys.executable, '-m', 'pip', '--python', str(python), 'install',
                    '--disable-pip-version-check', '--no-index', '--no-deps', str(wheel)],
                    text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=90)
            self.assertEqual(result.returncode, 0, result.stdout)
            result = subprocess.run([str(python), '-c', 'import lab2_env_probe; '
                'print(lab2_env_probe.VALUE); print(lab2_env_probe.__file__)'],
                check=True, capture_output=True, text=True, timeout=30)
            self.assertIn('fictional bootstrap check', result.stdout)
            self.assertIn(str(env), result.stdout)
            self.assertIsNone(importlib.util.find_spec('lab2_env_probe'))

    def test_packaged_setup_uses_host_pip_not_target_ensurepip(self):
        notebook = nb.read(ROOT / "part3_agents/notebooks/Part3_Partner_Baseline_Standalone.ipynb", as_version=4)
        setup = notebook.cells[1].source
        self.assertNotIn('with_pip=True', setup)
        self.assertNotIn('system_site_packages=True', setup)
        self.assertIn("[sys.executable, '-m', 'pip', '--python', str(PYTHON)", setup)
        self.assertIn('prepare_python_environment(ENV)', setup)
        self.assertIn("'-I', '-m', 'pip', 'check'", setup)
        self.assertIn('USE_HF_TOKEN = False', setup)

    def test_auth_uses_existing_environment_without_prompt(self):
        configure = self.authentication_helper()
        env = {"HF_TOKEN": "fictional-environment-credential"}
        def forbidden(*args):
            self.fail("An existing credential must not require another secret or prompt.")
        self.assertEqual(configure(env, forbidden, forbidden), "environment")

    def test_auth_uses_secret_without_prompt_or_output(self):
        configure = self.authentication_helper()
        env, output = {}, io.StringIO()
        with redirect_stdout(output), redirect_stderr(output):
            result = configure(env, lambda: " fictional-secret-value ",
                               lambda _: self.fail("Secret was already available."))
        self.assertEqual(result, "notebook secret")
        self.assertEqual(env["HF_TOKEN"], "fictional-secret-value")
        self.assertEqual(output.getvalue(), "")

    def test_auth_missing_secret_uses_hidden_prompt_without_echo(self):
        configure = self.authentication_helper()
        env, output = {}, io.StringIO()
        def missing():
            raise RuntimeError("fictional secret permission denied")
        with redirect_stdout(output), redirect_stderr(output):
            result = configure(env, missing, lambda _: " fictional-hidden-value ")
        self.assertEqual(result, "hidden prompt")
        self.assertEqual(env["HF_TOKEN"], "fictional-hidden-value")
        self.assertEqual(output.getvalue(), "")

    def test_auth_background_or_plaintext_fallback_is_blocked(self):
        import warnings
        from getpass import GetPassWarning
        configure = self.authentication_helper()
        def unavailable(_):
            raise EOFError("fictional background execution")
        def plaintext(_):
            warnings.warn("fictional unsafe fallback", GetPassWarning)
            self.fail("Must stop before accepting an echoing input.")
        for prompt in [unavailable, plaintext]:
            env = {}
            with self.assertRaisesRegex(RuntimeError, "interactively"):
                configure(env, lambda: "", prompt)
            self.assertNotIn("HF_TOKEN", env)
        with self.assertRaisesRegex(RuntimeError, "No token entered"):
            configure({}, lambda: "", lambda _: "  ")

    def unpack(self, path):
        with zipfile.ZipFile(io.BytesIO(payload())) as bundle:
            bundle.extractall(path)
        name = "baseline_runtime"
        spec = importlib.util.spec_from_file_location(name, path / "baseline_runtime.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_attribution_hashes_and_executed_snapshot(self):
        folder = ROOT / "part3_agents/partner_snapshot"
        source = json.loads((folder / "PROVENANCE.json").read_text())
        for name, expected in source["files"].items():
            self.assertEqual(hashlib.sha256((folder / name).read_bytes()).hexdigest(), expected, name)
        notebook = nb.read(folder / "Partner_Part3_Executed.ipynb", as_version=4)
        code = [c for c in notebook.cells if c.cell_type == "code"]
        self.assertEqual(len(code), 24)
        self.assertTrue(all(c.execution_count is not None for c in code))
        self.assertFalse([o for c in code for o in c.outputs if o.output_type == "error"])

    def test_packaged_notebook_compiles_and_has_no_fabricated_outputs(self):
        path = ROOT / "part3_agents/notebooks/Part3_Partner_Baseline_Standalone.ipynb"
        notebook = nb.read(path, as_version=4)
        nb.validate(notebook)
        self.assertLess(path.stat().st_size, 900_000)
        for cell in notebook.cells:
            if cell.cell_type == "code":
                compile(cell.source, "baseline-cell", "exec")
                self.assertEqual(cell.outputs, [])
        setup = ast.parse(notebook.cells[1].source)
        assignments = {node.targets[0].id: ast.literal_eval(node.value) for node in setup.body
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in ["PAYLOAD", "BUNDLE_SHA256"]}
        self.assertEqual(base64.b64decode(assignments["PAYLOAD"]), payload())
        self.assertEqual(hashlib.sha256(payload()).hexdigest(), assignments["BUNDLE_SHA256"])

    def test_mock_software_smoke_no_fake_real_metrics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            module = self.unpack(root)
            result = module.run(root / "smoke_outputs", mock_only=True)
            self.assertTrue(result["mock_only"])
            self.assertTrue((root / "smoke_outputs/SMOKE_ONLY.json").exists())
            self.assertFalse((root / "baseline_outputs/baseline_metrics.json").exists())
            self.assertTrue(list((root / "smoke_outputs/traces").glob("*.json")))

    def test_safety_checker_failure_blocks_instead_of_passing(self):
        with tempfile.TemporaryDirectory() as temporary:
            module = self.unpack(Path(temporary))
            from agents.backends import BackendConfig, RealBackends
            backend = module.strict_backend(BackendConfig())
            with patch.object(RealBackends, "nsfw_check", side_effect=RuntimeError("fictional checker failure")):
                unsafe, reason = backend.nsfw_check(None)
            self.assertTrue(unsafe)
            self.assertIn("blocked", reason)
            self.assertIn("fictional checker failure", backend.safety_errors[0])

    def test_private_access_failure_happens_before_large_downloads(self):
        with tempfile.TemporaryDirectory() as temporary:
            module = self.unpack(Path(temporary))
            with patch("huggingface_hub.HfApi") as api, patch("huggingface_hub.snapshot_download") as download:
                api.return_value.model_info.side_effect = PermissionError("fictional private access denied")
                with self.assertRaises(PermissionError):
                    module.resolve_models("fictional-test-secret", Path(temporary) / "lock.json")
                self.assertEqual(api.return_value.model_info.call_args.args[0],
                                 "Shibin22/sd15-naruto-run1-lora-r4")
                download.assert_not_called()

    def test_unsafe_output_refusal_is_not_confused_with_checker_outage(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as temporary:
            module = self.unpack(Path(temporary))
            from agents.state import AgentState
            def state():
                return AgentState(status="error", error="generation failed: Generated image blocked before saving: unsafe",
                                  use_critique=False, max_revisions=0)
            orch = SimpleNamespace(b=SimpleNamespace(safety_errors=[]), run=lambda *a, **k: state(),
                                   save_trace=lambda s: None)
            result = module.baseline_run(orch, {"request": "fictional fixture"}, Path(temporary))
            self.assertEqual(result.status, "refused")
            self.assertFalse(result.error)
            orch.b.safety_errors = ["fictional checker outage"]
            result = module.baseline_run(orch, {"request": "fictional fixture"}, Path(temporary))
            self.assertEqual(result.status, "error")
            self.assertTrue(result.error)

    def test_export_contains_integrity_manifest_and_not_environment_cache(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"
            root.mkdir()
            module = self.unpack(root)
            (root / ".venv").mkdir()
            (root / ".venv/should_not_export").write_text("fictional cache")
            archive = module.export(root / "baseline_outputs")
            with zipfile.ZipFile(archive) as bundle:
                self.assertFalse(any(".venv" in name for name in bundle.namelist()))
                manifest = json.loads(bundle.read("BUNDLE_MANIFEST.json"))
                self.assertFalse(manifest["complete_real_metrics_present"])
                for name, digest in manifest["files"].items():
                    self.assertEqual(hashlib.sha256(bundle.read(name)).hexdigest(), digest)

    def test_selected_good_part1_notebook_is_byte_preserved(self):
        path = ROOT / "part1_rag/notebooks/Lab2_Part1_Kaggle_Final.ipynb"
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
            "73cab0044dde8763b7f8e95511288d888da21962d55a405606769b1959be1c01")

    def test_executed_cpu_smoke_notebook_and_zip(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"
            root.mkdir()
            self.unpack(root)
            result = subprocess.run([sys.executable, str(root / "baseline_runtime.py"), "smoke-notebook"],
                cwd=root, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout)
            notebook = nb.read(root / "Baseline_SMOKE_Executed.ipynb", as_version=4)
            self.assertTrue(notebook.metadata.execution.completed)
            self.assertTrue(notebook.metadata.execution.mock_only)
            code = [c for c in notebook.cells if c.cell_type == "code"]
            self.assertTrue(all(c.execution_count is not None for c in code))
            self.assertFalse([o for c in code for o in c.outputs if o.output_type == "error"])
            self.assertTrue(list(Path(temporary).glob("Lab2_Part3_Baseline_*.zip")))


if __name__ == "__main__":
    unittest.main(verbosity=2)
