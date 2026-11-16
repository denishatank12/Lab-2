"""Offline checks for attribution, baseline-only routing and standalone packaging."""
import ast
import base64
import hashlib
import importlib.util
import io
import json
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
