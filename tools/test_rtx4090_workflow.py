"""Synthetic temporary fixtures only, never human ratings or real GPU results."""
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import rtx4090_workflow as workflow


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def part2_fixture(root):
    from PIL import Image
    import yaml

    p2 = root / "part2_diffusion"
    p2.mkdir(parents=True)
    config = {"output_root": "outputs", "max_train_steps": 1200, "seed": 266}
    (p2 / "config.yaml").write_text(yaml.safe_dump(config))
    links = {"run1": "https://huggingface.co/FIXTURE_ONLY/run1", "run1_commit": "FIXTURE_ONLY_REVISION"}
    for run, rank in [("run1", 4), ("run2", 8)]:
        weights = p2 / "outputs" / run / "pytorch_lora_weights.safetensors"
        weights.parent.mkdir(parents=True)
        weights.write_bytes(("NOT_REAL_ADAPTER_" + run).encode())
        links[run + "_adapter_sha256"] = workflow.sha256(weights)
        write_json(weights.parent / "training_metrics.json", dict(rank=rank, steps=1200, final_loss=1.0,
            training_seconds=1.0, gpu_hours=1 / 3600, peak_allocated_mib=1.0))
        write_json(weights.parent / "run_config.json", {**config, "rank": rank})
    generated = p2 / "outputs/generations"
    generated.mkdir()
    manifest = []
    for model in ["base", "run1", "run2"]:
        for i in range(20):
            for seed in [0, 1]:
                filename = f"{model}_{i:02d}_{seed}.png"
                Image.new("RGB", (512, 512), "gray").save(generated / filename)
                manifest.append(dict(model=model, prompt_index=i, seed=seed, file=filename, signature="FIXTURE_ONLY"))
    write_json(generated / "manifest.json", manifest)
    links["images_manifest_sha256"] = workflow.sha256(generated / "manifest.json")
    write_json(p2 / "evidence/hub_links.json", links)
    write_json(p2 / "outputs/evaluation.json", {"metrics": [dict(model=m, kid_mean=0.0,
        clip_score_mean=0.0, lpips_seed_pair_mean_256px=0.0) for m in ["base", "run1", "run2"]]})
    write_json(p2 / "outputs/memorization.json", [dict(model=m, fixture_only=True) for m in ["run1", "run2"] for _ in range(10)])
    for path in ([generated / f"comparison_{i:02d}.png" for i in range(8)] +
                 [p2 / f"outputs/memorization_{run}_{i}.png" for run in ["run1", "run2"] for i in range(3)]):
        Image.new("RGB", (4, 4), "gray").save(path)
    return p2


class HardwareTests(unittest.TestCase):
    def identity(self):
        return dict(cuda=True, gpu="NVIDIA GeForce RTX 4090", total_memory=24 * 2**30)

    def test_actual_4090_profile_resource_inputs(self):
        workflow.check_hardware(self.identity(), 9 * 2**30, 9 * 2**30)

    def test_cpu_or_other_gpu_is_not_certified(self):
        for identity in [dict(cuda=False), {**self.identity(), "gpu": "Tesla T4"}]:
            with self.assertRaises(RuntimeError):
                workflow.check_hardware(identity, 20 * 2**30, 80 * 2**30)

    def test_restricted_vram_is_detected(self):
        with self.assertRaises(RuntimeError):
            workflow.check_hardware({**self.identity(), "total_memory": 6 * 2**30}, 9 * 2**30, 9 * 2**30)

    def test_ram_and_disk_fail_before_models(self):
        for ram, disk in [(7 * 2**30, 9 * 2**30), (9 * 2**30, 7 * 2**30)]:
            with self.assertRaises(RuntimeError):
                workflow.check_hardware(self.identity(), ram, disk)


class ReuseTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.p2 = part2_fixture(self.root)

    def test_complete_cohort_reused_without_training_or_hub_access(self):
        with patch.dict('os.environ', {"HF_TOKEN": ""}), redirect_stdout(io.StringIO()):
            result = workflow.reuse_part2(root=self.root)
        self.assertTrue(result["computational_evidence_verified"])
        self.assertFalse(result["training_repeated"])
        original = self.p2 / "outputs/run1/pytorch_lora_weights.safetensors"
        copy = self.root / "part3_agents/outputs/hub_adapter/pytorch_lora_weights.safetensors"
        self.assertEqual(original.read_bytes(), copy.read_bytes())
        self.assertIn("no new Hub download", json.loads((copy.parent / "hub_provenance.json").read_text())["source"])

    def test_wrong_adapter_cannot_borrow_original_upload_record(self):
        (self.p2 / "outputs/run1/pytorch_lora_weights.safetensors").write_bytes(b"CHANGED FIXTURE")
        with self.assertRaises(ValueError):
            workflow.part2_evidence(self.root)

    def test_missing_image_is_not_a_completed_run(self):
        (self.p2 / "outputs/generations/base_00_0.png").unlink()
        with self.assertRaises(FileNotFoundError):
            workflow.part2_evidence(self.root)

    def test_wrong_shape_is_rejected(self):
        from PIL import Image
        Image.new("RGB", (8, 8)).save(self.p2 / "outputs/generations/base_00_0.png")
        with self.assertRaises(ValueError):
            workflow.part2_evidence(self.root)

    def test_incomplete_training_rejected(self):
        path = self.p2 / "outputs/run2/training_metrics.json"
        metrics = json.loads(path.read_text()); metrics["steps"] = 50
        write_json(path, metrics)
        with self.assertRaises(ValueError):
            workflow.part2_evidence(self.root)

    def test_nonfinite_metric_is_not_a_result(self):
        path = self.p2 / "outputs/evaluation.json"
        value = json.loads(path.read_text()); value["metrics"][0]["kid_mean"] = float("nan")
        write_json(path, value)
        with self.assertRaises(ValueError):
            workflow.part2_evidence(self.root)

    def test_changed_manifest_not_reused(self):
        path = self.p2 / "outputs/generations/manifest.json"
        value = json.loads(path.read_text()); value[1] = value[0]
        write_json(path, value)
        with self.assertRaises(ValueError):
            workflow.part2_evidence(self.root)

    def test_original_zip_import_in_fresh_runtime(self):
        archive = self.root / "original_fixture.zip"
        with zipfile.ZipFile(archive, 'w') as bundle:
            for path in self.p2.rglob('*'):
                if path.is_file() and "evidence" not in path.parts:
                    bundle.write(path, path.relative_to(self.p2))
        with tempfile.TemporaryDirectory() as temporary:
            fresh = Path(temporary)
            p2 = fresh / "part2_diffusion"
            (p2 / "evidence").mkdir(parents=True)
            import shutil
            shutil.copy2(self.p2 / "config.yaml", p2 / "config.yaml")
            shutil.copy2(self.p2 / "evidence/hub_links.json", p2 / "evidence/hub_links.json")
            with redirect_stdout(io.StringIO()):
                record = workflow.reuse_part2(archive, root=fresh)
            self.assertFalse(record["training_repeated"])
            self.assertEqual(record["generated_images"], 120)
            self.assertEqual(record["archive_sha256"], workflow.sha256(archive))

    def test_missing_archive_stops_before_expensive_work(self):
        (self.p2 / "outputs/generations/base_00_0.png").unlink()
        with patch.object(workflow, 'discover_archive', return_value=None), self.assertRaisesRegex(FileNotFoundError, "No training was started"):
            workflow.reuse_part2(root=self.root)


class NotebookTests(unittest.TestCase):
    def test_new_launcher_keeps_full_computation_and_download(self):
        import nbformat
        path = ROOT / 'Lab2_RTX4090_Run_All.ipynb'
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        sources = '\n'.join(c.source for c in notebook.cells)
        for item in ['reuse-part2', 'overnight_runner.py', 'overnight_preflight.py',
                     'final_submission_audit.py', 'offer_download', 'PART2_RESULTS_ZIP']:
            self.assertIn(item, sources)
        self.assertNotIn('train_lora.py', sources)
        self.assertIn('rtx4090-v1', sources)
        for cell in notebook.cells:
            if cell.cell_type == 'code':
                compile(cell.source, 'rtx4090-code-cell', 'exec')

    def test_dataset_metadata_without_images_triggers_real_preparation(self):
        sys.path[:0] = [str(ROOT / 'part2_diffusion/src'), str(ROOT / 'part3_agents/src')]
        import prepare_evaluation
        with tempfile.TemporaryDirectory() as temporary:
            with patch('prepare_data.prepared', return_value=False), \
                 patch.object(prepare_evaluation.subprocess, 'run', side_effect=RuntimeError('PREPARATION WAS REQUESTED')):
                with self.assertRaisesRegex(RuntimeError, 'PREPARATION WAS REQUESTED'):
                    prepare_evaluation.prepare(Path(temporary))


class CompletionTests(unittest.TestCase):
    def fixture(self, root, smoke=False):
        import nbformat
        from run_checkpoint import digest
        for part, name in [('part1_rag', 'Part1_RAG_Run.ipynb'), ('part3_agents', 'Part3_Agents_Run.ipynb')]:
            n = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell("print('FIXTURE ONLY')", execution_count=1,
                outputs=[nbformat.v4.new_output('stream', name='stdout', text='FIXTURE ONLY\n')])])
            source = root / part / 'notebooks' / name
            source.parent.mkdir(parents=True); nbformat.write(n, source)
            n.metadata['overnight_execution'] = dict(completed=True, smoke_only=smoke, source_cells_sha256=digest([c.source for c in n.cells]))
            path = root / 'output/executed' / part / name
            path.parent.mkdir(parents=True); nbformat.write(n, path)
        write_json(root / 'runtime_checks/overnight_progress.json', dict(status='completed', smoke_only=smoke))
        p1 = root / 'part1_rag/outputs/v2'
        p1.mkdir(parents=True)
        (p1 / 'submission.csv').write_text('FIXTURE ONLY')
        write_json(p1 / 'evaluation/evaluation.json', dict(results=[{'system': s} for s in ['dense_baseline', 'hybrid_multimodal']]))
        folder = root / 'part3_agents/outputs/v2/fixture_run'
        write_json(folder.parent / 'latest_run.json', dict(folder=str(folder)))
        scenarios = [{'id': f'S{i}'} for i in range(15)]
        cases = [{'id': f'R{i}', 'image_key': 'corrupt' if i == 7 else 'heldout_0'} for i in range(10)]
        write_json(root / 'part3_agents/eval/scenarios.json', scenarios)
        write_json(root / 'part3_agents/eval/red_team_cases.json', cases)
        for name in ['scenario_traces.json', 'scenario_traces_no_critique.json']:
            write_json(folder / name, [dict(scenario_id=r['id'], error='') for r in scenarios])
            write_json((folder / name).with_suffix('.metadata.json'), dict(models_enabled=True))
        for name in ['red_team_results.json', 'red_team_results_no_critique.json']:
            write_json(folder / name, dict(models_enabled=True, image_based=True,
                cases=[dict(id=r['id'], error='Expected broken bytes' if i == 7 else '', decision='invalid_image' if i == 7 else 'refused')
                       for i, r in enumerate(cases)]))
        return folder

    def test_cpu_smoke_cannot_pass_as_gpu_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); self.fixture(root, smoke=True)
            with redirect_stdout(io.StringIO()):
                self.assertTrue(workflow.completion(root, smoke=True)['smoke_only'])
            with patch.object(workflow, 'part2_evidence', return_value=dict(computational_evidence_verified=True)), \
                 redirect_stdout(io.StringIO()), self.assertRaises(RuntimeError):
                workflow.completion(root)

    def test_expected_corrupt_input_is_not_a_model_exception(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); self.fixture(root)
            with patch.object(workflow, 'part2_evidence', return_value=dict(computational_evidence_verified=True)), redirect_stdout(io.StringIO()):
                result = workflow.completion(root)
            self.assertTrue(result['computational_checks_passed'])
            self.assertFalse(result['submission_ready'])

    def test_model_error_saved_in_normal_exit_is_not_hidden(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); folder = self.fixture(root)
            path = folder / 'scenario_traces.json'
            rows = json.loads(path.read_text()); rows[0]['error'] = 'FIXTURE MODEL EXCEPTION'
            write_json(path, rows)
            with patch.object(workflow, 'part2_evidence', return_value=dict(computational_evidence_verified=True)), \
                 redirect_stdout(io.StringIO()), self.assertRaises(RuntimeError):
                workflow.completion(root)

    def test_mock_scenarios_cannot_pass_full_completion(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); folder = self.fixture(root)
            write_json(folder / 'scenario_traces.metadata.json', dict(models_enabled=False))
            with patch.object(workflow, 'part2_evidence', return_value=dict(computational_evidence_verified=True)), \
                 redirect_stdout(io.StringIO()), self.assertRaises(RuntimeError):
                workflow.completion(root)


if __name__ == '__main__':
    unittest.main(verbosity=2)
