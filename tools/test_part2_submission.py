"""Offline regression tests. All invented ballots/files here are temporary test fixtures, never results."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
import zipfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "part2_diffusion/src"))
sys.path.insert(0, str(ROOT / "tools"))
from submission import human_results, audit
import setup_runtime
from blinding import prepare_blind_pairs
from generation_state import prepare_generation_folder
from import_result_bundle import import_bundle


class TestResultImport(unittest.TestCase):
    """ZIP fixtures contain no real model or image measurements."""
    def fixture(self, archive, extra=None, omit=None):
        members = {"outputs/evaluation.json": "{}", "outputs/memorization.json": "[]"}
        rows = []
        for model in ["base", "run1", "run2"]:
            for index in range(20):
                for seed in [0, 1]:
                    name = f"{model}_{index:02d}_{seed}.png"
                    rows.append({"file": name})
                    members["outputs/generations/" + name] = "TEMPORARY IMAGE FIXTURE"
        members["outputs/generations/manifest.json"] = json.dumps(rows)
        for run in ["run1", "run2"]:
            for name in ["pytorch_lora_weights.safetensors", "training_metrics.json", "run_config.json"]:
                members[f"outputs/{run}/{name}"] = "TEMPORARY TEST FIXTURE"
        members["outputs/run1/checkpoint-1200/training_state.pt"] = "NEVER IMPORT"
        if extra:
            members.update(extra)
        if omit:
            members.pop(omit)
        with zipfile.ZipFile(archive, "w") as bundle:
            for name, value in members.items():
                bundle.writestr(name, value)

    def test_complete_import_preserves_changed_files_and_skips_optimizer_pickle(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "fixture.zip"
            self.fixture(archive)
            (root / "outputs").mkdir()
            (root / "outputs/evaluation.json").write_text("OLD FIXTURE")
            with redirect_stdout(io.StringIO()):
                result = import_bundle(archive, root)
            self.assertEqual(result["generation_records"], 120)
            self.assertFalse((root / "outputs/run1/checkpoint-1200").exists())
            self.assertEqual((Path(result["backup"]) / "outputs/evaluation.json").read_text(), "OLD FIXTURE")

    def test_missing_image_rejected_before_writes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "fixture.zip"
            self.fixture(archive, omit="outputs/generations/base_00_0.png")
            with self.assertRaises(ValueError):
                import_bundle(archive, root)
            self.assertFalse((root / "outputs").exists())

    def test_traversal_rejected_before_writes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "fixture.zip"
            self.fixture(archive, extra={"outputs/../../outside.txt": "UNSAFE FIXTURE"})
            with self.assertRaises(ValueError):
                import_bundle(archive, root)
            self.assertFalse((root / "outputs").exists())


class TestGenerationRecovery(unittest.TestCase):
    def fixture(self, output, signature="previous-protocol"):
        folder = output / "generations"
        folder.mkdir(parents=True)
        row = {"model": "run1", "prompt_index": 0, "seed": 0, "file": "run1_00_0.png", "signature": signature}
        (folder / "manifest.json").write_text(json.dumps([row]))
        (folder / row["file"]).write_bytes(b"TEMPORARY TEST IMAGE")
        for run in ["run1", "run2"]:
            path = output / run / "checkpoint-1200"
            path.mkdir(parents=True)
            (path.parent / "pytorch_lora_weights.safetensors").write_bytes(b"TEMPORARY ADAPTER TEST FIXTURE")
            (path / "training_state.pt").write_bytes(b"TEMPORARY CHECKPOINT TEST FIXTURE")
        (output / "evaluation.json").write_text('{"old": true}')
        (output / "human_run1").mkdir()
        (output / "human_run1/ratings.json").write_text("[]")
        return row

    def test_changed_protocol_archives_results_not_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "outputs"
            self.fixture(output)
            with redirect_stdout(io.StringIO()):
                folder, rows = prepare_generation_folder(output, "current-protocol")
            self.assertEqual(rows, [])
            self.assertEqual(list(folder.iterdir()), [])
            archives = list((output / "generation_history").iterdir())
            self.assertEqual(len(archives), 1)
            self.assertTrue((archives[0] / "generations/run1_00_0.png").is_file())
            self.assertTrue((archives[0] / "evaluation.json").is_file())
            self.assertTrue((archives[0] / "human_run1/ratings.json").is_file())
            self.assertFalse((output / "evaluation.json").exists())
            for run in ["run1", "run2"]:
                self.assertEqual((output / run / "pytorch_lora_weights.safetensors").read_bytes(), b"TEMPORARY ADAPTER TEST FIXTURE")
                self.assertTrue((output / run / "checkpoint-1200/training_state.pt").is_file())

    def test_same_protocol_resumes_without_archive(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "outputs"
            row = self.fixture(output, "same-protocol")
            folder, rows = prepare_generation_folder(output, "same-protocol")
            self.assertEqual(rows, [row])
            self.assertTrue((folder / row["file"]).is_file())
            self.assertFalse((output / "generation_history").exists())

    def test_explicit_fail_preserves_everything(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "outputs"
            self.fixture(output)
            with self.assertRaises(RuntimeError):
                prepare_generation_folder(output, "new", fail_on_change=True)
            self.assertTrue((output / "generations/manifest.json").is_file())
            self.assertFalse((output / "generation_history").exists())

    def test_interrupted_manifest_is_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "outputs"
            self.fixture(output)
            (output / "generations/manifest.json").write_text("[{")
            with redirect_stdout(io.StringIO()):
                _, rows = prepare_generation_folder(output, "new")
            self.assertEqual(rows, [])
            archive = next((output / "generation_history").iterdir())
            self.assertEqual((archive / "generations/manifest.json").read_text(), "[{")

    def test_generation_entrypoint_recovers_and_resumes_without_training(self):
        # Exercise the real 120-record orchestration with mocked model inference.
        # The tiny gray fixture images are not model-quality evidence.
        import generate_eval
        import yaml
        from PIL import Image
        class FakeScheduler:
            config = {}
            @classmethod
            def from_config(cls, config):
                return cls()
        class FakePipeline:
            calls = 0
            scheduler = FakeScheduler()
            safety_checker = object()
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                cls.calls += 1
                return cls()
            def enable_model_cpu_offload(self): pass
            def enable_vae_slicing(self): pass
            def enable_vae_tiling(self): pass
            def set_progress_bar_config(self, **kwargs): pass
            def load_lora_weights(self, folder): pass
            def __call__(self, *args, **kwargs):
                return type("Result", (), {"images": [Image.new("RGB", (4, 4), "gray")],
                                           "nsfw_content_detected": [False]})()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "outputs"
            self.fixture(output)
            cfg = {"output_root": "outputs", "base_model": "TEST FIXTURE ONLY",
                   "base_revision": "TEST REVISION", "variant": "fp16"}
            (root / "config.yaml").write_text(yaml.safe_dump(cfg))
            prompts = [{"prompt": f"TEST FIXTURE {i}", "category": "in_domain"} for i in range(20)]
            with patch.object(generate_eval, "ROOT", root), patch.object(generate_eval, "freeze_prompts", return_value=prompts), \
                 patch("torch.cuda.is_available", return_value=True), patch("torch.cuda.empty_cache"), \
                 patch("diffusers.StableDiffusionPipeline", FakePipeline), patch("diffusers.DDIMScheduler", FakeScheduler), \
                 patch.object(sys, "argv", ["generate_eval.py"]), redirect_stdout(io.StringIO()):
                generate_eval.main()
                generate_eval.main()
            manifest = json.loads((output / "generations/manifest.json").read_text())
            self.assertEqual(len(manifest), 120)
            self.assertEqual(len({r["signature"] for r in manifest}), 1)
            self.assertEqual(FakePipeline.calls, 3)  # Second invocation resumed, no model loads.
            self.assertTrue((output / "run1/checkpoint-1200/training_state.pt").exists())
            self.assertTrue((output / "run2/pytorch_lora_weights.safetensors").exists())
            self.assertEqual(len(list((output / "generation_history").iterdir())), 1)


class TestHumanJudgments(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.folder = Path(self.temporary.name)
        key = [{"pair_id": i, "final_is_left": i % 2 == 0} for i in range(25)]
        (self.folder / "human_evaluation_key.json").write_text(json.dumps(key))

    def tearDown(self):
        self.temporary.cleanup()

    def ballots(self):
        return [{"rater_id": rater, "role": role, "pair_id": i,
                 "domain_style": "A" if i % 2 == 0 else "B", "prompt_match": "tie"}
                for rater, role in [("fixture-one", "team_member"), ("fixture-two", "team_member"),
                                    ("fixture-outside", "external")] for i in range(25)]

    def test_missing_ratings_stays_pending(self):
        self.assertFalse(human_results(self.folder)["complete"])

    def test_no_raters_are_invented(self):
        (self.folder / "ratings.json").write_text("[]")
        self.assertFalse(human_results(self.folder)["complete"])

    def test_orientation_and_ties(self):
        (self.folder / "ratings.json").write_text(json.dumps(self.ballots()))
        result = human_results(self.folder)
        self.assertTrue(result["complete"])
        self.assertEqual(result["criteria"]["domain_style"]["win_rate"], 1)
        self.assertEqual(result["criteria"]["prompt_match"]["win_rate"], 0)
        self.assertEqual(result["criteria"]["prompt_match"]["tie_adjusted_win_rate"], .5)

    def test_two_raters_not_enough(self):
        (self.folder / "ratings.json").write_text(json.dumps(self.ballots()[:50]))
        self.assertFalse(human_results(self.folder)["complete"])

    def test_duplicate_ballot_rejected(self):
        rows = self.ballots()
        (self.folder / "ratings.json").write_text(json.dumps(rows + rows[:1]))
        with self.assertRaises(ValueError):
            human_results(self.folder)


class TestRuntimeSafety(unittest.TestCase):
    def test_setup_is_one_install_transaction_and_marks_only_success(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "requirements.txt").write_text("numpy==1.26.4\n")
            def create(folder):
                path = setup_runtime.python_in(folder)
                path.parent.mkdir(parents=True)
                path.touch()
            def health(path, cuda):
                return ".lab2-envs" in str(path)
            with patch.object(setup_runtime, "ROOT", root), patch.object(setup_runtime, "healthy", side_effect=health), \
                 patch.object(setup_runtime, "create_environment", side_effect=create), \
                 patch.object(setup_runtime.shutil, "disk_usage", return_value=type("Disk", (), {"free": 20 * 2**30})()), \
                 patch.object(setup_runtime.subprocess, "run") as run:
                setup_runtime.setup(True)
            commands = [c.args[0] for c in run.call_args_list]
            transactions = [c for c in commands if "-r" in c]
            self.assertEqual(len(transactions), 1)
            self.assertIn("torch==2.5.1+cu121", transactions[0])
            self.assertIn("--no-cache-dir", transactions[0])
            self.assertTrue((root / ".lab2_runtime_cuda.json").is_file())

    def test_failed_preflight_does_not_mark_ready(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "requirements.txt").write_text("numpy==1.26.4\n")
            with patch.object(setup_runtime, "ROOT", root), patch.object(setup_runtime, "healthy", return_value=True), \
                 patch.object(setup_runtime.subprocess, "run", side_effect=subprocess.CalledProcessError(1, "fixture")):
                with self.assertRaises(subprocess.CalledProcessError):
                    setup_runtime.setup(False)
            self.assertFalse((root / ".lab2_runtime_cpu.json").exists())

    def test_missing_or_broken_interpreter_not_healthy(self):
        self.assertFalse(setup_runtime.healthy(Path("/this/path/does/not/exist/python"), False))
        with patch.object(setup_runtime.subprocess, "run", side_effect=subprocess.CalledProcessError(1, "fixture")):
            self.assertFalse(setup_runtime.healthy(Path(sys.executable), False))

    def test_lock_serializes_and_recovers_after_kill(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "setup.lock"
            script = ("import sys,time; from pathlib import Path; "
                      f"sys.path.insert(0,{str(ROOT / 'tools')!r}); "
                      "from setup_runtime import installation_lock; "
                      "\nwith installation_lock(Path(sys.argv[1])):\n print('LOCKED',flush=True);time.sleep(20)")
            child = subprocess.Popen([sys.executable, "-c", script, str(path)], stdout=subprocess.PIPE, text=True)
            try:
                self.assertEqual(child.stdout.readline().strip(), "LOCKED")
                with self.assertRaises(TimeoutError):
                    with setup_runtime.installation_lock(path, timeout=.1):
                        self.fail("Two installers acquired the same lock")
                child.terminate()
                child.wait(timeout=5)
                with setup_runtime.installation_lock(path, timeout=.1):
                    pass
            finally:
                if child.poll() is None:
                    child.kill()
                    child.wait()
                child.stdout.close()

    def test_audit_does_not_call_empty_run_complete(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "config.yaml").write_text("output_root: outputs\nmax_train_steps: 1200\n")
            with redirect_stdout(io.StringIO()):
                result = audit(root)
            self.assertFalse(result["ready"])
            self.assertFalse(result["checks"]["run1_cuda_training"])
            self.assertFalse(result["checks"]["25_blind_pairs_3_real_raters"])
            self.assertTrue((root / "outputs/review_template.json").is_file())


class TestBlinding(unittest.TestCase):
    def test_25_blind_pairs_and_unchanged_ratings(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            generated = output / "generations"
            generated.mkdir()
            rows = []
            for model in ["base", "run1"]:
                for i in range(20):
                    for seed in [0, 1]:
                        filename = f"{model}_{i:02d}_{seed}.png"
                        Image.new("RGB", (4, 4), "gray").save(generated / filename)
                        rows.append({"prompt_index": i, "prompt": "TEMPORARY TEST FIXTURE", "file": filename})
            folder = prepare_blind_pairs(output, rows, "run1")
            self.assertEqual(len(list(folder.glob("blinded_pair_*.png"))), 25)
            ratings = folder / "ratings.json"
            ratings.write_text("[]")
            prepare_blind_pairs(output, rows, "run1")
            self.assertEqual(ratings.read_text(), "[]")
            before = (folder / "blinded_pair_00.png").read_bytes()
            with self.assertRaises(RuntimeError):
                prepare_blind_pairs(output, rows, "run1", seed=999)
            self.assertEqual((folder / "blinded_pair_00.png").read_bytes(), before)


class TestReportLayout(unittest.TestCase):
    def test_numeric_tables_and_loss_plot_render_from_temporary_fixtures(self):
        # This tests formatting only. These made-up values never leave the temp directory.
        import report_pdf
        from pypdf import PdfReader
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "config.yaml").write_text("output_root: outputs\nmax_train_steps: 1200\nlearning_rate: 0.0001\ntrain_batch_size: 1\ngradient_accumulation_steps: 4\n")
            training = []
            for name, rank in [("run1", 4), ("run2", 8)]:
                folder = root / "outputs" / name
                folder.mkdir(parents=True)
                (folder / "loss_history.json").write_text(json.dumps([{"step": i + 1, "loss": 1 / (i + 1)} for i in range(100)]))
                training.append({"model": name, "rank": rank, "trainable_parameters": 10000,
                    "learning_rate": .0001, "steps": 1200, "batch_size": 1, "gradient_accumulation_steps": 4,
                    "peak_allocated_mib": 1000., "training_seconds": 10., "gpu_hours": 10 / 3600})
            result = {"ready": False, "training": training, "data": {},
                "evaluation": {"metrics": [{"model": model, "kid_mean": .1, "clip_score_mean": 10.,
                     "lpips_seed_pair_mean_256px": .5} for model in ["base", "run1", "run2"]]},
                "human": {"complete": False}, "review": {}, "pending": ["TEMPORARY TEST FIXTURE"], "hub_links": {}}
            path = root / "formatting_test.pdf"
            with patch.object(report_pdf, "ROOT", root), patch.object(report_pdf, "audit", return_value=result), \
                 patch.object(sys, "argv", ["report_pdf.py", "--output", str(path)]), redirect_stdout(io.StringIO()):
                report_pdf.main()
            reader = PdfReader(path)
            self.assertLessEqual(len(reader.pages), 12)
            text = "\n".join(page.extract_text() for page in reader.pages)
            self.assertIn("Measured", text.replace("measured", "Measured"))
            self.assertIn("DRAFT", text)
            self.assertIn("LPIPS", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
