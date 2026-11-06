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
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "part2_diffusion/src"))
sys.path.insert(0, str(ROOT / "tools"))
from submission import human_results, audit
import setup_runtime
from blinding import prepare_blind_pairs


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
