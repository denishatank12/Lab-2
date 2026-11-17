"""Baseline-only runner for the attributed, executed partner snapshot.

The generated notebook embeds this file and the snapshot. Mock checks test
software, not model quality. Only a real CUDA run may create baseline metrics.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import statistics
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "source"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    temporary.replace(path)


def read_cases():
    folder = ROOT / "source/eval"
    scenarios = json.loads((folder / "scenarios.json").read_text())["scenarios"]
    attacks = json.loads((folder / "redteam.json").read_text())["cases"]
    if len(scenarios) != 15 or len(attacks) != 10:
        raise ValueError("Expected the partner's original 15 scenarios and 10 red-team cases.")
    return scenarios, attacks


def verify_source():
    record = json.loads((ROOT / "PROVENANCE.json").read_text())
    for relative, expected in record["files"].items():
        if relative.startswith("source/") and sha(ROOT / relative) != expected:
            raise ValueError(f"Partner snapshot changed: {relative}")
    return record


def authenticate():
    """Use anonymous public access by default; private access is explicit opt-in."""
    if os.environ.get("LAB2_P3_USE_HF_TOKEN", "0") != "1":
        return False
    from huggingface_hub import get_token
    token = os.environ.get("HF_TOKEN") or get_token()
    if not token:
        raise RuntimeError("Enable the HF_TOKEN notebook secret with access to the private partner adapter.")
    os.environ["HF_TOKEN"] = token
    return token


def resolve_models(token, output):
    """Resolve and record immutable revisions, including the actual adapter."""
    from huggingface_hub import HfApi, snapshot_download
    from agents.backends import BackendConfig
    config = BackendConfig(lora_repo=os.environ.get("LAB2_P3_LORA_REPO", "Shibin22/sd15-naruto-run1-lora-r4"),
                           cpu_offload=True)
    old = json.loads(output.read_text()) if output.is_file() else {}
    repos = [config.vlm_model, config.sd_model, config.pix2pix_model, config.clip_model, config.lora_repo]
    api = HfApi(token=token)
    records = {}
    # Check the requested adapter first, before downloading large public models.
    for repo in [config.lora_repo] + repos[:-1]:
        previous = old.get(repo, {})
        info = api.model_info(repo, revision=previous.get("revision") or None)
        records[repo] = {"revision": info.sha}
        if repo == config.lora_repo:
            names = [f.rfilename for f in info.siblings]
            if not any(n.endswith(".safetensors") for n in names):
                raise RuntimeError("The requested adapter has no safetensors weights; base-model fallback is forbidden.")
    save(output, records)
    # Local snapshot paths ensure each from_pretrained/load_lora call uses these exact revisions.
    for name in ["vlm_model", "sd_model", "pix2pix_model", "clip_model", "lora_repo"]:
        repo = getattr(config, name)
        local = snapshot_download(repo, revision=records[repo]["revision"], token=token,
            allow_patterns=(["*.json", "*.txt", "*.model", "*.jinja", "*.fp16.safetensors"]
                            if name in ["sd_model", "pix2pix_model"] else None),
            ignore_patterns=["*.bin", "*.msgpack", "*.h5", "*.onnx", "*.tflite", "*.ot",
                             "onnx/*", "openvino/*", "flax/*", "*.ckpt", "*.safetensors.index.json.tmp"])
        setattr(config, name, local)
        records[repo]["local_path"] = local
    save(output, records)
    return config, records


def prepare_inputs(token, output):
    """Download all 16 real photos; never silently substitute drawn placeholders."""
    import importlib.util
    from datasets import load_dataset
    from huggingface_hub import HfApi
    folder = ROOT / "source/eval"
    spec = json.loads((folder / "inputs.json").read_text())
    lock = output / "input_dataset.json"
    previous = json.loads(lock.read_text()) if lock.is_file() else {}
    repo = spec["coco_dataset"]
    revision = previous.get("revision") or HfApi(token=token).dataset_info(repo).sha
    save(lock, {"dataset": repo, "split": spec["coco_split"], "revision": revision})
    photos = output / "inputs"
    photos.mkdir(exist_ok=True)
    desired = {c["row"]: c for c in spec["coco"]}
    dataset = load_dataset(repo, split=spec["coco_split"], revision=revision, streaming=True, token=token)
    fetched = set()
    for row, sample in enumerate(dataset):
        if row in desired:
            item = desired[row]
            image = sample["image"].convert("RGB")
            image.save(photos / item["name"], quality=92)
            fetched.add(item["name"])
        if row >= max(desired):
            break
    if fetched != {c["name"] for c in spec["coco"]}:
        raise RuntimeError("Incomplete real input download; no placeholders will be scored.")
    loader = importlib.util.spec_from_file_location("partner_input_images", folder / "make_images.py")
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    # All real photos already exist. This call creates only the specified attack files.
    module.make_all(str(photos), offline=True)
    save(output / "input_manifest.json", {p.name: sha(p) for p in sorted(photos.iterdir()) if p.is_file()})
    return photos


def strict_backend(config):
    """Small declared safety patch around the otherwise unchanged partner backend."""
    from agents.backends import RealBackends

    class CheckedBackends(RealBackends):
        safety_errors = None

        def nsfw_check(self, image):
            try:
                return super().nsfw_check(image)
            except Exception as error:
                self.safety_errors.append(f"{type(error).__name__}: {error}")
                # The original nodes catch checker exceptions and continue. Returning
                # unsafe closes that path and prevents an unchecked image being accepted.
                return True, "safety checker unavailable; output blocked"

        def generate(self, *args, **kwargs):
            image, metadata = super().generate(*args, **kwargs)
            unsafe, reason = self.nsfw_check(image)
            if unsafe:
                raise RuntimeError("Generated image blocked before saving: " + reason)
            return image, metadata

    backend = CheckedBackends(config)
    backend.safety_errors = []
    return backend


def baseline_run(orch, case, images):
    """No retry/revision loop. Retain the partner's diagnostic critic and safety."""
    state = orch.run(case["request"], str(images / case["image"]) if case.get("image") else "",
                     use_critique=False, max_revisions=0, seed=266)
    if (state.error.startswith("generation failed: Generated image blocked before saving:")
            and not getattr(orch.b, "safety_errors", [])):
        # A genuine unsafe-output refusal is a result, not a broken model runtime.
        # Checker outages are different and still stop the run with an error.
        state.status, state.refused = "refused", True
        state.refusal_reason = state.error
        state.error = ""
        state.final_image = ""
        state.final_message = "The generated image failed the safety check and was not saved or displayed."
        orch.save_trace(state)
    return state


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate(backend, config, images, output):
    from agents.evaluate import auto_check_redteam, auto_check_scenario
    from agents.graph import Orchestrator
    from agents.state import AgentState
    scenarios, attacks = read_cases()
    orch = Orchestrator(backend, config, out_dir=str(output / "generated"), trace_dir=str(output / "traces"))
    rows = {"scenarios": [], "redteam": []}
    for group, cases in [("scenarios", scenarios), ("redteam", attacks)]:
        for case in cases:
            checkpoint = output / "states" / f"{group}_{case['id']}.json"
            if checkpoint.is_file():
                state = AgentState.model_validate_json(checkpoint.read_text())
                # Retry genuine runtime failures; retain successful completed work.
                if state.error:
                    state = baseline_run(orch, case, images)
            else:
                state = baseline_run(orch, case, images)
            save(checkpoint, state.model_dump())
            if state.use_critique or state.max_revisions or state.revisions:
                raise ValueError("Baseline unexpectedly enabled a revision loop.")
            if state.final_image and not Path(state.final_image).is_file():
                raise FileNotFoundError("A resumed state has a missing output image; restore the complete ZIP.")
            if group == "scenarios":
                ok, why = auto_check_scenario(state, case)
                score = {"auto_pass": ok}
            else:
                success, why = auto_check_redteam(state, case, backend)
                score = {"attack_success": success}
            first = state.critiques[0] if state.critiques else None
            rows[group].append({"id": case["id"], "category": case["category"], "request": case["request"],
                "status": state.status, **score, "why": why, "seconds": state.total_seconds,
                "revisions": state.revisions, "request_id": state.request_id, "final_image": state.final_image,
                "final_message": state.final_message, "critic_decision": first.decision if first else ""})
            write_csv(output / f"baseline_{group}.csv", rows[group])
            print(group, case["id"], state.status, score, flush=True)
            # Malformed-image tests deliberately return status=error with no node exception.
            # Real model failures are not converted to a completed evaluation.
            if state.error or backend.safety_errors:
                save(output / "runtime_failure.json", {"case": case["id"], "error": state.error,
                    "safety_errors": backend.safety_errors})
                raise RuntimeError(f"Runtime failure in {case['id']}; saved states and partial CSVs are retained.")
    metrics = {"condition": "baseline_without_revision_loop", "real_model_execution": True,
        "scenarios": len(rows["scenarios"]),
        "automatic_proxy_passes": sum(r["auto_pass"] for r in rows["scenarios"]),
        "automatic_proxy_rate": statistics.mean(r["auto_pass"] for r in rows["scenarios"]),
        "redteam_cases": len(rows["redteam"]),
        "attack_success_rate": statistics.mean(r["attack_success"] for r in rows["redteam"]),
        "median_scenario_seconds": statistics.median(r["seconds"] for r in rows["scenarios"]),
        "human_task_success": None, "human_critic_agreement": None,
        "human_ratings_status": "Both team members must inspect the actual outputs independently.",
        "timing_scope": "per-request wall time, including any lazy model initialization; resumed cases retain their original time",
        "partner_comparison": "Partner results are external automatic proxies; no causal timing comparison across hardware/runtime patches."}
    save(output / "baseline_metrics.json", metrics)
    # Never overwrite a partially or fully filled ballot on a resume.
    form = output / "baseline_human_ratings.csv"
    if not form.exists():
        write_csv(form, [{"id": r["id"], "request": r["request"], "final_image": r["final_image"],
            "final_message": r["final_message"], "critic_decision": r["critic_decision"],
            "member1_pass": "", "member2_pass": "", "member1_critic_agrees": "",
            "member2_critic_agrees": "", "resolution": ""} for r in rows["scenarios"]])
    return metrics


def smoke(output):
    """Explicit mock-only checks. No real baseline_metrics.json is created."""
    from agents.backends import BackendConfig, MockBackends
    from agents.graph import Orchestrator
    from PIL import Image
    image = output / "fixture.png"
    Image.new("RGB", (256, 256), (120, 130, 140)).save(image)
    config = BackendConfig(lora_repo="fictional-test-adapter")
    backend = MockBackends(cfg=config, vlm_scores=[1])
    orch = Orchestrator(backend, config, out_dir=str(output / "generated"), trace_dir=str(output / "traces"))
    state = baseline_run(orch, {"request": "Make this image a watercolor painting.", "image": image.name}, output)
    assert state.status == "done", state.model_dump()
    assert state.revisions == 0 and state.max_revisions == 0 and not state.use_critique
    assert len(state.generations) == 1 and len(state.critiques) == 1
    refused = baseline_run(orch, {"request": "Create explicit nude content."}, output)
    assert refused.status == "refused"
    assert len(read_cases()[0]) == 15 and len(read_cases()[1]) == 10
    result = {"mock_only": True, "software_checks_passed": True,
        "checks": ["one generation, no revisions despite low critic score", "safety refusal", "15+10 fixed cases", "JSON traces"],
        "not_evidence_of": ["CUDA compatibility", "image quality", "human success", "GPU timing"]}
    save(output / "SMOKE_ONLY.json", result)
    return result


def run(output, mock_only=False):
    output.mkdir(parents=True, exist_ok=True)
    save(output / "execution_status.json", {"completed": False, "mock_only": mock_only})
    source = verify_source()
    if mock_only:
        return smoke(output)
    import torch
    import numpy as np
    if not torch.cuda.is_available():
        raise RuntimeError("A real CUDA GPU is required. CPU mock checks cannot replace your baseline run.")
    token = authenticate()
    random.seed(266)
    np.random.seed(266)
    torch.manual_seed(266)
    torch.cuda.manual_seed_all(266)
    torch.cuda.reset_peak_memory_stats()
    gpu = torch.cuda.get_device_properties(0)
    identity = {"partner_commit": source["commit"], "baseline_runner_sha256": sha(__file__),
        "python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": gpu.name, "vram_bytes": gpu.total_memory, "seed": 266,
        "versions": {n: importlib.metadata.version(n) for n in
                     ["transformers", "diffusers", "accelerate", "peft", "bitsandbytes", "pydantic", "datasets"]}}
    save(output / "runtime_identity.json", identity)
    print(json.dumps(identity, indent=2), flush=True)
    config, models = resolve_models(token, output / "model_revisions.json")
    images = prepare_inputs(token, output)
    fingerprint = {"config": config.__dict__, "models": models,
        "input_manifest": json.loads((output / "input_manifest.json").read_text()),
        "runner": sha(__file__), "partner_commit": source["commit"], "condition": "no_revision_loop"}
    guard = output / "run_configuration.json"
    if guard.exists() and json.loads(guard.read_text()) != fingerprint:
        raise RuntimeError("Run inputs/settings changed. Choose a new output folder; previous results are safe.")
    save(guard, fingerprint)
    backend = strict_backend(config)
    from PIL import Image
    # Preflight all actual model families before scoring the 25 cases.
    backend.nsfw_check(Image.open(images / "coco_stop_sign.jpg").convert("RGB"))
    if backend.safety_errors:
        raise RuntimeError("The required real safety checker could not load: " + backend.safety_errors[-1])
    backend._load_vlm()
    backend.clip_score(Image.open(images / "coco_stop_sign.jpg").convert("RGB"), "a stop sign")
    backend.img2img_pipe(with_lora=True)
    if not backend._lora_loaded:
        raise RuntimeError("The requested adapter did not load; a base-model substitution is not allowed.")
    backend.pix2pix_pipe()
    save(output / "gpu_preflight.json", {"passed": True, "adapter_loaded": True,
        "note": "Loads and CLIP inference passed. Full generation is evaluated in the cases below."})
    start = time.perf_counter()
    result = evaluate(backend, config, images, output)
    result.update({"evaluation_wall_seconds_this_invocation": time.perf_counter() - start,
        "peak_torch_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_torch_reserved_bytes": torch.cuda.max_memory_reserved(),
        "memory_note": "PyTorch allocator peaks on GPU0; not total driver/process VRAM."})
    save(output / "baseline_metrics.json", result)
    save(output / "execution_status.json", {"completed": True, "mock_only": False})
    return result


def export(output):
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    archive = ROOT.parent / f"Lab2_Part3_Baseline_{stamp}_{time.time_ns() % 1000000:06}.zip"
    items = [p for p in ROOT.rglob("*") if p.is_file() and not any(x in p.parts for x in
             [".venv", "__pycache__", ".ipynb_checkpoints"]) and p.suffix != ".zip"]
    # Only this workflow's source/outputs are inside ROOT; caches and HF secrets stay outside it.
    manifest = {str(p.relative_to(ROOT)): sha(p) for p in items}
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in items:
            bundle.write(path, str(path.relative_to(ROOT)))
        status_path = output / "execution_status.json"
        status = json.loads(status_path.read_text()) if status_path.is_file() else {}
        bundle.writestr("BUNDLE_MANIFEST.json", json.dumps({"files": manifest,
            "complete_real_metrics_present": (output / "baseline_metrics.json").exists()
                and status.get("completed") is True and not status.get("mock_only", True),
            "human_ratings_must_be_real": True}, indent=2))
    print("RESULT ZIP:", archive, "\nSHA256:", sha(archive), flush=True)
    return archive


def execute_notebook(mock_only=False):
    """Keep a genuine executed notebook inside the ZIP, even if a cell fails."""
    import atexit
    import tempfile
    import nbformat as nb
    from nbclient import NotebookClient
    from jupyter_client import KernelManager
    notebook = nb.v4.new_notebook(cells=[
        nb.v4.new_markdown_cell("# Denisha's Part 3 baseline\n\nPartner implementation credited in PROVENANCE.json. "
            "The revision loop is off; diagnostic critique and safety remain active. "
            + ("This is a mock-only software smoke test, not a real baseline." if mock_only else
               "Human ratings are separate from automatic proxy results.")),
        nb.v4.new_code_cell("import sys, json\nfrom pathlib import Path\n"
            "sys.path.insert(0, str(Path.cwd()))\n"
            "import baseline_runtime as baseline\n"
            f"MOCK_ONLY = {mock_only!r}\n"
            "OUTPUT = baseline.ROOT / ('smoke_outputs' if MOCK_ONLY else 'baseline_outputs')\n"
            "result = baseline.run(OUTPUT, mock_only=MOCK_ONLY)\nprint(json.dumps(result, indent=2))"),
        nb.v4.new_code_cell("from IPython.display import display, Image\n"
            "import pandas as pd\n"
            "if not MOCK_ONLY:\n"
            "    results = pd.read_csv(OUTPUT / 'baseline_scenarios.csv')\n"
            "    display(results)\n"
            "    display(pd.read_csv(OUTPUT / 'baseline_redteam.csv'))\n"
            "    for path in results.final_image.dropna():\n"
            "        display(Image(filename=path, width=320))\n")])
    notebook.metadata["execution"] = {"completed": False, "mock_only": mock_only}
    path = ROOT / ("Baseline_SMOKE_Executed.ipynb" if mock_only else "Part3_Baseline_Executed.ipynb")
    bootstrap = '''from ipykernel.kernelapp import IPKernelApp
from ipykernel.ipkernel import IPythonKernel
from traitlets.config import Config
class IsolatedApp(IPKernelApp):
    def init_config_files(self):
        self.config_file_paths = [self.profile_dir.location]
        self.config_files = []
    def load_config_file(self, *args, **kwargs):
        return None
c = Config()
c.IPKernelApp.kernel_class = IPythonKernel
c.InteractiveShellApp.extensions = []
c.InteractiveShellApp.extra_extensions = []
c.InteractiveShellApp.exec_lines = []
c.InteractiveShellApp.exec_files = []
IsolatedApp.launch_instance(config=c)
'''
    def saved(**kwargs):
        save(path, json.loads(nb.writes(notebook)))
    saved()
    with tempfile.TemporaryDirectory(prefix="lab2-p3-kernel-") as profile:
        environment = os.environ.copy()
        for name in ["JUPYTER_CONFIG_PATH", "JUPYTER_CONFIG_DIR", "IPYTHONSTARTUP", "PYTHONSTARTUP",
                     "PYTHONPATH", "PYTHONHOME", "MATPLOTLIBRC"]:
            environment.pop(name, None)
        environment.update(JUPYTER_NO_CONFIG="1", IPYTHONDIR=profile,
                           MPLBACKEND="module://matplotlib_inline.backend_inline")
        manager = KernelManager(kernel_name="python3")
        manager.kernel_spec.argv = [sys.executable, "-I", "-c", bootstrap, "-f", "{connection_file}"]
        manager.kernel_spec.env = {}
        manager.kernel_spec.metadata = {}
        client = NotebookClient(notebook, km=manager, timeout=None, allow_errors=False,
            resources={"metadata": {"path": str(ROOT)}}, on_cell_executed=saved,
            on_cell_start=lambda cell_index, **kw: print("Executing baseline cell", cell_index + 1, flush=True))
        try:
            client.execute(env=environment, cleanup_kc=True)
            notebook.metadata["execution"]["completed"] = True
        finally:
            atexit.unregister(client._cleanup_kernel)
            if client.kc is not None:
                client.kc.stop_channels()
            saved()
            if manager.has_kernel:
                manager.shutdown_kernel(now=True)
            manager.cleanup_resources()
    print("Saved executed notebook:", path, flush=True)
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["run", "smoke", "export", "notebook", "smoke-notebook"])
    args = parser.parse_args()
    destination = ROOT / ("smoke_outputs" if "smoke" in args.action else "baseline_outputs")
    if args.action == "export":
        export(destination)
    else:
        try:
            if "notebook" in args.action:
                execute_notebook(mock_only=args.action == "smoke-notebook")
            else:
                print(json.dumps(run(destination, mock_only=args.action == "smoke"), indent=2))
        finally:
            export(destination)
