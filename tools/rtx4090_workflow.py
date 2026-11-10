"""Guard the RTX 4090 run and reuse verified Part 2 evidence without training."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from run_checkpoint import atomic_json, digest, runtime_identity


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_hardware(identity, available_ram, free_disk):
    """Inputs are injectable for tests; production passes actual measured values."""
    if not identity.get("cuda") or "4090" not in identity.get("gpu", ""):
        raise RuntimeError("This notebook targets an RTX 4090. Select that GPU runtime, not the Mac or a CPU kernel.")
    if identity.get("total_memory", 0) < 20 * 2**30:
        raise RuntimeError("The visible RTX 4090 does not expose its expected VRAM. Check the selected GPU or container limits.")
    if available_ram < 8 * 2**30:
        raise RuntimeError("Keep at least 8 GiB of host RAM available. Close other programs before the overnight run.")
    if free_disk < 8 * 2**30:
        raise RuntimeError("Keep at least 8 GiB of disk free beyond installed packages and model caches. A cold setup needs about 40 GB overall.")


def profile(root=ROOT):
    import psutil
    import torch

    identity = runtime_identity()
    available = psutil.virtual_memory().available
    free_disk = shutil.disk_usage(root).free
    check_hardware(identity, available, free_disk)
    # Verify an actual CUDA operation, rather than relying only on a GPU name.
    x = torch.randn(32, 32, device="cuda", dtype=torch.float16)
    if not torch.isfinite(x @ x.T).all().item():
        raise RuntimeError("The CUDA matrix multiplication check failed.")
    torch.cuda.synchronize()
    del x
    torch.cuda.empty_cache()
    record = dict(hardware=identity, available_host_ram_gib=available / 2**30,
                  free_disk_gib=free_disk / 2**30, cuda_kernel_passed=True,
                  scope="Actual RTX 4090 identity and CUDA check, not a 16-GB-card run or answer-quality result.")
    atomic_json(root / "runtime_checks/rtx4090_profile.json", record)
    print(json.dumps(record, indent=2), flush=True)
    return record


def part2_evidence(root=ROOT):
    """Validate the original cohort and pinned adapters, not merely file presence."""
    from PIL import Image
    import yaml

    p2 = root / "part2_diffusion"
    config = yaml.safe_load((p2 / "config.yaml").read_text())
    output = p2 / config["output_root"]
    links = json.loads((p2 / "evidence/hub_links.json").read_text())
    hashes = {}
    for run, rank in [("run1", 4), ("run2", 8)]:
        weights = output / run / "pytorch_lora_weights.safetensors"
        hashes[run] = sha256(weights)
        if hashes[run] != links[run + "_adapter_sha256"]:
            raise ValueError(f"{run} is not the original uploaded adapter. Do not mix runs or borrow its old upload record.")
        metrics = json.loads((output / run / "training_metrics.json").read_text())
        run_config = json.loads((output / run / "run_config.json").read_text())
        if run_config != {**config, "rank": rank} or metrics.get("rank") != rank:
            raise ValueError(f"{run} configuration differs from the recorded experiment.")
        if metrics.get("steps") != config["max_train_steps"]:
            raise ValueError(f"{run} did not complete the required training steps.")
        for key in ["final_loss", "training_seconds", "gpu_hours", "peak_allocated_mib"]:
            value = metrics.get(key)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{run} has no valid saved {key} measurement.")
    manifest_path = output / "generations/manifest.json"
    if sha256(manifest_path) != links["images_manifest_sha256"]:
        raise ValueError("The generation manifest does not match the original uploaded 120-image cohort.")
    manifest = json.loads(manifest_path.read_text())
    expected = {(m, i, seed) for m in ["base", "run1", "run2"] for i in range(20) for seed in [0, 1]}
    actual = {(r["model"], r["prompt_index"], r["seed"]) for r in manifest}
    if len(manifest) != 120 or actual != expected or len({r["signature"] for r in manifest}) != 1:
        raise ValueError("Part 2 needs all 120 unique model/prompt/seed combinations under one protocol.")
    for row in manifest:
        path = output / "generations" / row["file"]
        if not path.resolve().is_relative_to((output / "generations").resolve()):
            raise ValueError("A generation file escaped its results folder.")
        with Image.open(path) as image:
            if image.size != (512, 512) or image.mode != "RGB":
                raise ValueError(f"Unexpected image shape or mode: {path.name}")
            image.verify()
    evaluation = json.loads((output / "evaluation.json").read_text())
    rows = evaluation.get("metrics", [])
    if len(rows) != 3 or {r["model"] for r in rows} != {"base", "run1", "run2"}:
        raise ValueError("Part 2 metrics are incomplete.")
    for row in rows:
        for key in ["kid_mean", "clip_score_mean", "lpips_seed_pair_mean_256px"]:
            value = row.get(key)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"Missing or nonfinite {row['model']} metric: {key}")
    memorization = json.loads((output / "memorization.json").read_text())
    if any(len([r for r in memorization if r.get("model") == run]) != 10 for run in ["run1", "run2"]):
        raise ValueError("Saved nearest-training-image checks are incomplete.")
    required_images = [output / "generations" / f"comparison_{i:02d}.png" for i in range(8)]
    required_images += [output / f"memorization_{run}_{i}.png" for run in ["run1", "run2"] for i in range(3)]
    for path in required_images:
        with Image.open(path) as image:
            image.verify()
    return dict(computational_evidence_verified=True, adapter_sha256=hashes, generated_images=120,
                manifest_sha256=sha256(manifest_path), training_repeated=False,
                scope="Saved Part 2 computation only. Human ratings, access and team requirements are audited separately.")


def discover_archive(root=ROOT):
    # No recursive drive scan, URL guessing, or automatic import of arbitrary ZIPs.
    folders = [root, root.parent, Path.cwd(), Path.home() / "Downloads", Path("/content"), Path("/app")]
    candidates = sorted({p.resolve() for folder in folders if folder.is_dir()
                         for p in folder.glob("Part2_results_*.zip") if p.is_file()})
    if len(candidates) > 1:
        raise RuntimeError("Multiple Part 2 ZIPs were found. Set PART2_RESULTS_ZIP to the correct complete original archive.")
    return candidates[0] if candidates else None


def reuse_part2(archive=None, root=ROOT):
    previous_error = None
    try:
        record = part2_evidence(root)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        previous_error = exc
        record = None
    if record is None:
        archive = Path(archive).expanduser().resolve() if archive else discover_archive(root)
        if archive is None or not archive.is_file():
            raise FileNotFoundError(
                "The completed Part 2 outputs are not on this runtime. Put your original Part2_results_*.zip "
                "on this GPU machine and set PART2_RESULTS_ZIP in the first cell, then rerun. "
                "Downloading only the Hub adapter cannot restore the measured metrics and comparison figures. "
                f"No training was started. Local evidence check: {previous_error}")
        sys.path.insert(0, str(root / "part2_diffusion/src"))
        from import_result_bundle import import_bundle
        import_bundle(archive, root / "part2_diffusion")
        record = part2_evidence(root)
        record["imported_archive"] = str(archive)
        record["archive_sha256"] = sha256(archive)
    seed_selected_adapter(root)
    atomic_json(root / "runtime_checks/rtx4090_part2_reuse.json", record)
    print(json.dumps(record, indent=2), flush=True)
    return record


def seed_selected_adapter(root=ROOT):
    """Restore the exact uploaded adapter from its local export, without a token."""
    links = json.loads((root / "part2_diffusion/evidence/hub_links.json").read_text())
    p2 = root / "part2_diffusion"
    import yaml
    config = yaml.safe_load((p2 / "config.yaml").read_text())
    source = p2 / config["output_root"] / "run1"
    expected = links["run1_adapter_sha256"]
    if sha256(source / "pytorch_lora_weights.safetensors") != expected:
        raise ValueError("Local adapter does not match the pinned original. Part 3 was not started.")
    destination = root / "part3_agents/outputs/hub_adapter"
    destination.mkdir(parents=True, exist_ok=True)
    weights = destination / "pytorch_lora_weights.safetensors"
    if weights.exists() and sha256(weights) != expected:
        raise ValueError("A different Part 3 adapter exists here. Preserve it in another folder before proceeding.")
    for name in ["pytorch_lora_weights.safetensors", "run_config.json", "training_metrics.json"]:
        if not (destination / name).is_file():
            temporary = destination / (name + ".importing")
            shutil.copyfile(source / name, temporary)
            os.replace(temporary, destination / name)
    provenance = dict(repo_id=links["run1"].split("huggingface.co/")[1], revision=links["run1_commit"],
                      adapter_sha256=sha256(weights), path=str(destination.resolve()),
                      source="Hash-verified local export of the original uploaded adapter; no new Hub download or access verification.")
    atomic_json(destination / "hub_provenance.json", provenance)
    return provenance


def completion(root=ROOT, smoke=False):
    """A completed launcher is different from successful model executions."""
    import nbformat
    checks = {}
    for part, name in [("part1_rag", "Part1_RAG_Run.ipynb"), ("part3_agents", "Part3_Agents_Run.ipynb")]:
        path = root / "output/executed" / part / name
        try:
            notebook = nbformat.read(path, as_version=4)
            state = notebook.metadata.get("overnight_execution", {})
            codes = [c for c in notebook.cells if c.cell_type == "code"]
            source = nbformat.read(root / part / "notebooks" / name, as_version=4)
            checks[part + "_executed_notebook"] = bool(state.get("completed") and codes
                and state.get("smoke_only") is smoke
                and state.get("source_cells_sha256") == digest([c.source for c in source.cells])
                and all(c.execution_count is not None for c in codes)
                and not any(o.output_type == "error" for c in codes for o in c.outputs))
        except (OSError, ValueError):
            checks[part + "_executed_notebook"] = False
    if not smoke:
        checks["part2_saved_evidence"] = part2_evidence(root)["computational_evidence_verified"]
        progress = json.loads((root / "runtime_checks/overnight_progress.json").read_text())
        checks["full_sequential_run_finished"] = progress.get("status") == "completed" and progress.get("smoke_only") is False
        p1 = root / "part1_rag/outputs/v2"
        checks["part1_submission_exists"] = (p1 / "submission.csv").is_file()
        evaluation = json.loads((p1 / "evaluation/evaluation.json").read_text())
        checks["part1_both_systems_evaluated"] = {r.get("system") for r in evaluation.get("results", [])} == {"dense_baseline", "hybrid_multimodal"}
        latest = json.loads((root / "part3_agents/outputs/v2/latest_run.json").read_text())
        folder = Path(latest["folder"])
        expected_scenarios = {r["id"] for r in json.loads((root / "part3_agents/eval/scenarios.json").read_text())}
        expected_red = {r["id"]: r for r in json.loads((root / "part3_agents/eval/red_team_cases.json").read_text())}
        for name in ["scenario_traces.json", "scenario_traces_no_critique.json"]:
            traces = json.loads((folder / name).read_text())
            checks[name + "_15_traces"] = len(traces) == 15 and {r["scenario_id"] for r in traces} == expected_scenarios
            checks[name + "_no_runtime_errors"] = all(not r.get("error") for r in traces)
            metadata = json.loads((folder / name).with_suffix('.metadata.json').read_text())
            checks[name + "_models_enabled"] = metadata.get("models_enabled") is True
        for name in ["red_team_results.json", "red_team_results_no_critique.json"]:
            result = json.loads((folder / name).read_text())
            cases = result.get("cases", [])
            checks[name + "_10_cases"] = len(cases) == 10 and {r["id"] for r in cases} == set(expected_red)
            checks[name + "_no_runtime_errors"] = all(not r.get("error") or (
                r.get("decision") == "invalid_image" and r["id"] in expected_red
                and expected_red[r["id"]]["image_key"] in {"corrupt", "empty", "missing"}) for r in cases)
            checks[name + "_models_and_images"] = result.get("models_enabled") is True and result.get("image_based") is True
    record = dict(smoke_only=smoke, checks=checks, computational_checks_passed=all(checks.values()),
                  submission_ready=False,
                  scope="CPU smoke is not GPU evidence. Full computational success still does not replace real ratings, team review or external submission steps.")
    atomic_json(root / "runtime_checks/rtx4090_completion.json", record)
    print(json.dumps(record, indent=2), flush=True)
    if not all(checks.values()):
        raise RuntimeError("At least one completion check failed. Keep this diagnostic and the saved results; do not claim submission readiness.")
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["profile", "reuse-part2", "completion"])
    parser.add_argument("--archive")
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    if args.action == "profile":
        profile()
    elif args.action == "reuse-part2":
        reuse_part2(args.archive)
    else:
        completion(smoke=args.smoke_only)


if __name__ == "__main__":
    main()
