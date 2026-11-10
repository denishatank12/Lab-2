"""Self-contained notebook support: checked data download and original Hub recovery."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import sys
import time
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from run_checkpoint import atomic_json


def sha256(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def checked_copy(source, destination, expected):
    source, destination = Path(source), Path(destination)
    if sha256(source) != expected:
        raise ValueError(f"Hash mismatch for {source.name}; no result was substituted.")
    if destination.is_file():
        if sha256(destination) != expected:
            raise ValueError(f"An incompatible file already exists at {destination}. It was preserved.")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".restoring")
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def unpack(data, expected_sha256, parent):
    """No pre-existing repo, git executable, or package import is required."""
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError("The embedded notebook bundle is incomplete or changed.")
    root = Path(parent) / ("Lab2_Standalone_" + expected_sha256[:16])
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        if archive.testzip() is not None:
            raise ValueError("The embedded ZIP failed its integrity check.")
        manifest = json.loads(archive.read("STANDALONE_MANIFEST.json"))
        files = manifest["files"]
        if len(archive.namelist()) != len(set(archive.namelist())):
            raise ValueError("Duplicate archive paths are not permitted.")
        if set(archive.namelist()) != set(files) | {"STANDALONE_MANIFEST.json"}:
            raise ValueError("The archive does not match its file manifest.")
        # Check every path and hash before creating or changing any file.
        for member in archive.infolist():
            name = member.filename
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts or "\\" in name or stat.S_ISLNK(member.external_attr >> 16):
                raise ValueError("Unsafe embedded path.")
            if member.is_dir():
                raise ValueError("Only regular manifest files are permitted.")
            if name != "STANDALONE_MANIFEST.json" and hashlib.sha256(archive.read(name)).hexdigest() != files[name]:
                raise ValueError(f"Embedded source hash mismatch: {name}")
            target = root / name
            if target.is_symlink() or not target.resolve().is_relative_to(root.resolve()):
                raise ValueError("An extraction target escaped the standalone folder.")
            if target.is_file() and target.read_bytes() != archive.read(name):
                raise ValueError(f"Local edits at {target} were preserved. Use a new empty parent folder.")
        root.mkdir(parents=True, exist_ok=True)
        for name in archive.namelist():
            target = root / name
            if not target.is_file():
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(name))
    return root


def download_resources(root=ROOT, opener=urllib.request.urlopen, sleeper=time.sleep):
    """Only course data is fetched; executable Python is already inside the notebook."""
    resources = json.loads((root / "STANDALONE_MANIFEST.json").read_text())["resources"]
    for row in resources:
        target = root / row["path"]
        if target.exists():
            if sha256(target) != row["sha256"]:
                raise ValueError(f"Changed input was preserved: {target}. Do not mix question/data versions.")
            continue
        if not target.resolve().is_relative_to(root.resolve()):
            raise ValueError("Resource path escaped the standalone folder.")
        if target.suffix.lower() == ".py":
            raise ValueError("Python source must be embedded, not downloaded at runtime.")
        target.parent.mkdir(parents=True, exist_ok=True)
        last = None
        for attempt in range(4):
            temporary = target.with_name(target.name + ".downloading")
            try:
                request = urllib.request.Request(row["url"], headers={"User-Agent": "DATA266-Lab2-standalone"})
                with opener(request, timeout=90) as response, temporary.open("wb") as stream:
                    shutil.copyfileobj(response, stream)
                if sha256(temporary) != row["sha256"]:
                    raise ValueError("The downloaded input did not match its pinned SHA-256.")
                os.replace(temporary, target)
                print("Downloaded and verified", row["path"], flush=True)
                break
            except (OSError, ValueError) as exc:
                last = exc
                if attempt < 3:
                    sleeper(2**attempt)
            finally:
                if temporary.exists():
                    temporary.unlink()
        else:
            raise RuntimeError(f"Could not fetch {target.name}. Check internet, then rerun; completed files are retained.") from last


def restore_original_part2(root=ROOT, snapshot=None, token=None):
    """Recover real prior files, not fabricate a new T4 training or evaluation run."""
    from huggingface_hub import HfApi, snapshot_download
    from PIL import Image
    from rtx4090_workflow import part2_evidence, seed_selected_adapter

    snapshot = snapshot or snapshot_download
    token = token or os.environ.get("HF_TOKEN")
    evidence = root / "standalone_evidence/original_part2"
    record = json.loads((evidence / "RESTORE_MANIFEST.json").read_text())
    links = json.loads((root / "part2_diffusion/evidence/hub_links.json").read_text())
    if snapshot is snapshot_download:
        if not token:
            raise RuntimeError("Enter a Hugging Face read token in the hidden prompt. Private artifacts cannot be fetched anonymously.")
        check_hub_access(root, token)
    output = root / "part2_diffusion/outputs"
    for run in ["run1", "run2"]:
        source = Path(snapshot(links[run].split("huggingface.co/")[1],
            revision=links[run + "_commit"], token=token,
            allow_patterns=["pytorch_lora_weights.safetensors", "run_config.json", "training_metrics.json"],
            max_workers=2))
        expected = record["hub_files"][run]
        for name, digest in expected.items():
            checked_copy(source / name, output / run / name, digest)
    files = record["saved_files"]
    # These files are original measured records from the supplied result ZIP.
    # New GPU timings or human judgments are deliberately not synthesized.
    for relative, digest in files.items():
        checked_copy(evidence / relative, root / "part2_diffusion" / relative, digest)
    manifest_path = output / "generations/manifest.json"
    if sha256(manifest_path) != links["images_manifest_sha256"]:
        raise ValueError("The restored manifest does not identify the original uploaded cohort.")
    source = Path(snapshot(links["images"].split("datasets/")[1], repo_type="dataset",
        revision=links["images_commit"], token=token,
        allow_patterns=list(record["hub_files"]["images"]), max_workers=2))
    for name, digest in record["hub_files"]["images"].items():
        checked_copy(source / name, output / "generations" / name, digest)
        with Image.open(output / "generations" / name) as image:
            if image.mode != "RGB" or image.size != (512, 512):
                raise ValueError(f"Unexpected original image: {name}")
            image.verify()
    rebuild_original_panels(root)
    result = part2_evidence(root)
    seed_selected_adapter(root)
    # Preserve any later real access/rights review. On a fresh runtime, retain
    # the original content-bound upload record, including its pending flags.
    if not (output / "hub_links.json").exists():
        registry = root / "part2_diffusion/evidence/hub_links.json"
        checked_copy(registry, output / "hub_links.json", sha256(registry))
    result.update(original_archive_sha256=record["original_archive_sha256"],
                  origin="Original supplied RTX 4090 run and pinned private Hub commits.",
                  trained_on_current_gpu=False, metrics_recomputed=False,
                  scope="Restored original files and display panels. Not a new Colab training run, evaluation run, human study or 16-GB training proof.")
    atomic_json(root / "runtime_checks/standalone_part2_restoration.json", result)
    print(json.dumps(result, indent=2), flush=True)
    return result


def check_hub_access(root=ROOT, token=None):
    from huggingface_hub import HfApi

    token = token or os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("A read token is needed for the private Part 2 repositories. Enter it only in the hidden prompt.")
    api = HfApi(token=token)
    links = json.loads((root / "part2_diffusion/evidence/hub_links.json").read_text())
    for name in ["run1", "run2", "images"]:
        repo_id = links[name].split("huggingface.co/")[1].removeprefix("datasets/")
        api.repo_info(repo_id, repo_type="dataset" if name == "images" else "model", revision=links[name + "_commit"])
    print("Read access verified for both pinned Part 2 adapters and the image dataset. No token was saved.", flush=True)


def rebuild_original_panels(root=ROOT):
    from PIL import Image, ImageDraw

    output = root / "part2_diffusion/outputs"
    generated = output / "generations"
    rows = json.loads((generated / "manifest.json").read_text())
    by_key = {(r["model"], r["prompt_index"], r["seed"]): r for r in rows}
    for index in range(8):
        path = generated / f"comparison_{index:02d}.png"
        if path.is_file():
            continue
        grid = Image.new("RGB", (1536, 544), "white")
        draw = ImageDraw.Draw(grid)
        for column, model in enumerate(["base", "run1", "run2"]):
            draw.text((column * 512 + 10, 8), model, fill="black")
            with Image.open(generated / by_key[(model, index, 0)]["file"]) as image:
                grid.paste(image, (column * 512, 32))
        grid.save(path)
    # The nearest-neighbor identities/scores below are restored measurements.
    # Only their side-by-side displays are rebuilt from the original pixels.
    memorization = json.loads((output / "memorization.json").read_text())
    train = root / "part2_diffusion/data/processed/train"
    for model in ["run1", "run2"]:
        closest = sorted([r for r in memorization if r["model"] == model], key=lambda r: -r["clip_cosine"])[:3]
        for index, pair in enumerate(closest):
            path = output / f"memorization_{model}_{index}.png"
            if path.is_file():
                continue
            grid = Image.new("RGB", (1024, 544), "white")
            ImageDraw.Draw(grid).text((5, 5), f"Generated | nearest training image, CLIP cosine {pair['clip_cosine']:.4f}", fill="black")
            for offset, source in [(0, generated / pair["generated_file"]), (512, train / pair["training_file"])]:
                with Image.open(source) as image:
                    grid.paste(image.convert("RGB").resize((512, 512)), (offset, 32))
            grid.save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["resources", "check-hub-access", "restore-part2", "record-part2", "complete"])
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    if args.action == "resources":
        download_resources()
    elif args.action == "check-hub-access":
        check_hub_access()
    elif args.action == "restore-part2":
        restore_original_part2()
    elif args.action == "record-part2":
        from overnight_runner import execute_saved
        execute_saved(ROOT / "part2_diffusion/notebooks/Part2_FineTune_Run.ipynb",
                      ROOT / "output/executed/part2_diffusion/Part2_FineTune_Run.ipynb")
    else:
        from rtx4090_workflow import completion
        result = completion(smoke=args.smoke_only)
        result["part2_policy"] = "Original RTX 4090 computation restored from pinned Hub artifacts; not retrained on the Colab GPU."
        atomic_json(ROOT / "runtime_checks/standalone_completion.json", result)


if __name__ == "__main__":
    main()
