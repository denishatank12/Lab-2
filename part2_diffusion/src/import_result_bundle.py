"""Import a complete result ZIP without extracting history, traversal paths or optimizer pickles."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def import_bundle(archive, root=ROOT):
    archive = Path(archive)
    with zipfile.ZipFile(archive) as bundle:
        if bundle.testzip() is not None:
            raise ValueError("ZIP integrity check failed. Do not import an incomplete download.")
        selected = []
        for member in bundle.infolist():
            path = PurePosixPath(member.filename)
            if path.is_absolute() or ".." in path.parts or "\\" in member.filename:
                raise ValueError("Unsafe ZIP member path.")
            if stat.S_ISLNK(member.external_attr >> 16):
                raise ValueError("Symbolic links are not accepted in the results bundle.")
            if member.is_dir() or any(p in ["generation_history", "hub_export"] or p.startswith("checkpoint-") for p in path.parts):
                continue
            if path.parts[0] not in ["outputs", "output", "data"] and member.filename != "config.yaml":
                continue
            selected.append(member)
        names = {m.filename for m in selected}
        required = ["outputs/evaluation.json", "outputs/generations/manifest.json", "outputs/memorization.json"]
        required += [f"outputs/{run}/{name}" for run in ["run1", "run2"] for name in ["pytorch_lora_weights.safetensors", "training_metrics.json", "run_config.json"]]
        if not set(required).issubset(names):
            raise ValueError("Bundle is missing required model or evaluation files.")
        manifest = json.loads(bundle.read("outputs/generations/manifest.json"))
        if len(manifest) != 120 or any(f"outputs/generations/{r['file']}" not in names for r in manifest):
            raise ValueError("The current 120-image cohort is not complete.")
        backup = root / "outputs/import_history" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        imported = []
        for member in selected:
            destination = root / member.filename
            if not destination.resolve().is_relative_to(root.resolve()):
                raise ValueError("Import target escaped the Part 2 folder.")
            content = bundle.read(member)
            if destination.is_file() and destination.read_bytes() != content:
                saved = backup / member.filename
                saved.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(destination, saved)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
            imported.append(member.filename)
    record = {"archive": archive.name, "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
              "imported_files": len(imported), "generation_records": len(manifest),
              "skipped": "Historical generations, Hub export copies and optimizer checkpoints remain in the original ZIP.",
              "backup": str(backup) if backup.exists() else None}
    (root / "outputs/import_record.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    import_bundle(args.archive)


if __name__ == "__main__":
    main()
