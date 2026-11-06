"""Keep each generation protocol separate while leaving trained adapters untouched."""
from __future__ import annotations
from datetime import datetime, timezone
import json
from pathlib import Path
import uuid


def prepare_generation_folder(output, signature, fail_on_change=False, report=None):
    output = Path(output).resolve()
    if output == Path(output.anchor):
        raise ValueError("A filesystem root cannot be a generation output directory.")
    output.mkdir(parents=True, exist_ok=True)
    folder = output / "generations"
    record = folder / "manifest.json"
    previous, reason = [], None
    if record.exists():
        try:
            previous = json.loads(record.read_text(encoding="utf-8"))
            if not isinstance(previous, list) or any(not isinstance(r, dict) for r in previous):
                reason = "Invalid manifest structure"
            elif any(r.get("signature") != signature for r in previous):
                reason = "Saved generation fingerprint differs from the current protocol or adapters"
            elif any(not valid_row(row) for row in previous):
                reason = "Invalid generation record"
            elif len({(r["model"], r["prompt_index"], r["seed"]) for r in previous}) != len(previous):
                reason = "Duplicate generation records"
        except (ValueError, OSError):
            reason = "Unreadable or interrupted generation manifest"
    if not reason and folder.exists() and not previous and any(folder.iterdir()):
        reason = "Existing generation files have no usable manifest"
    if reason:
        if fail_on_change:
            raise RuntimeError(reason + ". No files were changed. Omit --fail-on-generation-change to archive and regenerate.")
        # Every target is named explicitly. Never move run1/run2, data or checkpoints.
        names = ["generations", "evaluation.json", "clip_scores_per_image.json", "review_candidates.json",
                 "memorization.json", "human_run1", "human_run2", "review.json", "review_template.json",
                 "submission_audit.json", "hub_links.json", "hub_export"]
        names += [f"memorization_{run}_{i}.png" for run in ["run1", "run2"] for i in range(3)]
        archive = output / "generation_history" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8])
        archive.mkdir(parents=True)
        # Record the recovery plan before moving anything so a partial move is traceable.
        plan = {"reason": reason, "new_signature": signature, "targets": names,
                "training_preserved": "Original run1/run2 adapters, checkpoints and training metrics remain in place."}
        (archive / "recovery.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
        for name in names:
            path = output / name
            if path.exists():
                path.rename(archive / name)
        if report is not None and Path(report).is_file():
            Path(report).rename(archive / "previous_findings.pdf")
        print(f"Previous generation results preserved at: {archive}", flush=True)
        print("Trained adapters and checkpoints are unchanged. Regenerating images and dependent metrics only.", flush=True)
        previous = []
    folder.mkdir(exist_ok=True)
    return folder, previous


def valid_row(row):
    return (row.get("model") in ["base", "run1", "run2"]
            and type(row.get("prompt_index")) is int and 0 <= row["prompt_index"] < 20
            and type(row.get("seed")) is int and row["seed"] in [0, 1]
            and isinstance(row.get("file"), str)
            and row["file"] == f"{row['model']}_{row['prompt_index']:02d}_{row['seed']}.png")
