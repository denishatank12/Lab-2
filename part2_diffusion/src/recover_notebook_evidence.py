"""Recover real saved measurements and PNG outputs; never reconstruct absent weights or samples."""
from __future__ import annotations
import argparse
import base64
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def json_objects(text):
    decoder, offset = json.JSONDecoder(), 0
    while offset < len(text):
        start = text.find("{", offset)
        if start < 0:
            return
        try:
            value, length = decoder.raw_decode(text[start:])
        except ValueError:
            offset = start + 1
            continue
        if isinstance(value, dict):
            yield value
        offset = start + length


def recover(path, root=ROOT):
    path = Path(path)
    notebook = json.loads(path.read_text(encoding="utf-8"))
    code = [c for c in notebook["cells"] if c["cell_type"] == "code"]
    errors = [o for c in code for o in c.get("outputs", []) if o["output_type"] == "error"]
    if errors:
        raise ValueError("This evidence import expects an error-free saved run; it will not hide errors.")
    training, evaluation, data, runtime, grids = {}, None, None, None, []
    for cell in code:
        source = "".join(cell["source"])
        streams = "\n".join("".join(o.get("text", [])) for o in cell.get("outputs", []) if o["output_type"] == "stream")
        for value in json_objects(streams):
            if "train_lora.py" in source and value.get("rank") in [4, 8] and value.get("steps") == 1200:
                training["run1" if value["rank"] == 4 else "run2"] = value
            if value.get("generated_images") == 120 and {m.get("model") for m in value.get("metrics", [])} == {"base", "run1", "run2"}:
                evaluation = value
            if value.get("dataset") == "lambdalabs/naruto-blip-captions" and value.get("schema_version") == 2:
                data = value
            if "versions" in value and value.get("cuda_available") is True:
                runtime = value
        if "generate_eval.py" in source and "evaluate.py" in source:
            for output in cell.get("outputs", []):
                image = output.get("data", {}).get("image/png")
                if image:
                    grids.append(base64.b64decode("".join(image) if isinstance(image, list) else image, validate=True))
    if set(training) != {"run1", "run2"} or evaluation is None or data is None or runtime is None or len(grids) != 8:
        raise ValueError("Missing the two full training records, metrics, data summary, CUDA runtime, or eight grids.")
    review = json.loads((root / "evidence/visual_review.json").read_text())
    output = root / "outputs/notebook_evidence"
    (output / "generations").mkdir(parents=True, exist_ok=True)
    image_hashes = []
    for index, content in enumerate(grids):
        if not content.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError("Saved comparison output is not a PNG.")
        (output / "generations" / f"comparison_{index:02d}.png").write_bytes(content)
        image_hashes.append(hashlib.sha256(content).hexdigest())
    pending = ["three_real_raters_for_25_blind_pairs", "original_adapters_and_120_images_to_hugging_face",
               "nearest_training_image_files_for_report", "verified_account_and_team_details", "final_combined_ieee_report_and_submission_links"]
    result = {"ready": False, "evidence_only": True, "training": [{"model": name, **training[name]} for name in ["run1", "run2"]],
        "evaluation": evaluation, "data": data, "runtime": runtime, "review": review,
        "human": {"complete": False}, "hub_links": {}, "pending": pending,
        "evidence_directory": "outputs/notebook_evidence",
        "provenance": {"source_notebook": path.name, "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "code_cells": len(code), "executed_code_cells": sum(c.get("execution_count") is not None for c in code),
            "saved_errors": len(errors), "comparison_png_sha256": image_hashes,
            "scope": "Saved notebook evidence only. No adapter, individual sample, loss-history or memorization file is invented."}}
    summary = root / "evidence/saved_gpu_run.json"
    summary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print("Recovered real measurement tables and eight original comparison grids.")
    print("Missing artifact files are still marked pending; GPU training was not rerun.")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("notebook", type=Path)
    args = parser.parse_args()
    recover(args.notebook)


if __name__ == "__main__":
    main()
