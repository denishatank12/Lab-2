"""Summarize saved notebook evidence without changing or rerunning the originals."""
import argparse
import hashlib
import json
from pathlib import Path
import re


def inspect(path):
    notebook = json.loads(path.read_text())
    cells = [c for c in notebook["cells"] if c["cell_type"] == "code"]
    errors, text = [], []
    for index, cell in enumerate(cells):
        for output in cell.get("outputs", []):
            if output["output_type"] == "error":
                errors.append({"code_cell": index + 1, "type": output["ename"], "message": output["evalue"]})
            elif output["output_type"] == "stream":
                text.append("".join(output.get("text", [])))
    combined = "\n".join(text)
    return {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "code_cells": len(cells), "executed_cells": sum(c.get("execution_count") is not None for c in cells),
            "saved_errors": errors, "gpu_names": sorted(set(re.findall(r'"gpu": "([^"]+)"', combined))),
            "validator_40_rows": "valid: 40 rows" in combined,
            "environment_missing_scipy_file": "No such file or directory" in combined and "scipy/stats/_mvn" in combined,
            "interpretation": "Saved execution evidence only; successful execution and schema validation do not establish answer quality or grading completeness."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("notebooks", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {"notebooks": [inspect(p) for p in args.notebooks],
              "part2_conclusion": "The supplied Part 2 notebook failed in setup. No fine-tuning, generation or model-quality results were produced.",
              "cause_limit": "The logs show missing package files and metadata in a shared environment. Concurrent installation is a plausible cause, not proven by the saved logs alone."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
