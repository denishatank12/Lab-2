"""Preserve the supplied GPU run and execute only a new, explicitly CPU review section."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
REVIEW_CODE = '''from pathlib import Path
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from IPython.display import Image, display

repo = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "part2_diffusion/src/submission.py").is_file())
part2 = repo / "part2_diffusion"
output = part2 / "outputs"
print("Postprocessing only; GPU training and image generation are not rerun.")
print("Review runtime:", sys.version.split()[0], sys.platform, datetime.now(timezone.utc).isoformat())
for name in ["hub_links.json", "visual_review.json"]:
    destination = "review.json" if name == "visual_review.json" else name
    if not (output / destination).exists():
        shutil.copy2(part2 / "evidence" / name, output / destination)
sys.path.insert(0, str(part2 / "src"))
from submission import audit
result = audit(part2)
print("\\nSelected model:", result["review"]["final_run"])
print(result["review"]["selection_reason"])
print("\\nVerified private artifact uploads:")
print(json.dumps(result["hub_links"], indent=2))
nearest = json.loads((output / "memorization.json").read_text())
nearest = sorted([r for r in nearest if r["model"] == result["review"]["final_run"]], key=lambda r:r["clip_cosine"], reverse=True)[:3]
print("\\nThree closest generated/training pairs:")
for index, row in enumerate(nearest):
    print(json.dumps(row))
    display(Image(filename=str(output / f"memorization_{result['review']['final_run']}_{index}.png")))
print("These displayed pairs share domain features but are not exact visible copies. This limited check does not prove that memorization is absent.")
subprocess.run([sys.executable, str(part2 / "src/report_pdf.py")], cwd=repo, check=True)
print("\\nRemaining required items:", result["pending"])
print("Human ratings are real-world judgments and are not supplied by this code.")
'''


def attach(source, destination):
    original = json.loads(source.read_text())
    errors = [o for c in original["cells"] if c["cell_type"] == "code"
              for o in c.get("outputs", []) if o["output_type"] == "error"]
    if errors:
        raise ValueError("Do not hide errors in the original GPU run.")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    section = nbformat.v4.new_notebook(cells=[
        nbformat.v4.new_markdown_cell(
            "## Review of the downloaded result bundle\n\n"
            "All cells above are preserved from the supplied RTX 4090 run. "
            "The section below was executed separately on a macOS CPU kernel after importing the real result ZIP. "
            "It reads saved measurements, shows original nearest-neighbor images, records verified private Hugging Face links "
            "and refreshes the report. It does not repeat or claim new GPU training. "
            "Its execution count belongs to that separate review kernel.\n\n"
            f"Original notebook SHA-256: `{digest}`."),
        nbformat.v4.new_code_cell(REVIEW_CODE)])
    NotebookClient(section, timeout=180, kernel_name="python3",
                   resources={"metadata": {"path": str(ROOT)}}).execute()
    original["cells"].extend(json.loads(nbformat.writes(section))["cells"])
    original["metadata"]["result_review"] = {
        "source_notebook_sha256": digest, "original_cells_preserved": True,
        "review_scope": "Separate CPU postprocessing of imported GPU artifacts; no GPU rerun."}
    destination.write_text(json.dumps(original, indent=1, ensure_ascii=False) + "\n")
    # Verifying the unchanged original cells also protects the saved figures and stdout.
    assert json.loads(destination.read_text())["cells"][:-2] == json.loads(source.read_text())["cells"]
    print("Saved GPU outputs unchanged and appended executed CPU review:", destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    attach(args.source, args.destination)
