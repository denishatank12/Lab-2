"""Build one notebook to repair evaluation in an existing Kaggle v5 session."""
import hashlib
from pathlib import Path

import nbformat as nb

ROOT = Path(__file__).resolve().parents[1]


def build():
    judge = (ROOT / 'tools/citation_judge.py').read_bytes()
    driver = (ROOT / 'tools/recover_part1_evaluation.py').read_bytes()
    payload = {name: (data.decode(), hashlib.sha256(data).hexdigest()) for name, data in
               [('citation_judge.py', judge), ('recover_part1_evaluation.py', driver)]}
    setup = '''from pathlib import Path
import hashlib, json, os, subprocess
ROOT = Path(os.environ.get("LAB2_PART1_SAVED_ROOT", "/kaggle/working/Lab2_Standalone_0ffb808bd0b0650c"))
if not (ROOT / "part1_rag/outputs/v2/submission.provenance.json").is_file():
    raise RuntimeError("The saved v5 run is not in this session. Do not Run All on the old notebook. "
                       "Download its Part 1 Results ZIP from its Output panel and upload it for recovery.")
marker = ROOT / ".lab2_runtime_cuda.json"
if not marker.is_file():
    raise RuntimeError("Original isolated GPU interpreter marker is missing. Existing results were not changed.")
PYTHON = Path(json.loads(marker.read_text())["python"])
if not PYTHON.is_file():
    raise RuntimeError("The original Python environment is no longer available. No generation was started.")
PATCHES = PAYLOAD
PATCH_DIR = ROOT / "runtime_checks/evaluation_recovery_v1"
PATCH_DIR.mkdir(parents=True, exist_ok=True)
for name, (source, digest) in PATCHES.items():
    data = source.encode()
    assert hashlib.sha256(data).hexdigest() == digest
    path = PATCH_DIR / name
    if path.exists() and path.read_bytes() != data:
        raise RuntimeError("A different recovery patch already exists; it was preserved.")
    path.write_bytes(data)
print("Saved CSV:", ROOT.parent / "submission_part1_v5.csv")
print("Only evaluation will resume. Competition predictions and original source files stay unchanged.")
'''.replace('PAYLOAD', repr(payload))
    run = '''log = ROOT / "runtime_checks/evaluation_recovery_v1.log"
command = [str(PYTHON), "-u", str(PATCH_DIR / "recover_part1_evaluation.py"), "--root", str(ROOT)]
with log.open("a", encoding="utf-8") as stream:
    process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, encoding="utf-8", errors="replace")
    for line in process.stdout:
        print(line, end="", flush=True)
        stream.write(line); stream.flush()
    status = process.wait()
if status:
    raise RuntimeError("Evaluation recovery stopped. Read the actual error immediately above. "
                       "The existing CSV is still available; log: " + str(log))
'''
    view = '''path = ROOT / "part1_rag/outputs/v2/evaluation/evaluation.json"
if path.is_file():
    from IPython.display import display
    import pandas as pd
    metrics = json.loads(path.read_text())
    display(pd.DataFrame(metrics["results"]))
    print(metrics["limitations"])
else:
    print("Evaluation is still incomplete. No quality scores were invented.")
record = ROOT / "runtime_checks/part1_evaluation_recovery.json"
if record.is_file():
    print(record.read_text())
print("Download submission_part1_v5.csv and the latest Lab2_Part1_Results ZIP from Output > /kaggle/working.")
print("Keep the original executed notebook and this recovery notebook with outputs. Neither submits to Kaggle/Canvas.")
'''
    md, code = nb.v4.new_markdown_cell, nb.v4.new_code_cell
    notebook = nb.v4.new_notebook(cells=[
        md('''# Part 1: recover the saved v5 evaluation

Use this only in the **same Kaggle session** that still contains the saved v5 folder. Keep that session open. This file is self-contained: no separate Python uploads, token or repository clone is needed. If importing a notebook, replace the source in the current editor; do not start a new background session with Save & Run All. If the old session is gone, retain its exported Results ZIP instead. Missing original files cause a safe stop, not automatic regeneration.

The supplied log says all 40 competition answers were generated and the validated `submission_part1_v5.csv` was published. The separate citation judge failed because its unconstrained response omitted/renamed required citation keys. This recovery constrains that fixed-key judging task from its first call; both supported=true and supported=false remain model decisions. It does not fabricate a judgment or mark missing evaluation complete.

Original answering files and the CSV are not modified. Genuine evaluation answer checkpoints resume on the same verified runtime; remaining golden-set items still require GPU inference. The replacement judge and recovery timings are recorded separately. Original failed notebook outputs are preserved, not cleared or relabeled as a clean run.

The supplied run took 5,828.5 seconds before stopping, about 97.1 minutes. It already exceeded the 90-minute limit. Recovery preserves that fact and cannot make the original run timing-compliant. Automatic citation judgments are not human ratings or a leaderboard score. Real team review, ratings, report, permissions, hardware/timing checks and external submission remain separate.'''),
        code(setup), md('## Resume only the unfinished evaluation'), code(run),
        md('## Check measured results and download the evidence'), code(view)], metadata={
            'kernelspec': {'name':'python3','display_name':'Python 3','language':'python'},
            'language_info': {'name':'python'},
            'validation_scope':'CPU regression fixtures are not a completed pretrained GPU evaluation.'})
    for cell in notebook.cells:
        if cell.cell_type == 'code': compile(cell.source, 'recovery-cell', 'exec')
    nb.validate(notebook)
    path = ROOT / 'part1_rag/notebooks/Lab2_Part1_Evaluation_Recovery.ipynb'
    nb.write(notebook, path)
    print('Created', path, path.stat().st_size, 'bytes')
    return path


if __name__ == '__main__':
    build()
