"""Build a self-contained Part 1 Kaggle notebook without private Part 2 inputs."""
import ast
import base64
import hashlib
import io
import json
from pathlib import Path
import subprocess
import textwrap
import zipfile

import nbformat as nb

ROOT = Path(__file__).resolve().parents[1]


def experiment():
    md, code = nb.v4.new_markdown_cell, nb.v4.new_code_cell
    return nb.v4.new_notebook(cells=[
        md('''# Part 1: Multimodal economics RAG

This run compares dense retrieval with hybrid dense/BM25/CLIP retrieval followed by a pinned cross-encoder. Explicit figure labels are matched exactly. The PDFs are read-only and low-trust Appendix C is quarantined. Nothing is trained on competition answers. Candidate references and real human ratings still need team review.'''),
        code('''from pathlib import Path
import os, sys, json, csv, time, subprocess, hashlib, shutil
ROOT = Path(os.environ["LAB2_STANDALONE_ROOT"])
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "part1_rag/src")]
from part1_kaggle import find_questions
from run_checkpoint import atomic_json, runtime_identity
SMOKE_ONLY = os.environ.get("LAB2_SMOKE_ONLY") == "1"
DATA = ROOT / "part1_rag/data"
P1 = ROOT / "part1_rag"
OUTPUT = P1 / "outputs/v2"
OUTPUT.mkdir(parents=True, exist_ok=True)
QUESTIONS = find_questions(DATA)
shutil.copy2(QUESTIONS, OUTPUT / "questions_used.csv")
atomic_json(OUTPUT / "run_inputs.json", dict(questions_path=str(QUESTIONS),
    questions_sha256=hashlib.sha256(QUESTIONS.read_bytes()).hexdigest(), hardware=runtime_identity(),
    seed=266, smoke_only=SMOKE_ONLY, protocol="hybrid-rerank-v1"))
print("Actual hardware and inputs:", (OUTPUT / "run_inputs.json").read_text())

def run_script(relative, *arguments):
    log = ROOT / "runtime_checks" / (Path(relative).stem + ".log")
    log.parent.mkdir(exist_ok=True)
    command = [sys.executable, "-u", str(ROOT / relative), *map(str, arguments)]
    with log.open("a", encoding="utf-8") as handle:
        process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace")
        for line in process.stdout:
            print(line, end="", flush=True)
            handle.write(line); handle.flush()
        status = process.wait()
    if status:
        raise RuntimeError(f"{Path(relative).name} failed. Diagnostic log: {log}")'''),
        md('''## Check data handling and structured decoding

These tests use fictional evidence and tiny random weights where needed. Passing them is not a model-quality score. They check that a new chart/year does not inherit an old one, malformed JSON stays an error, and no unsupported citation is accepted.'''),
        code('''run_script("tools/test_part1_retrieval.py")
run_script("tools/test_constrained_decoding.py", "SchemaTests", "TinyActualVisionGenerationTest", "CachedQwenTokenizerTests")
run_script("part1_rag/src/smoke_test.py")'''),
        md('''## Exercise the actual GPU path

The preflight checks a real chart request with the actual VLM. It also loads the pinned reranker before the longer stages. The dense baseline does not use BM25, CLIP fusion, exact-label promotion or reranking.'''),
        code('''if SMOKE_ONLY:
    print("SKIPPED: real pretrained-model GPU preflight. CPU smoke is not quality evidence.")
else:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Select a GPU accelerator before running Part 1.")
    from multimodal import MultimodalIndex, Conversation
    index = MultimodalIndex(DATA, OUTPUT / "index")
    chat = Conversation(index)
    try:
        print("Trusted figure/table objects:", len(index.objects))
        print("Quarantined objects:", len(index.ingestion["quarantined_objects"]))
        print("GPU PREFLIGHT:", chat.answer("Read Figure A-5. Identify its stated base date if readable; do not guess."))
    finally:
        chat.model.unload()
        index.db.close()
    del chat, index
    print("REAL PART 1 GPU PREFLIGHT PASSED")'''),
        md('''## Generate and validate the competition CSV

The question file is never rewritten or reordered. Each conversation has separate state. An attached Questions.csv takes precedence over the embedded practice file; QUESTIONS_PATH can explicitly select the hidden question file. Answers and per-item checkpoints are saved as they complete.'''),
        code('''if SMOKE_ONLY:
    print("SKIPPED: model-generated competition answers. No mock submission was created.")
else:
    run_script("part1_rag/src/generate_submission.py", "--data-dir", DATA,
        "--questions-path", QUESTIONS, "--output", OUTPUT / "submission.csv", "--require-dense")
    run_script("part1_rag/data/validate_submission.py", OUTPUT / "submission.csv", QUESTIONS)
    with (OUTPUT / "submission.csv").open(newline="", encoding="utf-8") as handle:
        answers = list(csv.DictReader(handle))
    print("Example model answers:", json.dumps(answers[:3], indent=2, ensure_ascii=False))
    print("Rows validated:", len(answers))'''),
        md('''## Measure baseline and final system on the separate 25-question set

These references were prepared separately from Questions.csv. Neither retrieval nor the answering model receives the reference answers. Correctness is an automatic numeric/status proxy, citation accuracy is model-judged, and human groundedness remains blank until both teammates supply real ratings.'''),
        code('''if SMOKE_ONLY:
    print("SKIPPED: baseline/final model evaluation and human ratings.")
else:
    run_script("part1_rag/src/evaluate_rag.py", "--data-dir", DATA,
        "--output", OUTPUT / "evaluation", "--golden", P1 / "eval/golden_candidate.json")
    metrics = json.loads((OUTPUT / "evaluation/evaluation.json").read_text())
    import pandas as pd
    display(pd.DataFrame(metrics["results"]))
    print(metrics["limitations"])
    print("Blank rating forms:", OUTPUT / "evaluation")'''),
        md('''## What to check before submitting

Compare correct answers, Recall@5, citation support and median latency in the table above. Read low-scoring answers against the actual PDF page rather than changing the references to make a score look better. Two teammates still need to review the candidate references and rate groundedness. The separate report, team details, final tag and competition upload are not completed by running this notebook.

The outer launcher saves this executed notebook even if a cell fails and exports all available Part 1 evidence. A CPU smoke is always marked as such.''')], metadata={
            'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'},
            'language_info': {'name': 'python'}})


def build():
    from build_colab_standalone import SETUP
    from standalone_colab import unpack
    import inspect
    names = ['requirements.txt', 'tools/setup_runtime.py', 'tools/preflight.py', 'tools/monitored_run.py',
        'tools/run_checkpoint.py', 'tools/vlm_runtime.py', 'tools/response_schemas.py', 'tools/review_measurements.py',
        'tools/standalone_colab.py', 'tools/overnight_runner.py', 'tools/export_lab_results.py',
        'tools/kaggle_runtime.py', 'tools/part1_kaggle.py', 'tools/test_part1_retrieval.py',
        'tools/test_constrained_decoding.py', 'part1_rag/eval/golden_candidate.json']
    files = {name: (ROOT / name).read_bytes() for name in names}
    for folder in ['part1_rag/src', 'part1_rag/data']:
        for path in (ROOT / folder).iterdir():
            if path.is_file() and path.suffix in {'.py', '.pdf', '.csv', '.json'}:
                files[path.relative_to(ROOT).as_posix()] = path.read_bytes()
    files['part1_rag/notebooks/Part1_Kaggle_Experiment.ipynb'] = nb.writes(experiment()).encode()
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip())
    manifest = dict(source_commit=commit, source_dirty_at_build=dirty, scope='Part 1 only', resources=[],
        files={name: hashlib.sha256(data).hexdigest() for name, data in files.items()})
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as bundle:
        for name, data in sorted(files.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            bundle.writestr(info, data)
        info = zipfile.ZipInfo('STANDALONE_MANIFEST.json', date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        bundle.writestr(info, json.dumps(manifest, sort_keys=True, indent=2))
    raw = payload.getvalue()
    digest = hashlib.sha256(raw).hexdigest()
    bootstrap = '''from pathlib import Path, PurePosixPath
import os, sys, subprocess, json, hashlib, base64, io, stat, zipfile
SMOKE_ONLY = os.environ.get("LAB2_SMOKE_ONLY", "0") == "1"
'''
    bootstrap += (ROOT / 'tools/kaggle_runtime.py').read_text() + '\n'
    bootstrap += 'print("Runtime setup:", configure_runtime(smoke_only=SMOKE_ONLY, require_private_artifacts=False))\n'
    bootstrap += 'os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"\n'
    bootstrap += 'os.environ.setdefault("OMP_NUM_THREADS", "2")\nos.environ.setdefault("MKL_NUM_THREADS", "2")\n'
    bootstrap += '''if not SMOKE_ONLY:
    try:
        print(subprocess.check_output(["nvidia-smi", "-L"], text=True, timeout=15))
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError("Select a GPU in Kaggle Session options before running. Setup and model work have not started.") from error
'''
    bootstrap += 'BUNDLE_SHA256 = ' + repr(digest) + '\nBUNDLE_BASE64 = (\n'
    bootstrap += '\n'.join('    ' + repr(s) for s in textwrap.wrap(base64.b64encode(raw).decode(), 120)) + '\n)\n'
    bootstrap += inspect.getsource(unpack) + '\n'
    bootstrap += '''parent = Path(os.environ.get("LAB2_STANDALONE_PARENT", "/kaggle/working"))
ROOT = unpack(base64.b64decode(BUNDLE_BASE64), BUNDLE_SHA256, parent)
os.environ["LAB2_STANDALONE_ROOT"] = str(ROOT)
print("Verified source and course data:", BUNDLE_SHA256)
'''
    bootstrap += SETUP
    md, code = nb.v4.new_markdown_cell, nb.v4.new_code_cell
    notebook = nb.v4.new_notebook(cells=[
        md('''# Lab 2 Part 1: Final Kaggle run

This notebook is self-contained. Import this one file into Kaggle and run it with a GPU accelerator and Internet enabled. It embeds the source and course PDFs, creates its helper files automatically and uses an isolated pinned Python environment. No repository clone, extra Python upload or Hugging Face token is required for Part 1's public models.

The revised system adds cross-encoder reranking and cleaner follow-up retrieval. It preserves exact chart labels, source citations, read-only documents and conversation-local assumptions. A higher leaderboard score is not guaranteed; use the real measured results after this run. No competition reference answers or question-specific lookup rules are used by the answering pipeline.

Only Part 1 runs here. Existing Part 2 results and Part 3 are not changed or rerun. On T4 x2 it uses one GPU, not pooled VRAM. The course requires the Part 1 inference notebook to finish within 90 minutes; elapsed time is recorded below, but download speed, hardware and quotas can affect it.

Attach the competition data if available. An attached Questions.csv is selected automatically; the embedded practice file is the fallback. Keep the notebook private. For a background run, use Save Version > Save & Run All. A new session cannot recover unsaved files from an old session automatically.'''),
        code(bootstrap),
        md('''## Run and keep the executed experiment

The isolated child notebook saves after every cell, so its outputs go into the result ZIP. Validation and checkpoint tests run before model generation. Errors are retained and partial results are exported automatically when possible. Unchanged completed answers can resume while this exact run folder still exists.'''),
        code('''run_script("tools/part1_kaggle.py", "run", *(["--smoke-only"] if SMOKE_ONLY else []))
print((ROOT / "runtime_checks/part1_completion.json").read_text())'''),
        md('''## View measured results

No placeholder quality numbers are shown. If running a CPU smoke, model inference and quality measurements are skipped.'''),
        code('''if SMOKE_ONLY:
    print("CPU SMOKE ONLY. No leaderboard CSV or model-quality scores were created.")
else:
    from IPython.display import display, HTML
    import html
    metrics = json.loads((ROOT / "part1_rag/outputs/v2/evaluation/evaluation.json").read_text())
    columns = ["system", "correct_out_of_25", "retrieval_recall_at_5", "model_judged_citation_accuracy", "median_seconds_per_question"]
    header = "".join("<th>" + html.escape(key) + "</th>" for key in columns)
    rows = "".join("<tr>" + "".join("<td>" + html.escape(str(row.get(key))) + "</td>" for key in columns) + "</tr>" for row in metrics["results"])
    display(HTML("<table><tr>" + header + "</tr>" + rows + "</table>"))
    print(metrics["limitations"])
    completion = json.loads((ROOT / "runtime_checks/part1_completion.json").read_text())
    print("Experiment elapsed minutes:", round(completion["seconds"] / 60, 2))
    print("Within 90-minute experiment budget:", completion["seconds"] <= 5400)
    print("This timing excludes the first setup/install cell. Save Version's runtime is the whole-run timing.")'''),
        md('''## Download the CSV, executed notebook and evidence

In Kaggle's Output panel download submission.csv (or its bundle-suffixed name if an older CSV was preserved), the Part 1 Results ZIP, and this notebook with its outputs. The ZIP includes the executed experiment, answer traces, evaluation, blank human-rating forms and diagnostic logs. It does not contain pretrained model caches or secrets. Running this cell never repeats model inference.

The completed run still needs real human ratings, team reference review, report/team records, a final tag and the actual Kaggle/Canvas submission. A green computation check does not stand in for those items.'''),
        code('''from pathlib import Path
import os, sys, subprocess, json
if "ROOT" not in globals():
    parent = Path(os.environ.get("LAB2_STANDALONE_PARENT", "/kaggle/working"))
    ROOT = parent / "Lab2_Standalone_BUNDLE_PREFIX"
    if not (ROOT / "tools/part1_kaggle.py").is_file():
        raise FileNotFoundError("This session has no matching saved run. Open the saved version's Output panel; do not restart model work just to export.")
subprocess.run([sys.executable, str(ROOT / "tools/part1_kaggle.py"), "export"], cwd=ROOT, check=True)
print("Download the Lab2_Part1_Results ZIP from Output > /kaggle/working.")'''.replace('BUNDLE_PREFIX', digest[:16]))], metadata={
            'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'},
            'language_info': {'name': 'python'}, 'kaggle': {'isInternetEnabled': True, 'isGpuEnabled': True},
            'accelerator': 'GPU', 'standalone_bundle_sha256': digest,
            'validation_scope': 'Local smoke and regression checks are not a completed Kaggle GPU quality evaluation.'})
    notebook.cells[1].metadata = {'jupyter': {'source_hidden': True}, 'tags': ['embedded-source-bundle']}
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            compile(cell.source, 'part1-kaggle-cell', 'exec')
    nb.validate(notebook)
    path = ROOT / 'part1_rag/notebooks/Lab2_Part1_Kaggle_Final.ipynb'
    nb.write(notebook, path)
    print('Created', path, 'embedded files:', len(files), 'bytes:', path.stat().st_size)
    return path


if __name__ == '__main__':
    build()
