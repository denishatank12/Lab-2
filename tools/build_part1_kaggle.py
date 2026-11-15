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


def experiment(evidence_first=False):
    md, code = nb.v4.new_markdown_cell, nb.v4.new_code_cell
    notebook = nb.v4.new_notebook(cells=[
        md('''# Part 1: Multimodal economics RAG

This run compares dense retrieval with hybrid dense/BM25/CLIP retrieval followed by a pinned cross-encoder. Both use the same Qwen2-VL-7B model in NF4, with the visual encoder kept in float16. Tables keep their row/column structure and explicit revision notices stay linked to the older object. The PDFs are read-only and low-trust Appendix C is quarantined. Nothing is trained on competition answers. The extra optional draft editor is disabled in this profile; it is not a required assignment component and repeatedly failed in the earlier run. Required citation evaluation remains enabled. Candidate references and real human ratings still need team review.'''),
        code('''from pathlib import Path
import os, sys, json, csv, time, subprocess, hashlib, shutil
ROOT = Path(os.environ["LAB2_STANDALONE_ROOT"])
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "part1_rag/src")]
from part1_kaggle import find_questions, publish_submission
from run_checkpoint import atomic_json, runtime_identity
from rag_model import MODEL, REVISION
SMOKE_ONLY = os.environ.get("LAB2_SMOKE_ONLY") == "1"
os.environ["LAB2_RAG_PROFILE"] = "7b-nf4"
os.environ["LAB2_RAG_NATURAL_JSON"] = "1"
os.environ["LAB2_RAG_EVIDENCE_FIRST"] = "0"
os.environ["LAB2_RAG_EVIDENCE_REVIEW"] = "0"
os.environ["LAB2_RAG_REVIEW_REQUIRED"] = "0"
DATA = ROOT / "part1_rag/data"
P1 = ROOT / "part1_rag"
OUTPUT = P1 / "outputs/v2"
OUTPUT.mkdir(parents=True, exist_ok=True)
QUESTIONS = find_questions(DATA)
shutil.copy2(QUESTIONS, OUTPUT / "questions_used.csv")
atomic_json(OUTPUT / "run_inputs.json", dict(questions_path=str(QUESTIONS),
    questions_sha256=hashlib.sha256(QUESTIONS.read_bytes()).hexdigest(), hardware=runtime_identity(),
    seed=266, smoke_only=SMOKE_ONLY, protocol="independent-fresh-7b-v6", evidence_review=False,
    answer_policy="natural JSON first; invalid responses retry with evidence-bounded constrained decoding",
    review_policy="optional draft editor disabled; required exact-key citation evaluation retained",
    answering_model=MODEL, model_revision=REVISION, language_weights="NF4", vision_weights="float16"))
print("Actual hardware and inputs:", (OUTPUT / "run_inputs.json").read_text())

def run_script(relative, *arguments):
    log = ROOT / "runtime_checks" / (Path(relative).stem + ".log")
    log.parent.mkdir(exist_ok=True)
    command = [sys.executable, "-u", str(ROOT / relative), *map(str, arguments)]
    process_env = os.environ.copy()
    if relative.startswith("tools/test_"):
        # Unit fixtures explicitly opt into the reviewer where it is tested.
        # They must not inherit a real-generation option from the launcher.
        process_env["LAB2_RAG_EVIDENCE_REVIEW"] = "0"
        process_env["LAB2_RAG_REVIEW_REQUIRED"] = "1"
        process_env["LAB2_RAG_EVIDENCE_FIRST"] = "0"
    with log.open("a", encoding="utf-8") as handle:
        process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", env=process_env)
        for line in process.stdout:
            print(line, end="", flush=True)
            handle.write(line); handle.flush()
        status = process.wait()
    if status:
        raise RuntimeError(f"{Path(relative).name} failed. Diagnostic log: {log}")'''),
        md('''## Check data handling and structured decoding

These tests use fictional evidence and tiny random weights where needed. Passing them is not a model-quality score. They check that a new chart/year does not inherit an old one, malformed JSON stays an error, and no unsupported citation is accepted.'''),
        code('''run_script("tools/test_part1_retrieval.py")
run_script("tools/test_part1_quality.py")
run_script("tools/test_evidence_review.py")
run_script("tools/test_citation_judge.py")
run_script("tools/test_part1_pipeline.py")
run_script("tools/test_constrained_decoding.py", "SchemaTests", "TinyActualVisionGenerationTest", "CachedQwenTokenizerTests")
run_script("part1_rag/src/smoke_test.py")'''),
        md('''## Exercise the actual GPU path

The preflight checks a real chart request and the citation judge with the actual VLM before the longer stages. Its short source-quote probe is only an interface test, not a competition answer or quality score. It also loads the pinned reranker. The dense baseline does not use BM25, CLIP fusion, exact-label promotion or reranking.'''),
        code('''if SMOKE_ONLY:
    print("SKIPPED: real pretrained-model GPU preflight. CPU smoke is not quality evidence.")
else:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Select a GPU accelerator before running Part 1.")
    from multimodal import MultimodalIndex, Conversation
    from citation_judge import judge_citations
    index = MultimodalIndex(DATA, OUTPUT / "index")
    chat = Conversation(index)
    try:
        print("Trusted figure/table objects:", len(index.objects))
        print("Quarantined objects:", len(index.ingestion["quarantined_objects"]))
        print("GPU PREFLIGHT:", chat.answer("Read Figure A-5. Identify its stated base date if readable; do not guess."))
        source = next(r for r in index.ingestion["passages"] if r["text"].strip())
        probe = dict(answer=source["text"][:240], citations=[source["citation"]])
        checks = judge_citations(chat.model, probe, [source])
        atomic_json(OUTPUT / "gpu_preflight.json", dict(
            purpose="GPU interface checks only, not quality evaluation", runtime=runtime_identity(),
            model=chat.model.profile, citation_judge_checked=True, judgments=checks))
        print("REAL GPU CITATION-JUDGE PREFLIGHT PASSED:", checks)
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
    publish_submission(ROOT)
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
    if evidence_first:
        for cell in notebook.cells:
            cell.source = cell.source.replace('independent-fresh-7b-v6', 'independent-evidence-first-7b-v7')
            if cell.cell_type == 'code' and 'LAB2_RAG_PROFILE' in cell.source:
                cell.source = cell.source.replace('os.environ["LAB2_RAG_EVIDENCE_FIRST"] = "0"',
                                                  'os.environ["LAB2_RAG_EVIDENCE_FIRST"] = "1"', 1)
            if cell.cell_type == 'code' and 'run_script("tools/test_part1_retrieval.py")' in cell.source:
                cell.source += '\nrun_script("tools/test_evidence_answer.py")'
            if cell.cell_type == 'code' and 'protocol="independent-evidence-first-7b-v7"' in cell.source:
                cell.source = cell.source.replace('evidence_review=False,', 'evidence_review=False, evidence_first=True,')
                cell.source = cell.source.replace(
                    'natural JSON first; invalid responses retry with evidence-bounded constrained decoding',
                    'one evidence-first JSON response with model source readings before answer/status; bounded structured retries')
        notebook.cells[0].source += '''

This v7 revision selects short model source observations before answering, removes
earlier assistant answers from the evidence prompt, and derives missing figure
labels only from exact object-citation metadata. User assumptions are passed in
chronological order, including an explicit assumption in the current turn. The
observations are not independently verified facts or a human rating. This is one
answering pass, not the earlier slow extra editor. Quality and the 90-minute
budget must be measured on the new GPU run; a 0.8 score is a target, not a promise.'''
    return notebook


def build(evidence_first=False):
    from build_colab_standalone import SETUP
    from standalone_colab import unpack
    import inspect
    names = ['requirements.txt', 'tools/setup_runtime.py', 'tools/preflight.py', 'tools/monitored_run.py',
        'tools/run_checkpoint.py', 'tools/vlm_runtime.py', 'tools/response_schemas.py', 'tools/review_measurements.py',
        'tools/standalone_colab.py', 'tools/overnight_runner.py', 'tools/export_lab_results.py',
        'tools/kaggle_runtime.py', 'tools/part1_kaggle.py', 'tools/test_part1_retrieval.py',
        'tools/test_constrained_decoding.py', 'tools/rag_model.py', 'tools/test_part1_quality.py',
        'tools/test_evidence_review.py', 'tools/part1_diagnostics.py',
        'tools/citation_judge.py', 'tools/test_citation_judge.py', 'tools/test_part1_pipeline.py',
        'tools/test_evidence_answer.py',
        'part1_rag/eval/golden_candidate.json']
    files = {name: (ROOT / name).read_bytes() for name in names}
    files['requirements.txt'] += b'\nbitsandbytes==0.44.1\n'
    resources = []
    # PDFs are data, not source. Verified downloads keep the actual notebook
    # below Kaggle's 1-MB source limit without changing a single course value.
    data_commit = 'c889354f3df2358157e725d1b3d11d741f24ef3b'
    for folder in ['part1_rag/src', 'part1_rag/data']:
        for path in (ROOT / folder).iterdir():
            if path.is_file() and path.suffix in {'.py', '.pdf', '.csv', '.json'}:
                name = path.relative_to(ROOT).as_posix()
                if path.suffix == '.pdf':
                    resources.append(dict(path=name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                        url=f'https://raw.githubusercontent.com/denishatank12/Lab-2/{data_commit}/{name}'))
                else:
                    files[name] = path.read_bytes()
    files['part1_rag/notebooks/Part1_Kaggle_Experiment.ipynb'] = nb.writes(experiment(evidence_first)).encode()
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip())
    manifest = dict(source_commit=commit, source_dirty_at_build=dirty, scope='Part 1 only', resources=resources,
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
import os, sys, subprocess, json, hashlib, base64, io, stat, zipfile, time
NOTEBOOK_STARTED = time.perf_counter()
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
print("Verified embedded source:", BUNDLE_SHA256)
'''
    bootstrap += SETUP
    bootstrap += '''\nrun_script("tools/standalone_colab.py", "resources")
os.environ["LAB2_RAG_PROFILE"] = "7b-nf4"
os.environ["LAB2_RAG_NATURAL_JSON"] = "1"
os.environ["LAB2_RAG_EVIDENCE_FIRST"] = "0"
os.environ["LAB2_RAG_EVIDENCE_REVIEW"] = "0"
os.environ["LAB2_RAG_REVIEW_REQUIRED"] = "0"
'''
    md, code = nb.v4.new_markdown_cell, nb.v4.new_code_cell
    notebook = nb.v4.new_notebook(cells=[
        md('''# Lab 2 Part 1: independent Kaggle run (v6)

Import this one file into Kaggle and run it with a nominal 16-GB GPU and Internet enabled. All executable source is embedded; course PDFs download from pinned URLs with SHA-256 checks. The notebook creates its helper files automatically and uses an isolated pinned Python environment. No repository clone, extra Python upload or Hugging Face token is required for the public models. Its source is checked to stay below Kaggle's 1-MB saving limit.

Start in a new Kaggle notebook with a GPU and Internet enabled, then run all cells in order. No saved run folder, recovery notebook, previous ZIP, separate Python file, repository checkout or Hugging Face token is needed. The first cell unpacks its own verified executable source, creates its isolated Python environment and downloads the three hash-checked course PDFs. On a fresh session it creates everything it needs automatically. Keep course work and saved outputs private.

The final system uses hybrid dense/BM25/CLIP retrieval, reranking, table grids, exact chart references, source trust and per-conversation memory. It compares against a dense-only baseline on a separate 25-question candidate set. The model never receives the candidate reference answers. Both systems use the same pinned Qwen2-VL-7B NF4 model with float16 vision on one nominal 16-GB GPU. T4 x2 is supported, but only one card is used; VRAM is not pooled.

Answers begin with natural JSON; malformed answers retry with evidence-bounded constrained decoding. Only supplied citation/figure labels are allowed, and an empty figure evidence list requires an empty figure-reference list. The citation judge always uses exact source keys and real boolean values, even when answer generation uses natural JSON. GPU preflight exercises both chart answering and citation judging before the long stages. Required validation is not bypassed and missing scores are not invented.

The earlier optional draft editor is disabled. It was not required by the rubric, repeatedly failed its own checks and consumed extra model calls. Required citation evaluation and the baseline comparison are still performed. This changes the measured profile, so old answer checkpoints are not reused or claimed as new results. No particular leaderboard score or runtime is guaranteed. The 90-minute requirement is checked against the actual elapsed time, including setup, rather than asserted in advance.

Attach the competition data if available. Its Questions.csv is selected automatically; otherwise the embedded official practice file is used. Questions are not rewritten or reordered. Complete predictions are validated and published as submission_part1_v6.csv before the separate golden evaluation. Results and per-item checkpoints are saved as they complete; diagnostics and a ZIP preserve available evidence if a later stage fails. On a new session, unsaved files from an old session are not needed or recovered.

Only Part 1 runs. Part 2 training and Part 3 are not repeated. A successful CPU smoke checks packaging/control flow and tiny model interfaces, not full pretrained CUDA execution or answer quality. Real groundedness ratings, team reference review, report/team records, a tag and external Kaggle/Canvas submission remain separate.

For a background run, use Save Version > Save & Run All after setting the accelerator and Internet. Download the new CSV, Results ZIP and executed notebook from Kaggle's Output panel when finished.'''),
        code(bootstrap),
        md('''## Run and keep the executed experiment

The isolated child notebook saves after every cell, so its outputs go into the result ZIP. Validation and checkpoint tests run before model generation. Errors are retained and partial results are exported automatically when possible. Unchanged completed answers can resume while this exact run folder still exists.'''),
        code('''try:
    run_script("tools/part1_kaggle.py", "run", *(["--smoke-only"] if SMOKE_ONLY else []))
except Exception:
    import importlib.util
    spec = importlib.util.spec_from_file_location("part1_diagnostics", ROOT / "tools/part1_diagnostics.py")
    diagnostics = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(diagnostics)
    print("\\nREAD-ONLY FAILURE DIAGNOSTICS:\\n" + diagnostics.failure_details(ROOT))
    print("Keep this session open. The final export cell can be run independently; it does not repeat inference.")
    raise
finally:
    elapsed = time.perf_counter() - NOTEBOOK_STARTED
    timing = dict(scope="Current notebook invocation including setup and experiment; not a sum of prior sessions",
                  elapsed_seconds=elapsed, smoke_only=SMOKE_ONLY,
                  within_90_minutes=(elapsed <= 5400) if not SMOKE_ONLY else None)
    (ROOT / "runtime_checks/notebook_wall_time.json").write_text(json.dumps(timing, indent=2))
    print("NOTEBOOK WALL TIME:", timing)
print((ROOT / "runtime_checks/part1_completion.json").read_text())'''),
        md('''## View measured results

No placeholder quality numbers are shown. If running a CPU smoke, model inference and quality measurements are skipped.'''),
        code('''if SMOKE_ONLY:
    print("CPU SMOKE ONLY. No leaderboard CSV or model-quality scores were created.")
elif not (ROOT / "part1_rag/outputs/v2/evaluation/evaluation.json").is_file():
    print("Evaluation is incomplete. No evaluation scores are available yet; no placeholder metrics were created.")
    print("Keep the saved CSV and Results ZIP. Inspect the failed stage above rather than running this display cell again.")
else:
    from IPython.display import display, HTML
    import html
    metrics = json.loads((ROOT / "part1_rag/outputs/v2/evaluation/evaluation.json").read_text())
    columns = ["system", "correct_out_of_25", "retrieval_recall_at_5", "model_judged_citation_accuracy", "median_seconds_per_question", "optional_reviews_completed", "optional_reviews_failed"]
    header = "".join("<th>" + html.escape(key) + "</th>" for key in columns)
    rows = "".join("<tr>" + "".join("<td>" + html.escape(str(row.get(key))) + "</td>" for key in columns) + "</tr>" for row in metrics["results"])
    display(HTML("<table><tr>" + header + "</tr>" + rows + "</table>"))
    print(metrics["limitations"])
    completion = json.loads((ROOT / "runtime_checks/part1_completion.json").read_text())
    print("Experiment elapsed minutes:", round(completion["seconds"] / 60, 2))
    print("Upload THIS new CSV, not the previous 0.419 file:", completion.get("competition_csv"))
    timing = json.loads((ROOT / "runtime_checks/notebook_wall_time.json").read_text())
    print("Notebook elapsed minutes, including setup:", round(timing["elapsed_seconds"] / 60, 2))
    print("Within 90-minute notebook budget:", timing["within_90_minutes"])
    print("Save Version's full runtime also includes the final display/export cells.")'''),
        md('''## Download the CSV, executed notebook and evidence

In Kaggle's Output panel download submission_part1_v6.csv (or its bundle-suffixed name if an older CSV was preserved), the Part 1 Results ZIP, and this notebook with its outputs. The exact new CSV path prints above. The ZIP includes the executed experiment, answer traces, evaluation, blank human-rating forms, timing and diagnostic logs. It does not contain pretrained model caches or secrets. Running this cell never repeats model inference.

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
    if evidence_first:
        for cell in notebook.cells:
            cell.source = cell.source.replace('submission_part1_v6', 'submission_part1_v7')
            cell.source = cell.source.replace('independent Kaggle run (v6)', 'evidence-first Kaggle run (v7)')
        notebook.cells[1].source = notebook.cells[1].source.replace(
            'os.environ["LAB2_RAG_EVIDENCE_FIRST"] = "0"',
            'os.environ["LAB2_RAG_EVIDENCE_FIRST"] = "1"')
        notebook.cells[0].source += '''

### What changed from the completed v6 run

The model now reads short source observations before writing its answer and
choosing a status. Its prompt contains only earlier user turns, not old assistant
claims that could be wrong. Table rows retain their headers without duplicating
the entire numeric matrix. Exact cited object metadata supplies matching figure
labels. Assumptions remain chat-only, and the original PDFs are never edited.

This notebook does not contain the previous run's answers or a guaranteed score.
It does not read candidate reference answers during generation. The software
checks cannot establish factual correctness: inspect the measured evaluation and
new Kaggle score after running. v6 remains a separate preserved notebook. This
revision still requires the full GPU run; CPU smoke results are not GPU results.'''
    notebook.cells[1].metadata = {'jupyter': {'source_hidden': True}, 'tags': ['embedded-source-bundle']}
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            compile(cell.source, 'part1-kaggle-cell', 'exec')
    nb.validate(notebook)
    filename = 'Lab2_Part1_Kaggle_v7.ipynb' if evidence_first else 'Lab2_Part1_Kaggle_Final.ipynb'
    path = ROOT / 'part1_rag/notebooks' / filename
    nb.write(notebook, path)
    if path.stat().st_size >= 950_000:
        raise ValueError('Notebook source exceeds the safety margin for Kaggle saving. Do not publish this artifact.')
    print('Created', path, 'embedded files:', len(files), 'bytes:', path.stat().st_size)
    return path


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence-first', action='store_true', help='Build a separate v7; preserve the completed v6 artifact.')
    build(evidence_first=parser.parse_args().evidence_first)
