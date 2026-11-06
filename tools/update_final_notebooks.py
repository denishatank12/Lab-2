"""Build the revised run notebooks without overwriting the completed Part 2 experiment."""
import ast
from datetime import datetime, timezone
from pathlib import Path
import shutil

import nbformat as nb

ROOT=Path(__file__).resolve().parents[1]
tree=ast.parse((ROOT/'tools/rebuild_notebooks.py').read_text())
BOOTSTRAP=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
               and any(isinstance(t,ast.Name) and t.id=='BOOTSTRAP' for t in n.targets))
BOOTSTRAP=BOOTSTRAP.replace('os.environ.get("LAB2_DEVICE", "auto")','os.environ.get("LAB2_DEVICE", "cuda")')
BOOTSTRAP+='''
# Monitor actual device memory in each subprocess. A larger GPU is not relabeled as a T4.
_run_stage = run_script
def run_script(relative, *arguments):
    record = ROOT / "runtime_checks" / (Path(relative).stem + "_" + str(time.time_ns()) + ".memory.json")
    _run_stage("tools/monitored_run.py", "--record", record, "--", PYTHON, ROOT / relative, *arguments)
'''


def md(text): return nb.v4.new_markdown_cell(text)
def code(text): return nb.v4.new_code_cell(text)


def save(part,name,cells):
    path=ROOT/part/'notebooks'/name
    if path.exists():
        backup=ROOT/part/'outputs/previous_notebooks'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        backup.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,backup/name)
    notebook=nb.v4.new_notebook(cells=cells)
    notebook.metadata={'kernelspec':{'name':'python3','language':'python','display_name':'Python 3'},
                       'language_info':{'name':'python'},'validation_scope':'Revised implementation; needs a fresh full GPU run. CPU smoke outputs are not evaluation results.'}
    nb.validate(notebook)
    nb.write(notebook,path)


def main():
    common='''This is the revised implementation, not the earlier baseline. Run it in a fresh GPU kernel,
one notebook at a time. A nominal 16-GB GPU is the lab target. The first run needs internet,
approximately 25 GB free disk, and time to download pinned models. No hosted model API is used.

The default is a full run. `LAB2_SMOKE_ONLY=1` explicitly skips model-backed measurements;
it is a developer check, never a completed experiment. Real human ratings are never generated.
Device-memory sampling covers the actual subprocess stages but is not proof that an RTX 4090
is a 16-GB GPU. Keep the printed hardware identity and all outputs.
'''
    save('part1_rag','Part1_RAG_Run.ipynb',[
        md('# Part 1: multimodal, conversation-aware RAG\n\n'+common),code(BOOTSTRAP),
        md('## Check the implementation\nThese offline regression tests exercise real PDF extraction, Qdrant storage with temporary vectors, exact figure identity and isolated conversation state. Mock model outputs are test fixtures only.'),
        code('run_script("tools/test_lab_completion.py")'),
        md('## Locate the read-only data\nOn Kaggle attach the supplied competition data. `QUESTIONS_PATH` supports the hidden set without code edits. No question-specific rules or answer lookup is used.'),
        code('''P1 = ROOT / "part1_rag"
DATA = Path(os.environ.get("LAB2_DATA_DIR", str(P1 / "data"))).resolve()
if Path("/kaggle/input").is_dir() and "LAB2_DATA_DIR" not in os.environ:
    manifests = list(Path("/kaggle/input").rglob("kb_manifest.json"))
    if manifests:
        DATA = manifests[0].parent
QUESTIONS = Path(os.environ.get("QUESTIONS_PATH", str(DATA / "Questions.csv"))).resolve()
OUTPUT = P1 / "outputs/v2/submission.csv"
if not QUESTIONS.is_file():
    raise FileNotFoundError("Attach Questions.csv or set QUESTIONS_PATH to the supplied hidden-set file.")
print("Data:", DATA, "Questions:", QUESTIONS)'''),
        md('## Ingest, retrieve and generate\nNumbered figures/tables retain exact labels, including the corrected version. Trusted text and CLIP image vectors are stored in local Qdrant. Appendix C is quarantined, never used as answer evidence. Each conversation starts with independent memory; user assumptions never modify the source PDFs or database.'),
        code('''if SMOKE_ONLY:
    print("SKIPPED: full VLM answers and competition submission. Unit-test vectors are not evaluation embeddings.")
else:
    run_script("part1_rag/src/generate_submission.py", "--data-dir", DATA,
               "--questions-path", QUESTIONS, "--output", OUTPUT, "--require-dense")
    run_script("part1_rag/data/validate_submission.py", OUTPUT, QUESTIONS)
    import csv
    with OUTPUT.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(json.dumps(rows[:3], indent=2))
    print("Validated submission:", OUTPUT)'''),
        md('## Dense-only baseline versus the final system\nThe 25 candidate golden references are AI-assisted proposals committed before this run. Both team members must review them and replace or correct them as needed. Their `team_verified` flags remain false until that happens. Numeric/status checks and VLM citation entailment are labeled automatic proxies, not fabricated human results.'),
        code('''if SMOKE_ONLY:
    print("SKIPPED: golden-set model evaluation and human ratings.")
else:
    run_script("part1_rag/src/evaluate_rag.py", "--data-dir", DATA,
               "--output", P1 / "outputs/v2/evaluation",
               "--golden", os.environ.get("LAB2_GOLDEN_PATH", str(P1 / "eval/golden_candidate.json")))
    print((P1 / "outputs/v2/evaluation/evaluation.json").read_text())'''),
        md('## Review and export\nComplete the real groundedness form for both systems, refresh the report, and save this executed notebook. Upload/select the validated submission on Kaggle manually; the code does not claim a leaderboard submission.'),
        code('''run_script("tools/prepare_rating_forms.py")
evaluation_folder = P1 / "outputs/v2/evaluation"
if not SMOKE_ONLY and (evaluation_folder / "evaluation.json").is_file():
    run_script("part1_rag/src/evaluate_rag.py", "--refresh-review", "--output", evaluation_folder,
               "--golden", os.environ.get("LAB2_GOLDEN_PATH", str(P1 / "eval/golden_candidate.json")))
run_script("tools/final_submission_audit.py")
run_script("tools/build_combined_report.py")
run_script("tools/export_lab_results.py", "--part", "part1_rag")''')])
    save('part3_agents','Part3_Agents_Run.ipynb',[
        md('# Part 3: image agents with measured evaluation\n\n'+common+'\nYour Part 2 training is reused, not repeated. Private adapter download asks for a read token securely before the expensive stages. Never put the token in this notebook or chat.'),
        code(BOOTSTRAP),
        md('## Offline regression tests\nNo mocked output is counted as task success. The full model-backed tests below are separate.'),
        code('run_script("tools/test_lab_completion.py")\nrun_script("part3_agents/src/smoke_test.py")'),
        md('## Real inputs and the pinned Part 2 Hub adapter\nThree actual held-out Naruto images are copied locally; synthetic instruction overlays and broken files are explicitly safety fixtures. This is a narrow anime-domain test set, not proof of broad photographic performance. Input hashes are saved. The adapter is fetched from its verified Hugging Face commit, not silently replaced by local/base weights.'),
        code('''P3 = ROOT / "part3_agents"
RUN_DIR = P3 / "outputs/v2" / ("smoke" if SMOKE_ONLY else "runs") / time.strftime("%Y%m%d_%H%M%S")
RUN_DIR.mkdir(parents=True, exist_ok=True)
(P3 / ("outputs/v2/latest_smoke.json" if SMOKE_ONLY else "outputs/v2/latest_run.json")).write_text(json.dumps({"folder": str(RUN_DIR)}))
INPUTS = P3 / "outputs/v2/inputs/inputs.json"
ADAPTER = P3 / "outputs/hub_adapter"
if SMOKE_ONLY:
    print("SKIPPED: real image preparation, private adapter download and full models.")
else:
    from getpass import getpass
    temporary_token = False
    if not os.environ.get("HF_TOKEN"):
        os.environ["HF_TOKEN"] = getpass("Hugging Face read token for your private adapter (hidden): ")
        temporary_token = True
    try:
        run_script("part3_agents/src/prepare_evaluation.py", "--download-adapter")
    finally:
        if temporary_token:
            os.environ.pop("HF_TOKEN", None)
    print("Adapter provenance:", (ADAPTER / "hub_provenance.json").read_text())'''),
        md('## Run both systems\nStyle changes use SD 1.5 img2img; edits use InstructPix2Pix. Domain requests use the pinned Part 2 adapter. Qwen2-VL transcribes image text, answers visual questions, rewrites prompts and critiques original versus generated images. The VLM is unloaded before diffusion to bound concurrent VRAM. Both configurations use the same inputs, prompts and initial seed; revisions are capped at two.'),
        code('''if SMOKE_ONLY:
    print("SKIPPED: 30 real model-backed scenario executions.")
else:
    for filename, without in [("scenario_traces_no_critique.json", True), ("scenario_traces.json", False)]:
        arguments = ["--inputs", INPUTS, "--output", RUN_DIR / filename,
                     "--enable-models", "--lora-path", ADAPTER]
        if without:
            arguments.append("--no-critique")
        run_script("part3_agents/src/run_scenarios.py", *arguments)
    from IPython.display import Image, display
    for filename in ["scenario_traces_no_critique.json", "scenario_traces.json"]:
        traces = json.loads((RUN_DIR / filename).read_text())
        for trace in traces:
            print(filename, trace["scenario_id"], trace["request"], trace["decision"], trace["message"])
            print("Caption:", trace["caption"], "Answer:", trace["visual_answer"])
            print("Critique:", trace["critique"], "Seconds:", trace["seconds"])
            if trace["output_path"]:
                display(Image(filename=trace["output_path"], width=384))'''),
        md('## Real image-based safety evaluation\nThe ten specified cases include instructions in actual image pixels, unsafe requests, a corrupt file, an empty file and missing input. A model/runtime error is not accepted as a successful safe refusal. Baseline and critique results are measured separately.'),
        code('''if SMOKE_ONLY:
    print("SKIPPED: image-based attack-success measurement.")
else:
    for filename, without in [("red_team_results_no_critique.json", True), ("red_team_results.json", False)]:
        arguments = ["--inputs", INPUTS, "--enable-models", "--output", RUN_DIR / filename]
        if without:
            arguments.append("--no-critique")
        run_script("part3_agents/src/run_red_team.py", *arguments)
    run_script("part3_agents/src/evaluate_agents.py", "--folder", RUN_DIR)
    print((RUN_DIR / "evaluation.json").read_text())'''),
        md('## Independent team judgments\nEach member rates every baseline and final scenario against its saved rubric. Fill `human_ratings_template.json`, save as `human_ratings.json`, and record how disagreements were resolved. Until then task-success and critic-agreement values stay pending. Automatic critic acceptance is never substituted for a human pass.'),
        code('''run_script("tools/prepare_rating_forms.py")
if not SMOKE_ONLY:
    run_script("part3_agents/src/evaluate_agents.py", "--folder", RUN_DIR)
run_script("tools/final_submission_audit.py")
run_script("tools/build_combined_report.py")
run_script("tools/export_lab_results.py", "--part", "part3_agents")'''),
        md('## Optional upload action after reviewing results\nThe generated images belong on Hugging Face, not in Git history. This action stays disabled until you review the actual outputs and permitted sharing. Enter a write token securely, never in source code. Only final safe outputs and their metadata are uploaded privately; arrange grader access separately.'),
        code('''UPLOAD_PART3_IMAGES = False
RIGHTS_REVIEWED = False
if UPLOAD_PART3_IMAGES and not SMOKE_ONLY:
    if not RIGHTS_REVIEWED:
        raise ValueError("Review educational sharing rights before upload.")
    from getpass import getpass
    os.environ["HF_TOKEN"] = getpass("Hugging Face write token (hidden): ")
    try:
        run_script("part3_agents/src/publish_results.py", "--folder", RUN_DIR,
                   "--namespace", "denishatank", "--rights-reviewed")
    finally:
        os.environ.pop("HF_TOKEN", None)
else:
    print("Part 3 image upload not requested; no upload is claimed.")'''),
        md('## Gradio demo\nRun this cell when you are ready to test the interactive demo. It stays local and does not expose a public server. Show one caption/question, one generated edit/style request and one safe failure; both team members need to rehearse. The notebook does not pretend a disabled app was demonstrated.'),
        code('''LAUNCH_APP = False
if LAUNCH_APP and not SMOKE_ONLY:
    app_process = subprocess.Popen([str(PYTHON), str(P3 / "src/app.py")], cwd=ROOT)
    print("Local demo started. Open the local address printed by Gradio; stop app_process when finished.")
else:
    print("Interactive demo not launched. Enable this cell when ready to rehearse.")''')])
    # Preserve every completed Part 2 cell/result, adding only new review/export cells.
    path=ROOT/'part2_diffusion/notebooks/Part2_FineTune_Run.ipynb'
    n=nb.read(path,as_version=4)
    if not any('## Completion refresh without retraining' in c.source for c in n.cells):
        n.cells.extend([
            md('## Completion refresh without retraining\nThe earlier RTX 4090 training cells and outputs are preserved. Run only the following review cell when ratings, team details or access evidence changes. It does not train or generate new images. Hub records are content-hashed, so a newly retrained adapter cannot borrow an old upload claim.'),
            code('''P2 = ROOT / "part2_diffusion"
import shutil
registry = P2 / "evidence/hub_links.json"
local_registry = P2 / "outputs/hub_links.json"
if not local_registry.exists():
    shutil.copy2(registry, local_registry)
run_script("part2_diffusion/src/submission.py")
run_script("part2_diffusion/src/report_pdf.py")
run_script("tools/final_submission_audit.py")
run_script("tools/build_combined_report.py")
run_script("tools/export_lab_results.py", "--part", "part2_diffusion")''')])
        nb.write(n,path)
    print('Updated Part 1 and Part 3 run notebooks; preserved Part 2 GPU results and appended refresh cells.')


if __name__=='__main__':
    main()
