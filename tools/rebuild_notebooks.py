"""Generate portable notebooks; all heavy work uses the isolated runtime."""
from pathlib import Path
import nbformat as nb

ROOT=Path(__file__).resolve().parents[1]
BOOTSTRAP='''from pathlib import Path
import os, subprocess, sys, platform, json, time

os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

SMOKE_ONLY = os.environ.get("LAB2_SMOKE_ONLY", "0") == "1"
roots = [Path.cwd(), *Path.cwd().parents]
ROOT = next((p for p in roots if (p / "tools/setup_runtime.py").exists()), None)
if ROOT is None:
    if os.environ.get("LAB2_ROOT"):
        ROOT = Path(os.environ["LAB2_ROOT"]).expanduser().resolve()
    elif Path("/content").exists():
        ROOT = Path("/content/Lab-2")
    elif Path("/kaggle/working").exists():
        ROOT = Path("/kaggle/working/Lab-2")
    else:
        ROOT = Path.home() / "DATA266" / "Lab-2"
    if not ROOT.exists():
        ROOT.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "https://github.com/denishatank12/Lab-2.git", str(ROOT)], check=True)
    if not (ROOT / "tools/setup_runtime.py").exists():
        raise RuntimeError(f"{ROOT} is an older/incomplete checkout. Update it with git pull, then restart this notebook.")
print("Notebook host:", platform.python_version(), "repository:", ROOT)
print("Mode:", "SMOKE ONLY: no full training or model-backed evaluation" if SMOKE_ONLY else "FULL RUN")
subprocess.run([sys.executable, str(ROOT / "tools/setup_runtime.py"),
                "--device", "cpu" if SMOKE_ONLY else os.environ.get("LAB2_DEVICE", "auto")], check=True)
PYTHON = Path(json.loads((ROOT / ".lab2_runtime.json").read_text())["python"])
SEED = 266
print("All remaining commands use:", PYTHON)

def run_script(relative, *arguments):
    command = [str(PYTHON), "-u", str(ROOT / relative), *map(str, arguments)]
    logs = ROOT / "runtime_checks"
    logs.mkdir(exist_ok=True)
    log = logs / (Path(relative).stem + ".log")
    print("Running", Path(relative).name, flush=True)
    with log.open("a", encoding="utf-8") as handle:
        process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        for line in process.stdout:
            print(line, end="", flush=True)
            handle.write(line)
            handle.flush()
        code = process.wait()
    if code:
        raise RuntimeError(f"{Path(relative).name} stopped (exit {code}). Full diagnostic log: {log}")
'''


def write(part, filename, cells):
    n=nb.v4.new_notebook(cells=cells)
    n.metadata={"kernelspec":{"name":"python3","language":"python","display_name":"Python 3"},
                "language_info":{"name":"python"}}
    nb.validate(n)
    nb.write(n,ROOT/part/'notebooks'/filename)


def md(text): return nb.v4.new_markdown_cell(text)
def code(text): return nb.v4.new_code_cell(text)


common = """The first code cell creates a pinned environment and prints its commit, versions,
device and seed. It does not replace packages inside the running notebook kernel.
Windows, Linux, Colab and Kaggle paths are handled without a hard-coded `/content`
directory on a PC. First-time setup needs internet and several GB of free disk.

Use a fresh kernel and Run All. The default is a full run. `LAB2_SMOKE_ONLY=1`
is only for developer checks and is clearly printed; those outputs are not a
completed GPU experiment. Keep your real executed copy after the full run.
"""
write('part1_rag','Part1_RAG_Run.ipynb',[
md('# Lab 2 Part 1: RAG runtime\n\n'+common+"\nThis remains the repository's extractive retrieval baseline, not a claim that all graded RAG features are finished."),
    code(BOOTSTRAP),
md('## Smoke test\nCheck PDF ingestion, trusted-page filtering and lexical retrieval before downloading an embedding model.'),
code('run_script("part1_rag/src/smoke_test.py")'),
md('## Generate and validate\nThe hidden-set questions path can be supplied through `QUESTIONS_PATH`. It is also passed to the unchanged official validator. A schema-valid file alone does not prove answer correctness.'),
code('''DATA = Path(os.environ.get("LAB2_DATA_DIR", str(ROOT / "part1_rag/data"))).resolve()
QUESTIONS = Path(os.environ.get("QUESTIONS_PATH", str(DATA / "Questions.csv"))).resolve()
OUTPUT = ROOT / "part1_rag/outputs/submission.csv"
run_script("part1_rag/src/generate_submission.py", "--data-dir", DATA,
           "--questions-path", QUESTIONS, "--output", OUTPUT,
           "--lexical-only" if SMOKE_ONLY else "--require-dense")
run_script("part1_rag/data/validate_submission.py", OUTPUT, QUESTIONS)
import csv
with OUTPUT.open(encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle))
print("Validated rows:", len(rows), "file:", OUTPUT)
print(json.dumps(rows[:2], indent=2))''')])
GPU_BOOTSTRAP=BOOTSTRAP.replace('os.environ.get("LAB2_DEVICE", "auto")','os.environ.get("LAB2_DEVICE", "cuda")')
write('part2_diffusion','Part2_FineTune_Run.ipynb',[
md('# Lab 2 Part 2: checkpointed LoRA\n\n'+common+"\nRTX 4050 profile: batch 1, accumulation 4, fp16 frozen UNet, gradient checkpointing, cached latents and text conditioning. No bitsandbytes or xformers dependency. The lab grading target remains a 16 GB GPU."),
    code(GPU_BOOTSTRAP),
md('## Preflight and real optimizer smoke test\nA tiny randomly initialized UNet tests forward/backward, frozen base weights, checkpoint save, adapter export and interrupted-run resume. It does not count as the SD 1.5 training run.'),
    code('''if not SMOKE_ONLY:
    run_script("tools/preflight.py", "--require-cuda")
run_script("part2_diffusion/src/smoke_test.py", *([] if SMOKE_ONLY else ["--require-cuda"]))'''),
md('## Prepare the actual dataset\nCorrupt images, duplicates and empty captions are counted. The seed really controls the 90/10 split. Evaluation prompts are frozen from held-out captions before either training run.'),
code('''P2 = ROOT / "part2_diffusion"
if SMOKE_ONLY:
    print("SKIPPED: full dataset download and conditioning cache (developer smoke mode).")
else:
    run_script("part2_diffusion/src/prepare_data.py", "--output-dir", P2 / "data/processed")
    run_script("part2_diffusion/src/generate_eval.py", "--freeze-prompts")'''),
md('## Train both ranks\nOnly one UNet is trained at a time. Complete checkpoints are resumed automatically. A finished compatible run is kept. Dataset and configuration mismatches stop rather than mixing experiments.'),
code('''if SMOKE_ONLY:
    print("SKIPPED: both full SD 1.5 runs. The tiny-model test is not submitted training.")
else:
    for name, rank in [("run1", 4), ("run2", 8)]:
        run_script("part2_diffusion/src/train_lora.py", "--config", P2 / "config.yaml",
                   "--run", name, "--rank", rank, "--resume-latest")'''),
md('## Generate and calculate metrics\nAll 120 images use identical settings. Missing adapters are errors, never silently replaced by the base model. The checkpoint safety checker stays enabled. Inference offloads components instead of loading the whole pipeline into VRAM. KID, CLIP and LPIPS are actually computed; human judgments still need real raters.'),
code('''if SMOKE_ONLY:
    print("SKIPPED: real image generation, quality metrics and human judgments.")
else:
    run_script("part2_diffusion/src/generate_eval.py", "--config", P2 / "config.yaml")
    run_script("part2_diffusion/src/evaluate.py", "--config", P2 / "config.yaml")
    for name in ["run1", "run2"]:
        print(name, json.loads((P2 / "outputs" / name / "training_metrics.json").read_text()))
    print((P2 / "outputs/evaluation.json").read_text())
    from IPython.display import Image, display
    for index in range(8):
        display(Image(filename=str(P2 / "outputs/generations" / f"comparison_{index:02d}.png")))'''),
md('## Save your work\nKeep this executed notebook, both adapters and the outputs folder. Upload weights and images to your team Hugging Face account, not GitHub. Complete the 25 blinded pairs with at least three raters; the code does not invent their choices.')])
write('part3_agents','Part3_Agents_Run.ipynb',[
md('# Lab 2 Part 3: image agents\n\n'+common+"\nRun Part 2 first for the domain-style adapter. This notebook uses a clearly marked synthetic image to check wiring. Replace `INPUT_IMAGE` with your chosen lab test images for the real evaluation; one synthetic input is not the required diverse test set."),
    code(GPU_BOOTSTRAP),
md('## Smoke and red-team checks\nMocked model calls exercise routing, missing/corrupt inputs, bounded revisions and generation exceptions. These tests are not model-quality or human-evaluation scores.'),
code('''run_script("part3_agents/src/smoke_test.py")
run_script("part3_agents/src/run_red_team.py", "--output", ROOT / "runtime_checks/red_team.json")
if not SMOKE_ONLY:
    run_script("tools/preflight.py", "--require-cuda")'''),
md('## Choose an image\nProvide a real file path with `LAB2_INPUT_IMAGE`, or replace `INPUT_IMAGE` below. The synthetic fallback is only a functional demonstration.'),
code('''P3 = ROOT / "part3_agents"
INPUT_IMAGE = Path(os.environ.get("LAB2_INPUT_IMAGE", str(P3 / "sample_input.png")))
if not INPUT_IMAGE.exists() and "LAB2_INPUT_IMAGE" not in os.environ:
    run_script("tools/make_sample_image.py", INPUT_IMAGE)
    print("Using SYNTHETIC DEBUG IMAGE, not the final evaluation set.")
if not INPUT_IMAGE.is_file():
    raise FileNotFoundError(f"Input image does not exist: {INPUT_IMAGE}")
print("Input:", INPUT_IMAGE)'''),
md('## Compare with and without critique\nCaptioning and visual QA run on CPU; a single img2img pipeline is offloaded for the RTX 4050. Every scenario and revision gets a separate image path. Generation failures are saved as errors, not marked successful. There are at most two revisions.'),
code('''ADAPTER = ROOT / "part2_diffusion/outputs/run1"
if not SMOKE_ONLY and not (ADAPTER / "pytorch_lora_weights.safetensors").is_file():
    raise RuntimeError("Finish Part 2 run1 first, or point ADAPTER to a completed adapter folder.")
for filename, without in [("scenario_traces_no_critique.json", True), ("scenario_traces.json", False)]:
    arguments = ["--image", INPUT_IMAGE, "--output", P3 / "outputs" / filename]
    if without:
        arguments.append("--no-critique")
    if not SMOKE_ONLY:
        arguments.extend(["--enable-models", "--lora-path", ADAPTER])
    run_script("part3_agents/src/run_scenarios.py", *arguments)
print("Traces saved. Judge task success and critic agreement using your real scenario rubrics.")'''),
md('## Optional local demo app\nLaunch only when you want the interactive demo. It is disabled by default so Run All can finish. The server stays local; public sharing is not enabled.'),
code('''LAUNCH_APP = False
if LAUNCH_APP:
    app_process = subprocess.Popen([str(PYTHON), str(P3 / "src/app.py")], cwd=ROOT)
    print("Local app started. Stop app_process when finished.")
else:
    print("App not launched. Run the app.py command when ready for the demo.")'''),
md('## Limits\nVisible text is currently a VQA guess, not verified OCR. The image critic uses CLIP and a generated caption, not independent human judgment. Text guardrails and the diffusion safety checker reduce risk but are not a guarantee. Full grading also requires real test images, human judgments, team documentation and model uploads.')])
print('Rebuilt all three notebooks.')
