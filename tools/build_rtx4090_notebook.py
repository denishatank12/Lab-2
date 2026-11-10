"""Build the RTX 4090 launcher without replacing any completed experiment notebook."""
from pathlib import Path
import nbformat as nb

from update_final_notebooks import BOOTSTRAP, add_download_section

ROOT = Path(__file__).resolve().parents[1]


def build():
    # Also recover a patched checkout that has schemas but not this launcher.
    bootstrap = BOOTSTRAP.replace('tools/response_schemas.py', 'tools/rtx4090_workflow.py')
    bootstrap = bootstrap.replace('-structured-v1', '-rtx4090-v1')
    # Large pretrained downloads should not start on an accidentally selected CPU.
    bootstrap = bootstrap.replace('SMOKE_ONLY = os.environ.get("LAB2_SMOKE_ONLY", "0") == "1"', '''SMOKE_ONLY = os.environ.get("LAB2_SMOKE_ONLY", "0") == "1"
os.environ["LAB2_DEVICE"] = "cpu" if SMOKE_ONLY else "cuda"
if not SMOKE_ONLY:
    try:
        gpu_names = subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True, timeout=15)
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError("Connect this notebook to your RTX 4090 kernel first. A normal hosted Colab kernel is not your PC GPU.") from error
    if "4090" not in gpu_names:
        raise RuntimeError("No RTX 4090 is visible to this kernel. Correct the runtime before downloading models.")
    print("Visible GPU(s):", gpu_names.strip())''')
    md, code = nb.v4.new_markdown_cell, nb.v4.new_code_cell
    notebook = nb.v4.new_notebook(cells=[
        md('''# Lab 2 on the RTX 4090

This runs the repaired RAG and image-agent experiments in order and keeps their executed notebooks. It reuses your completed Part 2 training, 120 generated images and measured metrics. It does not repeat training or count the three incomplete RAG answers as finished work.

Before Run All, connect Colab to the local runtime on the RTX 4090 PC, or open this file in Jupyter on that PC. A hosted Colab GPU is a different machine. Use Python 3.10 through 3.12 if available; setup creates a separate pinned environment if needed. Git, internet access and the NVIDIA driver must work. Allow roughly 40 GB free disk for a cold installation and at least 8 GiB available host RAM. Close other GPU notebooks, plug in power and disable sleep.

Keep your original `Part2_results_*.zip` on the GPU machine. Its files are not restored just by opening GitHub. The next cell can locate that archive in Downloads, the repository, `/content` or `/app`. If it is elsewhere, set its runtime path below. A Docker kernel needs a mounted folder or an uploaded ZIP, not a Windows path the container cannot see.

Leave `LAB2_SMOKE_ONLY` unset for real computation. Setting it to `1` is a developer test that skips all model-backed work. Human ratings, reference review, team details, permissions, the demo and final external submissions still need real people. A 24-GB 4090 run is not evidence of execution on the rubric's 16-GB card.'''),
        code('''import os

# Only change this if the original ZIP is not found automatically.
# Example on Windows: r"C:\\Users\\YOUR_NAME\\Downloads\\Part2_results_20261106_025010.zip"
# Example in a Linux container with the ZIP uploaded: "/app/Part2_results_20261106_025010.zip"
PART2_RESULTS_ZIP = os.environ.get("LAB2_PART2_RESULTS_ZIP", "")

# Optional: set LAB2_ROOT before setup if you use a different existing checkout.
# Never put a Hugging Face token or password in this cell.
os.environ.setdefault("LAB2_UPDATE_REPOSITORY", "1")'''),
        md('''## Set up the isolated runtime

This checks the visible GPU before installing packages. A patched old checkout is preserved; a clean sibling checkout is used instead. All following scripts run in the pinned environment, not whichever packages happen to be in the Colab kernel.'''),
        code(bootstrap),
        md('''## Test the code and check the actual hardware

The offline tests cover response structure, image handling, recovery and ZIP downloads. The CUDA check then runs on the real 4090. These checks are separate from model-quality evaluation.'''),
        code('''for test in ["test_lab_completion.py", "test_structured_responses.py", "test_constrained_decoding.py",
             "test_overnight_resume.py", "test_part2_submission.py", "test_result_download.py",
             "test_rtx4090_workflow.py"]:
    run_script("tools/" + test)
if SMOKE_ONLY:
    print("SKIPPED: actual RTX 4090 verification. CPU smoke is not GPU evidence.")
else:
    run_script("tools/rtx4090_workflow.py", "profile")'''),
        md('''## Restore the completed Part 2 results

If all original files are already here, they are reused. Otherwise, the original ZIP is imported with backups. Both adapters must match their recorded uploaded hashes. All 120 image combinations, image shapes, saved training settings, quality metrics and memorization figures are checked. A missing archive stops here before expensive inference.

The selected adapter is copied from this verified local export, so no private Hub token is needed for that copy. This does not verify grader access to the private Hub repositories. Dataset metadata alone is not enough: missing source images are downloaded again using the original pinned dataset and split.'''),
        code('''DATA = Path(os.environ.get("LAB2_DATA_DIR", str(ROOT / "part1_rag/data"))).expanduser().resolve()
QUESTIONS = Path(os.environ.get("QUESTIONS_PATH", str(DATA / "Questions.csv"))).expanduser().resolve()
if not SMOKE_ONLY:
    if not QUESTIONS.is_file():
        raise FileNotFoundError("Questions.csv is missing. Attach the supplied Part 1 data or set LAB2_DATA_DIR and QUESTIONS_PATH.")
    arguments = ["reuse-part2"]
    if PART2_RESULTS_ZIP:
        arguments += ["--archive", PART2_RESULTS_ZIP]
    run_script("tools/rtx4090_workflow.py", *arguments)
    run_script("part2_diffusion/src/prepare_data.py", "--output-dir", ROOT / "part2_diffusion/data/processed")
    run_script("part3_agents/src/prepare_evaluation.py", "--download-adapter")
else:
    print("SKIPPED: original Part 2 import, dataset download and adapter preparation.")'''),
        md('''## Run the real models before leaving the PC

These checks load the real embedding, visual-language, image-editing and style-generation paths, including the original domain adapter. They run sequentially and populate the model cache. Keep the notebook connected and confirm `BOTH REAL GPU PREFLIGHTS PASSED` before leaving the PC. Two-step diagnostic images are not final evaluation samples.'''),
        code('''if SMOKE_ONLY:
    print("SKIPPED: real GPU model preflights. Do not leave a GPU job based only on this smoke output.")
else:
    run_script("tools/overnight_preflight.py", "--part", "part1_rag", "--data-dir", DATA)
    run_script("tools/overnight_preflight.py", "--part", "part3_agents")
    print("BOTH REAL GPU PREFLIGHTS PASSED. The sequential full run starts in the next cell.")'''),
        md('''## Run Parts 1 and 3 and save the executed notebooks

Part 1 generates and validates the supplied question CSV, then evaluates the dense baseline and final system. Part 3 runs 15 scenarios with and without critique and ten image-aware safety cases for each. There are at most two revisions per request. Models run one at a time.

Completed unchanged items resume after an interruption. A changed source, input or GPU creates a separate checkpoint namespace. Model errors are saved rather than turned into fake answers or successful safety refusals. Executed notebook copies save after every cell under `output/executed`. Keep the GPU runtime and local server running.'''),
        code('''run_script("tools/overnight_runner.py", *(["--smoke-only"] if SMOKE_ONLY else []))
print("Executed notebook folder:", ROOT / "output/executed")
print("Progress record:", ROOT / "runtime_checks/overnight_progress.json")'''),
        md('''## Check the saved results

This checks execution separately from answer quality and submission readiness. A script exiting normally does not hide model errors in its saved traces. Expected corrupt-image refusals remain safety tests, not runtime crashes. Real ratings stay blank until the raters fill them in.

The reports refresh from saved measurements in the inner notebooks. They stay drafts while real reviews and external requirements are pending. If this cell fails, keep its diagnostic and run the ZIP cell below; do not delete completed work or edit failures out of the notebook.'''),
        code('''run_script("tools/rtx4090_workflow.py", "completion", *(["--smoke-only"] if SMOKE_ONLY else []))
run_script("tools/prepare_rating_forms.py")
run_script("tools/final_submission_audit.py")
print("Readiness audit:", ROOT / "runtime_checks/final_submission_audit.json")
print("The next cell exports all available results. It does not submit to Canvas or Kaggle.")''')])
    add_download_section(notebook)
    notebook.metadata = {
        'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'},
        'language_info': {'name': 'python'},
        'validation_scope': 'RTX 4090 full computation requires the actual GPU run. CPU smoke is developer validation only.',
        'part2_policy': 'Reuse original hash-verified completed experiment; never retrain automatically.'}
    nb.validate(notebook)
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == 'code':
            compile(cell.source, f'RTX4090-cell-{index}', 'exec')
    path = ROOT / 'Lab2_RTX4090_Run_All.ipynb'
    nb.write(notebook, path)
    print('Created', path)
    return path


if __name__ == '__main__':
    build()
