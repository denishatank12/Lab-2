"""Package executable source inside one notebook; users supply no helper files."""
from __future__ import annotations

import ast
import base64
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import textwrap
import zipfile

import nbformat as nb

ROOT = Path(__file__).resolve().parents[1]

# The main Colab kernel need not import a single ML package. Children use the
# isolated pinned interpreter; installing packages cannot invalidate the kernel.
SETUP = '''from pathlib import Path
import os, sys, subprocess, json, time

ROOT = Path(os.environ["LAB2_STANDALONE_ROOT"]).resolve()
SMOKE_ONLY = os.environ.get("LAB2_SMOKE_ONLY", "0") == "1"
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
if not SMOKE_ONLY:
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)
os.environ["LAB2_DEVICE"] = "cpu" if SMOKE_ONLY else "cuda"

# This override exists only for a developer CPU smoke, never for a GPU run.
test_python = os.environ.get("LAB2_STANDALONE_TEST_PYTHON", "") if SMOKE_ONLY else ""
if test_python:
    PYTHON = Path(test_python).expanduser().absolute()
    if not PYTHON.is_file():
        raise FileNotFoundError("The explicit CPU test interpreter does not exist.")
    subprocess.run([str(PYTHON), str(ROOT / "tools/preflight.py")], cwd=ROOT, check=True)
    (ROOT / ".lab2_runtime_cpu.json").write_text(json.dumps({"python": str(PYTHON), "cuda_requested": False}))
else:
    subprocess.run([sys.executable, str(ROOT / "tools/setup_runtime.py"),
                    "--device", "cpu" if SMOKE_ONLY else "cuda"], cwd=ROOT, check=True)
    marker = ROOT / (".lab2_runtime_cpu.json" if SMOKE_ONLY else ".lab2_runtime_cuda.json")
    PYTHON = Path(json.loads(marker.read_text())["python"])
SEED = 266
print("Mode:", "CPU SMOKE ONLY: model calls are skipped" if SMOKE_ONLY else "FULL MODEL RUN")
print("Working folder:", ROOT)
print("Pinned interpreter:", PYTHON)

def run_script(relative, *arguments):
    record = ROOT / "runtime_checks" / (Path(relative).stem + "_" + str(time.time_ns()) + ".memory.json")
    command = [str(PYTHON), "-u", str(ROOT / "tools/monitored_run.py"), "--record", str(record),
               "--", str(PYTHON), "-u", str(ROOT / relative), *map(str, arguments)]
    logs = ROOT / "runtime_checks"
    logs.mkdir(exist_ok=True)
    log = logs / (Path(relative).stem + ".log")
    print("Running", Path(relative).name, flush=True)
    with log.open("a", encoding="utf-8") as stream:
        process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        for line in process.stdout:
            print(line, end="", flush=True)
            stream.write(line)
            stream.flush()
        status = process.wait()
    if status:
        raise RuntimeError(f"{Path(relative).name} stopped with exit {status}. Saved log: {log}. Run the final ZIP cell to retain all partial results.")
'''


def strip_notebook(path):
    notebook = nb.read(path, as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
    return notebook


def package_notebooks():
    files = {}
    for part, name in [("part1_rag", "Part1_RAG_Run.ipynb"), ("part3_agents", "Part3_Agents_Run.ipynb")]:
        n = strip_notebook(ROOT / part / "notebooks" / name)
        next(c for c in n.cells if c.cell_type == "code").source = SETUP
        for cell in n.cells:
            if cell.cell_type == "markdown":
                cell.source = cell.source.replace("on this RTX 4050", "on this GPU").replace("RTX 4050 readiness", "GPU readiness")
        # The restored adapter is already verified locally and has its original
        # pinned Hub provenance. No child kernel needs a second login prompt.
        n.metadata["standalone_execution"] = True
        files[f"{part}/notebooks/{name}"] = nb.writes(n).encode()
    md, code = nb.v4.new_markdown_cell, nb.v4.new_code_cell
    n = nb.v4.new_notebook(cells=[
        md("# Restored Part 2 evidence\n\nThese are your original completed RTX 4090 results, restored from the pinned private Hub artifacts and saved measured records. This execution verifies and displays them. It does not retrain the model or relabel the original timing as a new Colab run."),
        code(SETUP),
        code('''if SMOKE_ONLY:
    print("SKIPPED: private artifact recovery and actual Part 2 evidence display.")
else:
    P2 = ROOT / "part2_diffusion"
    print((ROOT / "runtime_checks/standalone_part2_restoration.json").read_text())
    for name in ["run1", "run2"]:
        print("ORIGINAL RTX 4090 TRAINING:", name, (P2 / "outputs" / name / "training_metrics.json").read_text())
    print("ORIGINAL QUALITY METRICS:", (P2 / "outputs/evaluation.json").read_text())'''),
        code('''if not SMOKE_ONLY:
    from IPython.display import Image, display
    for index in range(8):
        display(Image(filename=str(P2 / "outputs/generations" / f"comparison_{index:02d}.png")))
    run_script("part2_diffusion/src/submission.py")
    run_script("part2_diffusion/src/report_pdf.py")
else:
    print("SKIPPED: generating a measured report from CPU smoke.")''')])
    n.metadata = {'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'},
        'language_info': {'name': 'python'}, 'part2_execution_scope': 'Restore and display the original measured RTX 4090 run; no new training.'}
    files['part2_diffusion/notebooks/Part2_FineTune_Run.ipynb'] = nb.writes(n).encode()
    return files


def collect_files():
    files = {}
    for folder in ['tools', 'part1_rag/src', 'part2_diffusion/src', 'part3_agents/src']:
        for path in (ROOT / folder).glob('*.py'):
            if path.name == 'test_standalone_colab.py':
                continue  # Its clean-room integration test lives outside its own payload.
            files[path.relative_to(ROOT).as_posix()] = path.read_bytes()
    for name in ['requirements.txt', 'README.md', 'RERUN_GUIDE.md', 'AI_USAGE.md', 'team_review.json',
                 'part2_diffusion/config.yaml', 'part2_diffusion/submission_details.json',
                 'part2_diffusion/evidence/hub_links.json', 'part1_rag/eval/golden_candidate.json',
                 'part3_agents/eval/scenarios.json', 'part3_agents/eval/red_team_cases.json',
                 'part1_rag/data/validate_submission.py', 'Lab2_RTX4090_Run_All.ipynb']:
        files[name] = (ROOT / name).read_bytes()
    for path in (ROOT / 'docs/rating_forms').glob('*.md'):
        files[path.relative_to(ROOT).as_posix()] = path.read_bytes()
    files.update(package_notebooks())
    # Original data/images, weights and full previous notebooks are not embedded
    # or added to Git. Saved scalar records preserve their actual provenance.
    from rtx4090_workflow import part2_evidence, sha256
    part2_evidence(ROOT)
    saved = ['outputs/evaluation.json', 'outputs/memorization.json', 'outputs/generations/manifest.json',
             'outputs/review.json', 'data/processed/data_summary.json', 'data/processed/eval_prompts.json',
             'data/processed/train/metadata.jsonl', 'data/processed/test/metadata.jsonl']
    saved += [f'outputs/{run}/loss_history.json' for run in ['run1', 'run2']]
    recovery = dict(original_archive_sha256='5085f961414fa35c795233513219be4218730ac1591e271c8fca28f8696d8d7e',
        provenance='Measured records from the original user-provided Part 2 ZIP. Not new Colab measurements or human ratings.',
        saved_files={}, hub_files={})
    for relative in saved:
        path = ROOT / 'part2_diffusion' / relative
        recovery['saved_files'][relative] = sha256(path)
        files['standalone_evidence/original_part2/' + relative] = path.read_bytes()
    for run in ['run1', 'run2']:
        recovery['hub_files'][run] = {name: sha256(ROOT / 'part2_diffusion/outputs' / run / name)
            for name in ['pytorch_lora_weights.safetensors', 'run_config.json', 'training_metrics.json']}
    manifest = json.loads((ROOT / 'part2_diffusion/outputs/generations/manifest.json').read_text())
    recovery['hub_files']['images'] = {row['file']: sha256(ROOT / 'part2_diffusion/outputs/generations' / row['file']) for row in manifest}
    files['standalone_evidence/original_part2/RESTORE_MANIFEST.json'] = json.dumps(recovery, indent=2).encode()
    return files


def bundle():
    files = collect_files()
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip())
    resources = []
    for path in sorted((ROOT / 'part1_rag/data').iterdir()):
        if path.suffix.lower() not in {'.pdf', '.csv', '.json'}:
            continue
        name = path.relative_to(ROOT).as_posix()
        resources.append(dict(path=name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            url=f'https://raw.githubusercontent.com/denishatank12/Lab-2/{commit}/{name}'))
    manifest = dict(format_version=1, source_commit=commit, source_dirty_at_build=dirty,
        files={name: hashlib.sha256(data).hexdigest() for name, data in files.items()}, resources=resources,
        scope='Executable code is embedded. Course inputs and model artifacts are fetched with pinned hashes/revisions.')
    files['STANDALONE_MANIFEST.json'] = json.dumps(manifest, sort_keys=True, indent=2).encode()
    output = io.BytesIO()
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, data in sorted(files.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, data)
    return output.getvalue(), manifest


def build():
    from update_final_notebooks import add_download_section

    payload, manifest = bundle()
    digest = hashlib.sha256(payload).hexdigest()
    encoded = base64.b64encode(payload).decode()
    helper = (ROOT / 'tools/standalone_colab.py').read_text()
    tree = ast.parse(helper)
    unpack_source = ast.get_source_segment(helper, next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'unpack'))
    bootstrap = '''from pathlib import Path, PurePosixPath
import base64, hashlib, io, json, os, stat, subprocess, sys, zipfile

SMOKE_ONLY = os.environ.get("LAB2_SMOKE_ONLY", "0") == "1"
if not SMOKE_ONLY:
    try:
        visible_gpu = subprocess.check_output(["nvidia-smi", "-L"], text=True, timeout=15)
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError("Select Runtime > Change runtime type > GPU, then reconnect and Run All. A CPU runtime cannot run this experiment.") from error
    print(visible_gpu.strip())
'''
    bootstrap += '\n# All helper source is embedded below. No repository or .py upload is needed.\n'
    bootstrap += 'BUNDLE_SHA256 = ' + repr(digest) + '\n'
    bootstrap += 'BUNDLE_BASE64 = (\n' + '\n'.join('    ' + repr(s) for s in textwrap.wrap(encoded, 120)) + '\n)\n'
    bootstrap += '\n' + unpack_source + '\n'
    bootstrap += '''
default_parent = Path("/content") if Path("/content").is_dir() else Path.cwd()
parent = Path(os.environ.get("LAB2_STANDALONE_PARENT", str(default_parent))).expanduser().resolve()
ROOT = unpack(base64.b64decode(BUNDLE_BASE64), BUNDLE_SHA256, parent)
os.environ["LAB2_STANDALONE_ROOT"] = str(ROOT)
print("Embedded code verified:", BUNDLE_SHA256)
print("Source base commit:", json.loads((ROOT / "STANDALONE_MANIFEST.json").read_text())["source_commit"])
'''
    bootstrap += '\n' + SETUP
    md, code = nb.v4.new_markdown_cell, nb.v4.new_code_cell
    n = nb.v4.new_notebook(cells=[
        md('''# Lab 2 standalone Colab run

Open only this notebook. You do not need to upload Python files, clone a repository or provide a result ZIP. The first cell verifies its embedded code and creates its internal helper files automatically. Required course inputs and public model weights download automatically. Use Runtime > Change runtime type > GPU; T4 is the course's nominal 16-GB option. The actual GPU identity is recorded, not assumed to be an RTX 4090.

Part 2 already completed on your RTX 4090. This notebook restores both original adapters and all 120 original images from your private Hugging Face repositories, together with their saved measured records. It then runs the corrected Parts 1 and 3. Original Part 2 timings are not presented as a new Colab training run. This also does not establish that Part 2 trained on a 16-GB card.

Before leaving it running, provide a Hugging Face read token for those private repositories in the hidden prompt, or use a Colab secret named `HF_TOKEN`. Never paste it into source code or this chat. Keep roughly 40 GB of disk available for a cold setup and at least 8 GiB available host RAM. Leave other GPU notebooks stopped. Colab can still disconnect or end a session; no code can guarantee against that.

Use Run All and wait for both real-model preflights to pass. Results resume only while the same runtime files remain available. At the end, download the combined ZIP and save this notebook with its outputs. Real human ratings, team records, permissions and final external submission actions remain separate.'''),
        code(bootstrap),
        md('''## Download and verify the course inputs

The PDFs, question CSV and knowledge-base manifest use a pinned source revision and checked SHA-256 values. Executable source is already embedded; this cell does not download Python scripts. Completed unchanged input files are kept when a download needs retrying.'''),
        code('''run_script("tools/standalone_colab.py", "resources")
DATA = ROOT / "part1_rag/data"
os.environ["LAB2_DATA_DIR"] = str(DATA)
os.environ.setdefault("QUESTIONS_PATH", str(DATA / "Questions.csv"))
print("Inputs ready:", DATA)'''),
        md('''## Prepare standard Python kernels

This cell prevents Colab's host-only kernel configuration from leaking into the isolated Python environment. It runs before the notebook execution tests. It does not restart this notebook, reinstall packages, remove saved results or repeat Part 2 training.'''),
        code('''import os
import tempfile
from pathlib import Path

profile_parent = ROOT / "runtime_checks"
profile_parent.mkdir(parents=True, exist_ok=True)
repair = Path(tempfile.mkdtemp(prefix="standard-ipython-", dir=profile_parent))
profile = repair / "profile_default"
profile.mkdir()
(profile / "ipython_kernel_config.py").write_text(
    "c = get_config()\\n"
    "c.IPKernelApp.kernel_class = 'ipykernel.ipkernel.IPythonKernel'\\n",
    encoding="utf-8",
)
os.environ["JUPYTER_NO_CONFIG"] = "1"
os.environ["IPYTHONDIR"] = str(repair)
print("Standard Python kernel settings ready. Continue with the checks below.")'''),
        md('''## Run the offline regression checks

These temporary fixtures check structure, data handling, checkpoints and export. They are not pretrained-model quality scores. The actual model preflights follow later.'''),
        code('''for name in ["test_lab_completion.py", "test_structured_responses.py", "test_constrained_decoding.py",
             "test_overnight_resume.py", "test_part2_submission.py", "test_result_download.py", "test_rtx4090_workflow.py"]:
    run_script("tools/" + name)'''),
        md('''## Recover the original Part 2 without uploading any files

Use your own Hugging Face account's read token. It must be able to read both private adapters and the image dataset. Access is checked before expensive work. The original dataset split is restored as needed, then each saved adapter and generated image is checked against its recorded hash. Comparison panels are rebuilt from those same images, not generated again.

The measured values displayed in the saved Part 2 notebook are explicitly labeled as original RTX 4090 results. A completed restoration does not create new training evidence, human judgments, sharing rights or grader access.'''),
        code('''if SMOKE_ONLY:
    print("SKIPPED: private Hugging Face access, dataset download and original Part 2 restoration.")
    run_script("tools/standalone_colab.py", "record-part2")
else:
    temporary_token = False
    if not os.environ.get("HF_TOKEN"):
        try:
            from google.colab import userdata
            secret = userdata.get("HF_TOKEN")
        except Exception:
            secret = None
        if not secret:
            from getpass import getpass
            secret = getpass("Hugging Face READ token for your private Part 2 artifacts (hidden): ").strip()
        if not secret:
            raise ValueError("A read token is required. No private download, training or model evaluation was started.")
        os.environ["HF_TOKEN"] = secret
        del secret
        temporary_token = True
    try:
        run_script("tools/standalone_colab.py", "check-hub-access")
        run_script("part2_diffusion/src/prepare_data.py", "--output-dir", ROOT / "part2_diffusion/data/processed")
        run_script("tools/standalone_colab.py", "restore-part2")
        run_script("tools/standalone_colab.py", "record-part2")
        run_script("part3_agents/src/prepare_evaluation.py", "--download-adapter")
    finally:
        if temporary_token:
            os.environ.pop("HF_TOKEN", None)'''),
        md('''## Check the real model paths on this GPU

The RAG model, image understanding, prompt rewriting, critic, image editor and domain adapter are exercised sequentially. Tiny diagnostic images are not final samples. Stay connected until this prints `BOTH REAL GPU PREFLIGHTS PASSED`. CPU smoke output is never accepted as a passed GPU preflight.'''),
        code('''if SMOKE_ONLY:
    print("SKIPPED: actual GPU model preflights. No Colab GPU readiness is certified.")
else:
    run_script("tools/overnight_preflight.py", "--part", "part1_rag", "--data-dir", DATA)
    run_script("tools/overnight_preflight.py", "--part", "part3_agents")
    print("BOTH REAL GPU PREFLIGHTS PASSED. The full sequential run starts next.")'''),
        md('''## Run Part 1 and Part 3 and keep the executed notebooks

Part 1 produces the validated CSV and evaluates the dense baseline against the final system. Part 3 compares 15 scenarios with and without critique and ten safety cases for each. Each request keeps its own trace and image path. Models run sequentially; revisions are capped at two.

The embedded notebooks save after every cell under `output/executed`. Unchanged successful items resume after interruption while those files remain on the runtime. Errors stay visible and are not replaced by made-up successful results.'''),
        code('''run_script("tools/overnight_runner.py", *(["--smoke-only"] if SMOKE_ONLY else []))
print("Saved executed notebooks:", ROOT / "output/executed")'''),
        md('''## Check the results and pending review

A successful script exit is not enough: saved model errors and disabled-model fixtures fail the computation check. The separate audit lists real ratings, reference review, team details and other submission requirements that still need attention. Reports remain drafts while those are pending.'''),
        code('''run_script("tools/standalone_colab.py", "complete", *(["--smoke-only"] if SMOKE_ONLY else []))
run_script("tools/prepare_rating_forms.py")
run_script("tools/final_submission_audit.py")
print("Computation record:", ROOT / "runtime_checks/standalone_completion.json")
print("The final cell below downloads your saved results; it does not submit them.")''')])
    add_download_section(n)
    # A kernel restart may lose ROOT while its /content files still exist.
    # The export-only cell must still locate this exact bundle without setup.
    n.cells[-1].source = n.cells[-1].source.replace(
        'Path("/content/Lab-2"), Path("/kaggle/working/Lab-2")]',
        f'Path("/content/Lab2_Standalone_{digest[:16]}"), '
        f'Path.cwd() / "Lab2_Standalone_{digest[:16]}", '
        'Path(os.environ.get("LAB2_STANDALONE_ROOT", "/content/Lab-2"))]')
    n.cells[1].metadata = {'jupyter': {'source_hidden': True}, 'tags': ['embedded-source-bundle']}
    n.metadata = {'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'},
        'language_info': {'name': 'python'}, 'colab': {'provenance': [], 'gpuType': 'T4'}, 'accelerator': 'GPU',
        'standalone_bundle_sha256': digest, 'source_commit': manifest['source_commit'],
        'validation_scope': 'All source is embedded. Full pretrained-model GPU execution still requires the actual Colab run.'}
    nb.validate(n)
    for index, cell in enumerate(n.cells):
        if cell.cell_type == 'code':
            compile(cell.source, f'standalone-cell-{index}', 'exec')
    path = ROOT / 'part1_rag/notebooks/Lab2_Colab_Standalone.ipynb'
    nb.write(n, path)
    print('Created', path, 'bytes:', path.stat().st_size, 'embedded source files:', len(manifest['files']))
    return path


if __name__ == '__main__':
    build()
