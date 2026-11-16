"""Package the attributed executed source and baseline runner into one notebook."""
import base64
import hashlib
import io
import json
from pathlib import Path
import subprocess
import zipfile

import nbformat as nb

ROOT = Path(__file__).resolve().parents[1]


def payload():
    snapshot = ROOT / "part3_agents/partner_snapshot"
    files = {str(p.relative_to(snapshot)): p.read_bytes() for p in (snapshot / "source").rglob("*") if p.is_file()
             and "__pycache__" not in p.parts}
    files["PROVENANCE.json"] = (snapshot / "PROVENANCE.json").read_bytes()
    files["baseline_runtime.py"] = (ROOT / "tools/part3_partner_baseline.py").read_bytes()
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, content in sorted(files.items()):
            info = zipfile.ZipInfo(name, (2026, 11, 16, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, content)
    return buffer.getvalue()


def build():
    raw = payload()
    digest = hashlib.sha256(raw).hexdigest()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    md, code = nb.v4.new_markdown_cell, nb.v4.new_code_cell
    setup = '''from pathlib import Path
import base64, hashlib, io, json, os, re, shutil, subprocess, sys, venv, zipfile

SMOKE_ONLY = False  # True checks software with fictional images, not a real baseline.
BUNDLE_SHA256 = __DIGEST__
BUILD_SOURCE_COMMIT = __COMMIT__
PAYLOAD = __PAYLOAD__
packed = base64.b64decode(PAYLOAD)
assert hashlib.sha256(packed).hexdigest() == BUNDLE_SHA256
PARENT = Path('/kaggle/working' if Path('/kaggle/working').is_dir() else '/content' if Path('/content').is_dir() else Path.cwd())
ROOT = PARENT / ('Lab2_Part3_Baseline_' + BUNDLE_SHA256[:16])
ROOT.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(io.BytesIO(packed)) as bundle:
    for item in bundle.infolist():
        relative = Path(item.filename)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unsafe embedded path')
        path = ROOT / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        content = bundle.read(item)
        if path.exists() and path.read_bytes() != content:
            raise RuntimeError('An embedded source file changed. Keep the previous results and use a fresh output folder.')
        path.write_bytes(content)
print('Notebook source commit:', BUILD_SOURCE_COMMIT)
print('Partner source commit:', json.loads((ROOT / 'PROVENANCE.json').read_text())['commit'])
print('Embedded bundle SHA256:', BUNDLE_SHA256)
print('Python:', sys.version)
import torch
print('Torch:', torch.__version__, 'CUDA:', torch.version.cuda)
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No CUDA GPU')
print('Seed: 266. Mode:', 'MOCK SOFTWARE CHECK ONLY' if SMOKE_ONLY else 'REAL BASELINE')
if not SMOKE_ONLY and not torch.cuda.is_available():
    raise RuntimeError('Select a GPU before running. No CPU or mock substitution will be made.')

# Caches stay outside the result folder and are never included in the ZIP.
scratch = Path('/kaggle/temp' if Path('/kaggle').is_dir() else PARENT)
os.environ['HF_HOME'] = str(scratch / 'lab2_p3_hf_cache')
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['PYTHONUNBUFFERED'] = '1'
if not SMOKE_ONLY:
    secret = os.environ.get('HF_TOKEN', '').strip()
    if not secret and Path('/kaggle').is_dir():
        try:
            from kaggle_secrets import UserSecretsClient
            secret = UserSecretsClient().get_secret('HF_TOKEN').strip()
        except Exception:
            raise RuntimeError('In Add-ons > Secrets, enable HF_TOKEN with access to the partner adapter. Do not paste it into code.') from None
    elif not secret:
        try:
            from google.colab import userdata
            secret = userdata.get('HF_TOKEN').strip()
        except Exception:
            from getpass import getpass
            secret = getpass('Hugging Face read token (hidden): ').strip()
    if not secret:
        raise RuntimeError('A Hugging Face read token is required for the private partner adapter.')
    os.environ['HF_TOKEN'] = secret
    del secret

# An isolated site-packages environment preserves the platform's CUDA torch.
ENV = scratch / 'lab2_p3_baseline_env'
PYTHON = ENV / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
if not PYTHON.is_file():
    venv.EnvBuilder(with_pip=True, system_site_packages=True).create(ENV)
PINS = ['transformers==5.22.2', 'accelerate==1.15.0', 'bitsandbytes==0.50.2',
        'diffusers==0.43.0', 'peft==0.21.2', 'pydantic==2.14.3', 'qwen-vl-utils==0.0.14',
        'datasets==5.2.0', 'pandas>=2.2,<4', 'Pillow>=11,<13', 'sentencepiece',
        'nbformat>=5.10,<6', 'nbclient>=0.10,<0.11', 'ipykernel>=6.29,<8', 'matplotlib-inline']
subprocess.run([str(PYTHON), '-m', 'pip', 'install', '--disable-pip-version-check', *PINS], check=True)
subprocess.run([str(PYTHON), '-c', 'import torch, transformers, diffusers, peft, bitsandbytes, pandas, PIL; '
    'print("ISOLATED IMPORTS:", torch.__version__, transformers.__version__, diffusers.__version__); '
    + ('' if SMOKE_ONLY else 'assert torch.cuda.is_available(), "Isolated environment lost CUDA access"')], check=True)

def run_baseline(action):
    log = ROOT / ('RUN_LOG_' + action + '.txt')
    command = [str(PYTHON), '-u', str(ROOT / 'baseline_runtime.py'), action]
    with log.open('a', encoding='utf-8') as handle:
        process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding='utf-8', errors='replace')
        for line in process.stdout:
            line = re.sub(r'\\bhf_[A-Za-z0-9]+\\b', '[TOKEN REDACTED]', line)
            print(line, end='', flush=True)
            handle.write(line)
            handle.flush()
        status = process.wait()
    if status:
        raise RuntimeError(f'Baseline stopped. Genuine partial results and error notebook are retained in {ROOT}. Read {log}; run the export cell below.')
'''
    setup = setup.replace("__DIGEST__", repr(digest)).replace("__COMMIT__", repr(commit)).replace(
        "__PAYLOAD__", repr(base64.b64encode(raw).decode()))
    notebook = nb.v4.new_notebook(cells=[
        md("""# Lab 2 Part 3 baseline only

Name: Denisha Ketan Tank. This notebook runs my own baseline using my partner Shibin's executed agent implementation, credited in the embedded provenance. It does not retrain Part 2 or repeat my partner's final critique-loop experiment.

Before Run All, use a fresh Kaggle GPU session, enable Internet, and enable **HF_TOKEN** under Add-ons > Secrets. The token must read `Shibin22/sd15-naruto-run1-lora-r4`. Keep the notebook and outputs private. The code uses GPU 0 only, with diffusion CPU offload; two T4 cards do not combine their memory. Colab GPU sessions are also supported.

The baseline uses Qwen3-VL-8B in 4-bit, SD 1.5 plus the real private LoRA, InstructPix2Pix, CLIP and an image safety checker. It runs 15 scenarios and 10 red-team cases. The revision loop is disabled, but the partner's diagnostic critic and safety checks remain. The run cannot establish human task success; both members still need to inspect and rate the outputs.

The embedded partner source is unchanged. The runner adds strict real-photo downloads, exact model revision records, missing-adapter checks, fail-closed safety and per-case resume. These declared runtime differences mean this is not a byte-identical reproduction of the partner's measured environment. Their existing run used an RTX 4090; a full 16-GB run remains to be measured.

No repository clone or separate Python uploads are needed. Internal helper files are created automatically."""),
        code(setup),
        md("""## Check the baseline control flow

These are fictional-image software checks, not scores. In particular, a low mock critic score must never trigger a second generation in this baseline."""),
        code("run_baseline('smoke')"),
        md("""## Run and keep the executed notebook

Model access is checked before large downloads. Real photos are required; download errors never become placeholder evaluation results. Results and JSON traces are saved after every case. A rerun in the same session reuses completed cases only when the source, inputs and settings match. Do not delete the run folder after an interruption.

The automatic pass rate is a routing/file/text proxy, not a human image-quality score. Expected corrupt-image errors count as valid red-team tests, not model runtime failures."""),
        code("run_baseline('smoke-notebook' if SMOKE_ONLY else 'notebook')\n"
             "metrics = ROOT / ('smoke_outputs/SMOKE_ONLY.json' if SMOKE_ONLY else 'baseline_outputs/baseline_metrics.json')\n"
             "print(metrics.read_text())"),
        md("""## Download your results

Download the newest `Lab2_Part3_Baseline_*.zip` from the Kaggle Output panel under `/kaggle/working`. It contains the genuine executed baseline notebook, metrics, images, per-request traces, input/model manifests, source and blank human-rating form. If a GPU step fails, run this export-only cell to save partial work. It does not repeat inference. On Colab, a browser download is offered.

Return the ZIP for review. Do not replace my partner's results or my Part 2 outputs with this baseline's files. Team ratings, permissions, report integration and external submission remain separate."""),
        code("run_baseline('export')\n"
             "archives = sorted(PARENT.glob('Lab2_Part3_Baseline_*.zip'), key=lambda p: p.stat().st_mtime)\n"
             "print('Download from Output:', archives[-1].name)\n"
             "if not Path('/kaggle').is_dir():\n"
             "    try:\n"
             "        from google.colab import files\n"
             "        files.download(str(archives[-1]))\n"
             "    except ImportError:\n"
             "        print('Saved locally:', archives[-1])")])
    notebook.metadata.update(kernelspec={"display_name": "Python 3", "language": "python", "name": "python3"},
        language_info={"name": "python", "version": "3.12"},
        baseline_bundle_sha256=digest, source_build_commit=commit,
        partner_commit="a79529d1d59a6efb05bf0524ce6d60bda9bf6c09")
    path = ROOT / "part3_agents/notebooks/Part3_Partner_Baseline_Standalone.ipynb"
    nb.write(notebook, path)
    assert path.stat().st_size < 900_000, "Keep the source below Kaggle's 1 MB limit."
    print(path, path.stat().st_size, "bytes")
    return path


if __name__ == "__main__":
    build()
