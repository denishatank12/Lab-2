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
LORA_REPO = 'Shibin22/sd15-naruto-run1-lora-r4'
USE_HF_TOKEN = False  # The partner's adapter is public. No token is needed.
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
print('Seed: 266. Mode:', 'MOCK SOFTWARE CHECK ONLY' if SMOKE_ONLY else 'REAL BASELINE')
if not SMOKE_ONLY:
    device_info = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total',
        '--format=csv,noheader'], check=True, capture_output=True, text=True)
    print('NVIDIA GPU(s):', device_info.stdout.strip())

# Caches stay outside the result folder and are never included in the ZIP.
scratch = Path('/kaggle/temp' if Path('/kaggle').is_dir() else PARENT)
os.environ['HF_HOME'] = str(scratch / 'lab2_p3_hf_cache')
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN'] = '1'
os.environ['LAB2_P3_LORA_REPO'] = LORA_REPO
os.environ['LAB2_P3_USE_HF_TOKEN'] = '1' if USE_HF_TOKEN else '0'
os.environ['DOMAIN_NAME'] = 'Naruto'

def configure_hf_token(environ, secret_loader=None, hidden_prompt=None):
    # Never return, print, or save the credential. Child processes inherit it.
    from getpass import getpass, GetPassWarning
    import warnings
    token = (environ.get('HF_TOKEN') or '').strip()
    source = 'environment'
    if not token and secret_loader is not None:
        try:
            token = (secret_loader() or '').strip()
        except Exception:
            token = ''
        source = 'notebook secret'
    if not token:
        source = 'hidden prompt'
        try:
            # Refuse a terminal fallback that could echo the value as plain text.
            with warnings.catch_warnings():
                warnings.simplefilter('error', GetPassWarning)
                token = (hidden_prompt or getpass)('Hugging Face read token (hidden): ').strip()
        except Exception:
            raise RuntimeError('Hidden token entry is unavailable. Run this setup cell interactively, '
                'or enable HF_TOKEN in Kaggle Add-ons > Secrets before a background run.') from None
    if not token:
        raise RuntimeError('No token entered. Enable HF_TOKEN or rerun this cell and use the hidden input box.')
    environ['HF_TOKEN'] = token
    return source

def check_adapter_access(repo, token=None, opener=None):
    # A small metadata request before package/model downloads. Never substitute
    # a different adapter or a base model if this exact repo cannot be read.
    import json, re
    from urllib.error import HTTPError, URLError
    from urllib.parse import quote
    from urllib.request import Request, urlopen
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*', repo):
        raise ValueError('LORA_REPO must be an exact Hugging Face namespace/repository ID.')
    headers = {'User-Agent': 'Lab2-Part3-access-check'}
    if token:
        headers['Authorization'] = 'Bearer ' + token
    request = Request('https://huggingface.co/api/models/' + quote(repo, safe='/'), headers=headers)
    try:
        with (opener or urlopen)(request, timeout=30) as response:
            info = json.load(response)
    except HTTPError as error:
        if error.code in (401, 403, 404):
            raise RuntimeError('Cannot read adapter ' + repo + '. Confirm its ID and public visibility '
                'or request the exact adapter ZIP from its owner. No base-model fallback was used.') from None
        raise RuntimeError('Hugging Face access check returned HTTP ' + str(error.code) + '. Retry setup later.') from None
    except (URLError, TimeoutError):
        raise RuntimeError('Cannot reach Hugging Face. Enable Internet and retry this setup cell.') from None
    if not info.get('sha') or not any(f.get('rfilename', '').endswith('.safetensors') for f in info.get('siblings', [])):
        raise RuntimeError('The requested adapter has no verifiable revision/safetensors weights.')
    return {'repo': repo, 'revision': info['sha'], 'private': info.get('private', False),
            'authenticated': bool(token)}

if not SMOKE_ONLY and USE_HF_TOKEN:
    def notebook_secret():
        if Path('/kaggle').is_dir():
            from kaggle_secrets import UserSecretsClient
            return UserSecretsClient().get_secret('HF_TOKEN')
        from google.colab import userdata
        return userdata.get('HF_TOKEN')
    token_source = configure_hf_token(os.environ, secret_loader=notebook_secret)
    print('Hugging Face credential loaded through:', token_source, '(value not displayed or saved)')
if not SMOKE_ONLY:
    access = check_adapter_access(LORA_REPO, os.environ.get('HF_TOKEN') if USE_HF_TOKEN else None)
    (ROOT / 'adapter_access.json').write_text(json.dumps(access, indent=2))
    print('Adapter access passed:', access['repo'], access['revision'], '(no token required)' if not USE_HF_TOKEN else '')

def prepare_python_environment(folder):
    # Kaggle's Debian Python can lack a working ensurepip. Do not invoke it.
    from pathlib import Path
    import json, os, subprocess, sys, venv
    folder = Path(folder).absolute()
    if folder.is_symlink():
        raise RuntimeError('The environment directory must not be a symlink.')
    python = folder / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    probe_code = ('import json,sys,sysconfig; print(json.dumps(dict(prefix=sys.prefix, '
                  'base=sys.base_prefix, version=list(sys.version_info[:2]), '
                  'purelib=sysconfig.get_path("purelib"))))')
    def inspect():
        try:
            result = subprocess.run([str(python), '-c', probe_code], check=True,
                capture_output=True, text=True, timeout=30)
            return json.loads(result.stdout)
        except (OSError, subprocess.SubprocessError, ValueError):
            return {}
    info = inspect() if python.is_file() else {}
    cfg = folder / 'pyvenv.cfg'
    includes_host = cfg.is_file() and any(line.strip().lower() == 'include-system-site-packages = true'
        for line in cfg.read_text().splitlines())
    expected_version = list(sys.version_info[:2])
    if includes_host or list(folder.glob('lib/python*/site-packages/_lab2_host_packages.pth')):
        raise RuntimeError('An older host-inheriting environment was preserved. Use the new clean environment path.')
    if not info or not cfg.is_file() or info.get('version') != expected_version:
        # Rewrites only venv scaffolding; keeps installed packages and all run outputs.
        # This also repairs the partial folder left behind by a failed ensurepip.
        venv.EnvBuilder(with_pip=False, system_site_packages=False).create(folder)
        info = inspect()
    if (not info or Path(info['prefix']).resolve() != folder.resolve()
            or info['prefix'] == info['base'] or info['version'] != expected_version):
        raise RuntimeError('The isolated interpreter failed its identity check; refusing to install into the host.')
    return python

def dependency_profile(smoke_only=False):
    import sys
    pins = ['pip==26.3', 'transformers==5.22.2', 'accelerate==1.15.0',
        'diffusers==0.43.0', 'peft==0.21.2', 'pydantic==2.14.3', 'qwen-vl-utils==0.0.14',
        'datasets==5.2.0', 'numpy==2.2.6', 'pandas==2.3.3', 'pyarrow==26.0.1',
        'Pillow==12.3.0', 'sentencepiece==0.2.1', 'nbformat==5.11.1', 'nbclient==0.10.4',
        'ipykernel==7.4.0', 'matplotlib-inline==0.2.2', 'matplotlib==3.11.2',
        'huggingface-hub==1.32.0', 'safetensors==0.8.0', 'tokenizers==0.23.1', 'requests==2.34.2']
    if smoke_only and sys.platform == 'darwin':
        return {'pins': pins + ['torch==2.11.0', 'torchvision==0.26.0'], 'extra_index': None, 'cuda': False}
    if smoke_only:
        return {'pins': pins + ['torch==2.11.0+cpu', 'torchvision==0.26.0+cpu'],
                'extra_index': 'https://download.pytorch.org/whl/cpu', 'cuda': False}
    return {'pins': pins + ['torch==2.11.0+cu128', 'torchvision==0.26.0+cu128', 'bitsandbytes==0.50.2'],
            'extra_index': 'https://download.pytorch.org/whl/cu128', 'cuda': True}

PROFILE = dependency_profile(SMOKE_ONLY)
PINS = PROFILE['pins']
environment_key = hashlib.sha256(json.dumps(PROFILE, sort_keys=True).encode()).hexdigest()[:12]
ENV = scratch / ('lab2_p3_clean_py' + str(sys.version_info.major) + str(sys.version_info.minor) + '_' + environment_key)
PYTHON = prepare_python_environment(ENV)
# Host pip's supported --python mode installs into a target even when it has
# no pip of its own. No ensurepip, get-pip script, sudo, or live-kernel upgrade.
pip_environment = os.environ.copy()
for option in ['PIP_TARGET', 'PIP_PREFIX', 'PIP_USER', 'PIP_PYTHON', 'PYTHONPATH', 'PYTHONHOME', 'PYTHONUSERBASE']:
    pip_environment.pop(option, None)
pip_environment.update(PYTHONNOUSERSITE='1', PIP_CONFIG_FILE=os.devnull, PIP_PROGRESS_BAR='off', NO_COLOR='1')
command = [sys.executable, '-m', 'pip', '--python', str(PYTHON), 'install', '--no-cache-dir',
    '--only-binary=:all:', '--disable-pip-version-check', '--progress-bar', 'off',
    '--report', str(ROOT / 'install_report.json'), '--index-url', 'https://pypi.org/simple']
if PROFILE['extra_index']:
    command += ['--extra-index-url', PROFILE['extra_index']]
subprocess.run(command + PINS, check=True, env=pip_environment)
subprocess.run([str(PYTHON), '-I', '-m', 'pip', 'check'], check=True, env=pip_environment)
runtime_check = r''' + "'''" + '''import importlib.metadata as meta, importlib.util, json, pathlib, sys
import torch, torchvision, transformers, diffusers, peft, pandas, PIL, datasets
from transformers import AutoProcessor, AutoModelForImageTextToText, CLIPModel, CLIPProcessor
from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionInstructPix2PixPipeline
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
prefix = pathlib.Path(sys.prefix).resolve()
foreign = [p for p in sys.path if ('site-packages' in p or 'dist-packages' in p)
           and not pathlib.Path(p).resolve().is_relative_to(prefix)]
assert not foreign, ('Host package paths leaked into the clean environment', foreign)
for unwanted in ['google-colab', 'bigframes', 'cudf-cu13', 'nbconvert', 'jupyter-kernel-gateway', 'torchao']:
    try:
        meta.version(unwanted)
    except meta.PackageNotFoundError:
        continue
    raise AssertionError('Unexpected host package: ' + unwanted)
assert torchvision.ops.nms(torch.tensor([[0.,0.,1.,1.]]), torch.tensor([.5]), .5).tolist() == [0]
needs_cuda = sys.argv[2] == '1'
if needs_cuda:
    import bitsandbytes as bnb
    assert torch.cuda.is_available(), 'CUDA is unavailable in the clean environment'
    layer = bnb.nn.Linear4bit(32, 16, bias=False, compute_dtype=torch.float16, quant_type='nf4').to('cuda:0')
    with torch.inference_mode():
        value = layer(torch.randn(1,32,device='cuda:0',dtype=torch.float16))
    torch.cuda.synchronize()
    assert torch.isfinite(value).all()
packages = {d.metadata['Name']: d.version for d in meta.distributions()}
record = dict(prefix=str(prefix), python=sys.version, torch=torch.__version__, torchvision=torchvision.__version__,
    cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0) if needs_cuda else 'CPU smoke only',
    foreign_site_paths=foreign, dependency_check='passed', cuda_4bit_smoke=needs_cuda, packages=packages)
pathlib.Path(sys.argv[1]).write_text(json.dumps(record, indent=2))
print('CLEAN IMPORTS:', torch.__version__, torchvision.__version__, transformers.__version__, diffusers.__version__)
print('DEPENDENCY/ISOLATION CHECKS PASSED. CUDA 4-bit check:', needs_cuda)
''' + "'''" + '''
subprocess.run([str(PYTHON), '-I', '-c', runtime_check, str(ROOT / 'runtime_environment.json'),
    '1' if PROFILE['cuda'] else '0'], check=True, env=pip_environment)
resolved = subprocess.check_output([str(PYTHON), '-I', '-m', 'pip', 'freeze', '--all'],
    text=True, env=pip_environment)
(ROOT / 'requirements.resolved.txt').write_text(resolved)

def run_baseline(action):
    log = ROOT / ('RUN_LOG_' + action + '.txt')
    command = [str(PYTHON), '-I', '-u', str(ROOT / 'baseline_runtime.py'), action]
    with log.open('a', encoding='utf-8') as handle:
        process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding='utf-8', errors='replace', env=pip_environment)
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

Before Run All, use a fresh Kaggle GPU session and enable Internet. **No Hugging Face token is required:** Shibin made `Shibin22/sd15-naruto-run1-lora-r4` public, and setup verifies access before large downloads. Leave `USE_HF_TOKEN = False`; an old token in the runtime is ignored. Optional private-repository use can explicitly enable secure secret/hidden input, never a hard-coded token. Keep course outputs private. The code uses GPU 0 only, with diffusion CPU offload; two T4 cards do not combine their memory. Colab GPU sessions are also supported.

The baseline uses Qwen3-VL-8B in 4-bit, SD 1.5 plus the real private LoRA, InstructPix2Pix, CLIP and an image safety checker. It runs 15 scenarios and 10 red-team cases. The revision loop is disabled, but the partner's diagnostic critic and safety checks remain. The run cannot establish human task success; both members still need to inspect and rate the outputs.

The embedded partner source is unchanged. The runner adds strict real-photo downloads, exact model revision records, missing-adapter checks, fail-closed safety and per-case resume. These declared runtime differences mean this is not a byte-identical reproduction of the partner's measured environment. Their existing run used an RTX 4090; a full 16-GB run remains to be measured.

No repository clone or separate Python uploads are needed. Internal helper files are created automatically. Setup bypasses `ensurepip` and creates a **clean environment without host packages**. It installs the matching PyTorch 2.11.0 / torchvision 0.26.0 CUDA 12.8 pair, checks dependencies and isolation, and tests CUDA 4-bit execution before downloading models. Kaggle/Colab packages, cuDF and old torchao cannot leak into it. Earlier environments and result folders are preserved. CPU smoke mode uses a separate CPU package profile and never claims GPU results."""),
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
