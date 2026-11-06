"""Install once under an OS lock, without modifying an environment in use."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
import platform
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
import venv

ROOT = Path(__file__).resolve().parents[1]
SETUP_SCHEMA = 2


def python_in(folder):
    return folder / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


@contextmanager
def installation_lock(path, timeout=1800):
    """The OS releases this lock even if an installer is interrupted or killed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        handle.seek(0)
        handle.write(b"0")
        handle.flush()
        started, announced = time.monotonic(), False
        while True:
            try:
                handle.seek(0)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if not announced:
                    print("Another notebook is setting up this repository. Waiting safely for its installer.", flush=True)
                    announced = True
                if time.monotonic() - started > timeout:
                    raise TimeoutError("Setup lock timed out. Let the other setup finish, then rerun this cell.")
                time.sleep(2)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def atomic_json(path, value):
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def healthy(interpreter, cuda):
    if not interpreter.is_file():
        return False
    # Metadata alone misses the broken SciPy extension seen in the uploaded run.
    check = '''import importlib.metadata as m, sys
import numpy, scipy.stats, scipy.linalg, PIL.Image, torch, torchvision
assert (3,10) <= sys.version_info[:2] <= (3,12)
for line in open(sys.argv[1], encoding="utf-8"):
    line=line.strip()
    if not line or line.startswith("#"): continue
    name, expected=line.split("==")
    assert m.version(name).split("+")[0] == expected, name
assert not int(sys.argv[2]) or torch.cuda.is_available()
'''
    try:
        subprocess.run([str(interpreter), "-c", check, str(ROOT / "requirements.txt"), str(int(cuda))],
                       check=True, capture_output=True, text=True, timeout=120)
        subprocess.run([str(interpreter), "-m", "pip", "check"], check=True,
                       capture_output=True, text=True, timeout=90)
        return True
    except (OSError, subprocess.SubprocessError):
        return False


def create_environment(folder):
    if (3, 10) <= sys.version_info[:2] <= (3, 12):
        venv.EnvBuilder(with_pip=True).create(folder)
    else:
        bootstrap = ROOT / ".bootstrap"
        if not python_in(bootstrap).exists():
            venv.EnvBuilder(with_pip=True).create(bootstrap)
        subprocess.run([str(python_in(bootstrap)), "-m", "pip", "install", "uv==0.5.31"], check=True)
        subprocess.run([str(python_in(bootstrap)), "-m", "uv", "venv", "--python", "3.12",
                        "--seed", str(folder)], check=True)


def setup(cuda):
    requirements = ROOT / "requirements.txt"
    digest = hashlib.sha256(requirements.read_bytes() + str(cuda).encode()).hexdigest()
    pointer = ROOT / (".lab2_runtime_cuda.json" if cuda else ".lab2_runtime_cpu.json")
    candidates = []
    if pointer.exists():
        try:
            candidates.append(Path(json.loads(pointer.read_text())["python"]))
        except (ValueError, KeyError):
            pass
    # Reuse healthy environments read-only. Never patch an old damaged one in place.
    candidates += [python_in(ROOT / ".venv"), python_in(ROOT / ".lab2-python312")]
    interpreter = next((p for p in candidates if healthy(p, cuda)), None)
    if interpreter is None:
        minimum_gib = 12 if cuda else 4
        free_gib = shutil.disk_usage(ROOT).free / 2**30
        if free_gib < minimum_gib:
            raise RuntimeError(f"A clean runtime needs at least {minimum_gib} GiB free; found {free_gib:.1f}. "
                               "Free disk space first. Existing environments and results were not deleted.")
        base = ROOT / ".lab2-envs"
        base.mkdir(exist_ok=True)
        # Console scripts embed absolute paths, so do not rename a populated venv.
        environment = Path(tempfile.mkdtemp(prefix=("cuda-" if cuda else "cpu-") + digest[:10] + "-", dir=base))
        create_environment(environment)
        interpreter = python_in(environment)
        env = dict(os.environ, PIP_DISABLE_PIP_VERSION_CHECK="1", PIP_PROGRESS_BAR="off")
        subprocess.run([str(interpreter), "-m", "pip", "install", "pip==24.3.1"], check=True, env=env)
        command = [str(interpreter), "-m", "pip", "install", "--retries", "5", "--timeout", "120",
                   "--no-cache-dir", "-r", str(requirements)]
        if cuda:
            command += ["torch==2.5.1+cu121", "torchvision==0.20.1+cu121",
                        "--extra-index-url", "https://download.pytorch.org/whl/cu121"]
        # Resolve pinned NumPy/Pillow with torch, rather than downgrade them later.
        try:
            subprocess.run(command, check=True, env=env)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"Clean installation failed. Partial files are isolated at {environment}; "
                               "no existing runtime or results were changed. Check disk/network, then rerun.") from exc
        if not healthy(interpreter, cuda):
            raise RuntimeError("Installed runtime failed its integrity check and was not selected for use.")
    subprocess.run([str(interpreter), str(ROOT / "tools/preflight.py"),
                    *(["--require-cuda"] if cuda else [])], check=True)
    result = {"python": str(interpreter), "cuda_requested": cuda, "requirements_sha256": digest}
    # Only a completely installed and tested environment gets a ready marker.
    atomic_json(pointer, result)
    atomic_json(ROOT / ".lab2_runtime.json", result)
    print("LAB2_PYTHON=" + str(interpreter), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    cuda = args.device == "cuda" or (args.device == "auto" and shutil.which("nvidia-smi") is not None)
    if cuda and platform.system() == "Darwin":
        raise RuntimeError("CUDA is unavailable on macOS. Use --device cpu for smoke tests.")
    with installation_lock(ROOT / ".lab2-setup.lock"):
        setup(cuda)


if __name__ == "__main__":
    main()
