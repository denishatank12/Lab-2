"""Create an isolated runtime without replacing the notebook kernel's torch."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import platform
from pathlib import Path
import shutil
import subprocess
import sys
import venv

ROOT = Path(__file__).resolve().parents[1]


def python_in(folder):
    return folder / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    cuda = args.device == "cuda" or (args.device == "auto" and shutil.which("nvidia-smi") is not None)
    if cuda and platform.system() == "Darwin":
        raise RuntimeError("CUDA is not available on macOS. Use --device cpu for smoke tests or run on your NVIDIA PC.")
    environment = ROOT / ".venv"
    existing = python_in(environment)
    if existing.exists():
        try:
            version = json.loads(subprocess.check_output([str(existing), '-c',
                'import sys,json; print(json.dumps(list(sys.version_info[:2])))'], text=True))
            if not [3,10] <= version <= [3,12]:
                environment = ROOT / '.lab2-python312'
        except (OSError,subprocess.CalledProcessError,ValueError):
            environment = ROOT / '.lab2-python312'
    elif environment.exists():
        environment = ROOT / '.lab2-python312'
    interpreter = python_in(environment)
    if not interpreter.exists():
        if (3, 10) <= sys.version_info[:2] <= (3, 12):
            venv.EnvBuilder(with_pip=True).create(environment)
        else:
            bootstrap = ROOT / ".bootstrap"
            if not python_in(bootstrap).exists():
                venv.EnvBuilder(with_pip=True).create(bootstrap)
            subprocess.run([str(python_in(bootstrap)), "-m", "pip", "install", "uv==0.5.31"], check=True)
            subprocess.run([str(python_in(bootstrap)), "-m", "uv", "venv", "--python", "3.12",
                            "--seed", str(environment)], check=True)
    requirements = ROOT / "requirements.txt"
    digest = hashlib.sha256(requirements.read_bytes() + str(cuda).encode() + Path(__file__).read_bytes()).hexdigest()
    stamp = environment / "lab2_runtime.json"
    if not stamp.exists() or json.loads(stamp.read_text()).get("requirements_sha256") != digest:
        subprocess.run([str(interpreter), "-m", "pip", "install", "pip==24.3.1"], check=True)
        index = "https://download.pytorch.org/whl/cu121" if cuda else "https://pypi.org/simple"
        torch_pin = "torch==2.5.1+cu121" if cuda else "torch==2.5.1"
        vision_pin = "torchvision==0.20.1+cu121" if cuda else "torchvision==0.20.1"
        subprocess.run([str(interpreter), "-m", "pip", "install", torch_pin, vision_pin,
                        "--index-url", index], check=True)
        subprocess.run([str(interpreter), "-m", "pip", "install", "-r", str(requirements)], check=True)
        subprocess.run([str(interpreter), "-m", "pip", "check"], check=True)
        stamp.write_text(json.dumps({"requirements_sha256": digest, "requested_cuda": cuda}, indent=2))
    subprocess.run([str(interpreter), str(ROOT / "tools/preflight.py"),
                    *(["--require-cuda"] if cuda else [])], check=True)
    (ROOT/'.lab2_runtime.json').write_text(json.dumps({'python':str(interpreter),'cuda_requested':cuda}))
    print("LAB2_PYTHON=" + str(interpreter), flush=True)


if __name__ == "__main__":
    main()
