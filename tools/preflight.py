"""Check real imports and CUDA kernels before a long training run."""
import argparse
import importlib.metadata
import json
import platform
from pathlib import Path
import random
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    import numpy as np
    import torch
    import torchvision
    import diffusers
    import peft
    import datasets
    import transformers
    from diffusers import StableDiffusionImg2ImgPipeline
    from torchvision.ops import nms
    from torchmetrics.image.kid import KernelInceptionDistance
    random.seed(266)
    np.random.seed(266)
    torch.manual_seed(266)
    root = Path(__file__).resolve().parents[1]
    info = {"commit": subprocess.run(["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
                                    capture_output=True, text=True).stdout.strip(),
            "python": platform.python_version(), "seed": 266, "executable": sys.executable,
            "versions": {p: importlib.metadata.version(p) for p in
                         ["torch", "torchvision", "diffusers", "transformers", "peft", "datasets", "huggingface-hub"]},
            "cuda_available": torch.cuda.is_available(), "gpu": "cpu"}
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Install the NVIDIA driver and run setup_runtime.py --device cuda. "
                           "CPU-only torch or a disabled notebook GPU cannot run the full Part 2/3 pipeline.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    nms(torch.tensor([[0., 0., 1., 1.]], device=device), torch.ones(1, device=device), .5)
    if device == "cuda":
        torch.cuda.manual_seed_all(266)
        x = torch.randn(32, 32, device=device, dtype=torch.float16)
        assert torch.isfinite(x @ x.T).all()
        info["gpu"] = torch.cuda.get_device_name(0)
        info["vram_gib"] = torch.cuda.get_device_properties(0).total_memory / 2**30
        if info["vram_gib"] < 5.4:
            raise RuntimeError("This 512px profile needs approximately 6 GB VRAM; use a 16 GB T4 instead.")
    print(json.dumps(info, indent=2), flush=True)
    print("PASS: core imports, matching torchvision kernels and device preflight")


if __name__ == "__main__":
    main()
