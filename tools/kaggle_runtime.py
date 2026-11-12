"""Kaggle-specific secrets, temporary caches and persistent result delivery."""
import hashlib
import os
from pathlib import Path
import shutil
import zipfile


def configure_runtime(working=None, scratch=None, environ=None, secret_loader=None, smoke_only=False):
    """Configure before ML imports. Never prompt for a token during a batch run."""
    if smoke_only:
        return {"mode": "CPU smoke only; no Kaggle GPU execution is certified"}
    environ = os.environ if environ is None else environ
    working = Path(working or "/kaggle/working").resolve()
    scratch = Path(scratch or "/kaggle/temp/lab2-runtime").resolve()
    if not working.is_dir():
        raise RuntimeError("Import this notebook into Kaggle, select a GPU and enable Internet before running it.")
    if scratch == working or working in scratch.parents:
        raise ValueError("Large caches and environments must be outside Kaggle's saved output directory.")
    token = environ.get("HF_TOKEN", "").strip()
    if not token:
        try:
            if secret_loader is None:
                from kaggle_secrets import UserSecretsClient
                secret_loader = UserSecretsClient().get_secret
            token = secret_loader("HF_TOKEN").strip()
        except Exception:
            raise RuntimeError("In Kaggle, open Add-ons > Secrets, add HF_TOKEN with your Hugging Face read token, "
                               "enable it for this notebook, then rerun. Never paste the token into a cell.") from None
    if not token:
        raise RuntimeError("The HF_TOKEN secret is empty. Set a read token with access to your private Part 2 artifacts.")
    scratch.mkdir(parents=True, exist_ok=True)
    environ["HF_TOKEN"] = token
    # A T4 x2 runtime still uses one card. Two devices are not one 32-GB GPU.
    visible = environ.get("CUDA_VISIBLE_DEVICES", "0") or "0"
    environ["CUDA_VISIBLE_DEVICES"] = visible.split(",")[0]
    paths = {
        "HF_HOME": scratch / "huggingface",
        "HF_HUB_CACHE": scratch / "huggingface/hub",
        "HF_DATASETS_CACHE": scratch / "datasets",
        "TORCH_HOME": scratch / "torch",
        "XDG_CACHE_HOME": scratch / "cache",
        "UV_CACHE_DIR": scratch / "uv-cache",
        "UV_PYTHON_INSTALL_DIR": scratch / "uv-python",
        "LAB2_ENV_DIR": scratch / "environments",
        "LAB2_BOOTSTRAP_DIR": scratch / "bootstrap",
    }
    for name, path in paths.items():
        environ[name] = str(path)
    environ["LAB2_KAGGLE_WORKING"] = str(working)
    environ["LAB2_STANDALONE_PARENT"] = str(working)
    # Return paths and hardware selection, never the secret.
    return {"working": str(working), "temporary_cache": str(scratch),
            "visible_gpu": environ["CUDA_VISIBLE_DEVICES"], "secret_loaded": True}


def publish_zip(archive, working=None, display_link=True):
    """Keep the exact validated archive at the top of Kaggle's Output panel."""
    archive = Path(archive).resolve()
    working = Path(working or "/kaggle/working").resolve()
    if not working.is_dir() or not archive.is_file() or archive.suffix.lower() != ".zip":
        raise FileNotFoundError("A complete result ZIP and an existing output directory are required.")
    with zipfile.ZipFile(archive) as bundle:
        if bundle.testzip() is not None:
            raise ValueError("Result ZIP failed its integrity check; it was not published.")
    def checksum(path):
        value = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(2**20), b""):
                value.update(block)
        return value.hexdigest()
    destination = working / archive.name
    expected = checksum(archive)
    if destination.is_symlink():
        raise FileExistsError("The output destination is a symlink and was preserved.")
    if destination != archive:
        if destination.exists():
            if checksum(destination) != expected:
                raise FileExistsError("A different same-named result ZIP exists and was preserved.")
        else:
            if shutil.disk_usage(working).free < archive.stat().st_size + 64 * 2**20:
                raise RuntimeError("Not enough space for the Output-panel copy. The original ZIP is preserved.")
            shutil.copy2(archive, destination)
    if checksum(destination) != expected:
        raise ValueError("Output copy checksum mismatch. Keep the original ZIP for recovery.")
    print("KAGGLE OUTPUT ZIP:", destination.name)
    print("SHA256:", expected)
    print("Download this file from Kaggle's Output panel. Save the completed version with outputs.")
    if display_link:
        from IPython.display import FileLink, display
        # Kaggle's normal notebook cwd is /kaggle/working. The Output panel is
        # the reliable fallback if a frontend does not render a file link.
        if Path.cwd().resolve() == working:
            display(FileLink(destination.name))
    return destination
