# Run Parts 1 and 3 on a Windows GPU machine from PowerShell (no Jupyter UI needed).
# Notebooks are executed headless with nbconvert and saved WITH their outputs.
#
#   powershell -ExecutionPolicy Bypass -File .\run_lab.ps1 setup   # venv + CUDA torch + packages, checks the GPU
#   powershell -ExecutionPolicy Bypass -File .\run_lab.ps1 check   # data files / golden set in place?
#   powershell -ExecutionPolicy Bypass -File .\run_lab.ps1 test    # unit tests (1-2 min)
#   powershell -ExecutionPolicy Bypass -File .\run_lab.ps1 p1      # Part 1 notebook -> submission.csv + results
#   powershell -ExecutionPolicy Bypass -File .\run_lab.ps1 p2      # Part 2 notebook: data, two LoRA runs, generation, evaluation
#   powershell -ExecutionPolicy Bypass -File .\run_lab.ps1 p3      # Part 3 notebook (set $env:LORA_REPO first)
#   powershell -ExecutionPolicy Bypass -File .\run_lab.ps1 app     # Gradio demo at http://localhost:7860
#
# Options (set before running, e.g.  $env:LORA_REPO = "user/repo"):
#   $env:GPU          which GPU (default 0)
#   $env:TORCH_INDEX  PyTorch CUDA wheel index (default https://download.pytorch.org/whl/cu128)
#   $env:PYTHON       Python to build the venv with (default: py -3.12, then py -3.13, then python)
#   $env:HF_HOME      model cache folder (~20 GB)
#   $env:LORA_REPO    Part 2 LoRA on Hugging Face;  $env:DOMAIN_NAME = Naruto or Pokemon
param([string]$Cmd = "help")

$ErrorActionPreference = "Continue"   # native tools write progress to stderr; we check exit codes instead
Set-Location $PSScriptRoot
$Root = $PSScriptRoot
$env:PYTHONUTF8 = "1"                 # UTF-8 file I/O on Windows
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
if ($env:GPU) { $env:CUDA_VISIBLE_DEVICES = $env:GPU } else { $env:CUDA_VISIBLE_DEVICES = "0" }
$Py = Join-Path $Root ".venv\Scripts\python.exe"
New-Item -ItemType Directory -Force -Path "logs", "report\results" | Out-Null

function Fail($msg) { Write-Host "ERROR: $msg" -ForegroundColor Red; exit 1 }
function Check($what) { if ($LASTEXITCODE -ne 0) { Fail "$what failed (exit code $LASTEXITCODE)" } }
function Need-Venv { if (-not (Test-Path $Py)) { Fail "no .venv yet - run: .\run_lab.ps1 setup" } }

function Find-Python {
    if ($env:PYTHON) { return ,@($env:PYTHON) }
    if (Get-Command py -ErrorAction SilentlyContinue) {
        foreach ($v in @("3.12", "3.13")) {
            & py "-$v" -c "import sys" 2>$null
            if ($LASTEXITCODE -eq 0) { return ,@("py", "-$v") }
        }
    }
    return ,@("python")
}

switch ($Cmd) {
    "setup" {
        nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
        $base = Find-Python
        $exe = $base[0]; $rest = @(); if ($base.Count -gt 1) { $rest = $base[1..($base.Count - 1)] }
        & $exe @rest -c "import sys; assert sys.version_info >= (3, 12), 'need Python 3.12 or 3.13, found ' + sys.version.split()[0]; print('Python', sys.version.split()[0])"
        Check "Python 3.12+ check (install Python 3.12 from python.org, or set `$env:PYTHON to its python.exe)"
        if (-not (Test-Path $Py)) { & $exe @rest -m venv .venv; Check "creating .venv" }
        & $Py -m pip install --upgrade pip; Check "pip upgrade"
        $idx = "https://download.pytorch.org/whl/cu128"
        if ($env:TORCH_INDEX) { $idx = $env:TORCH_INDEX }
        Write-Host "Installing CUDA PyTorch from $idx ..."
        & $Py -m pip install torch torchvision --index-url $idx; Check "torch install"
        Get-Content requirements.txt | Where-Object { $_ -notmatch '^(torch|torchvision)==' } | Set-Content -Encoding ascii logs\req_no_torch.txt
        & $Py -m pip install -r logs\req_no_torch.txt; Check "requirements install"
        & $Py -c "import torch; ok = torch.cuda.is_available(); print('torch', torch.__version__, '| CUDA available:', ok, '|', torch.cuda.get_device_name(0) if ok else '-')"
        Write-Host "If CUDA available is False, re-run setup with:  `$env:TORCH_INDEX='https://download.pytorch.org/whl/cu130'  (or cu126)"
    }
    "check" {
        $files = @("Document.pdf", "Document2_Supplement_A_Outturns_2011-2013.pdf", "Document3_Briefing_B_Labour_Prices_Policy.pdf",
                   "kb_manifest.json", "Questions.csv", "validate_submission.py")
        foreach ($f in $files) {
            if (Test-Path "data\$f") { Write-Host "ok       data\$f" } else { Write-Host "MISSING  data\$f" -ForegroundColor Yellow }
        }
        if (Test-Path "part1\eval\golden_set.json") { Write-Host "ok       part1\eval\golden_set.json" }
        else { Write-Host "MISSING  part1\eval\golden_set.json (copy golden_set_TEMPLATE.json, write your 25 questions, commit it)" -ForegroundColor Yellow }
        if ($env:LORA_REPO) { Write-Host "LORA_REPO=$($env:LORA_REPO)" } else { Write-Host "LORA_REPO not set (needed for Part 3 domain style)" -ForegroundColor Yellow }
    }
    "test" {
        Need-Venv
        & $Py -m pytest -q part1\tests part2\tests part3\tests; Check "tests"
    }
    "p1" {
        Need-Venv
        & $Py part1\build_notebook.py; Check "building the Part 1 notebook"
        $env:DATA_DIR = Join-Path $Root "data"
        Write-Host "Running Part 1 on GPU $($env:CUDA_VISIBLE_DEVICES) (log: logs\p1.log) ..."
        & $Py -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebooks\p1_rag.ipynb 2>&1 |
            ForEach-Object { "$_" } | Tee-Object -FilePath logs\p1.log
        Check "Part 1 notebook (see logs\p1.log)"
        Copy-Item notebooks\submission.csv . -ErrorAction SilentlyContinue
        Copy-Item notebooks\p1_*.csv, notebooks\p1_*.json report\results\ -ErrorAction SilentlyContinue
        Write-Host "done: notebooks\p1_rag.ipynb (with outputs), .\submission.csv, report\results\p1_*"
    }
    "p2" {
        Need-Venv
        & $Py part2\build_notebook.py; Check "building the Part 2 notebook"
        Write-Host "Running Part 2 on GPU $($env:CUDA_VISIBLE_DEVICES) (log: logs\p2.log) ..."
        & $Py -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebooks\p2_finetune.ipynb 2>&1 |
            ForEach-Object { "$_" } | Tee-Object -FilePath logs\p2.log
        Check "Part 2 notebook (see logs\p2.log)"
        Copy-Item notebooks\p2_*.csv, notebooks\p2_*.json, notebooks\p2_grid.png report\results\ -ErrorAction SilentlyContinue
        Write-Host "done: notebooks\p2_finetune.ipynb (with outputs), report\results\p2_*"
        Write-Host "Part 3 can use the final adapter:  `$env:LORA_REPO = '<path printed in section 8 of the notebook>'"
    }
    "p3" {
        Need-Venv
        if (-not $env:LORA_REPO) { Write-Host "WARNING: LORA_REPO not set - domain-style scenarios use base SD 1.5" -ForegroundColor Yellow }
        & $Py part3\build_notebook.py; Check "building the Part 3 notebook"
        Write-Host "Running Part 3 on GPU $($env:CUDA_VISIBLE_DEVICES) (log: logs\p3.log) ..."
        & $Py -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebooks\p3_agents.ipynb 2>&1 |
            ForEach-Object { "$_" } | Tee-Object -FilePath logs\p3.log
        Check "Part 3 notebook (see logs\p3.log)"
        Copy-Item notebooks\p3_*.csv, notebooks\p3_*.json report\results\ -ErrorAction SilentlyContinue
        New-Item -ItemType Directory -Force -Path traces | Out-Null
        Copy-Item notebooks\p3_traces\*.json traces\ -ErrorAction SilentlyContinue
        Write-Host "done: notebooks\p3_agents.ipynb (with outputs), report\results\p3_*, images in notebooks\p3_images\"
    }
    "app" {
        Need-Venv
        & $Py part3\app.py
    }
    default {
        Get-Content $PSCommandPath -TotalCount 17 | Select-Object -Skip 1
    }
}
