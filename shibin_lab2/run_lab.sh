#!/usr/bin/env bash
# Run Parts 1 and 3 on a GPU machine from the terminal (no Jupyter UI needed).
# Notebooks are executed headless with nbconvert and saved WITH their outputs.
#
#   bash run_lab.sh setup     # venv + packages, checks the GPU
#   bash run_lab.sh check     # are the data files / golden set in place?
#   bash run_lab.sh test      # CPU unit tests (1 min)
#   bash run_lab.sh p1        # execute notebooks/p1_rag.ipynb     -> submission.csv, results
#   bash run_lab.sh p2        # execute notebooks/p2_finetune.ipynb (data, two LoRA runs, generation, evaluation)
#   bash run_lab.sh p3        # execute notebooks/p3_agents.ipynb  (set LORA_REPO first)
#   bash run_lab.sh app       # Gradio demo (prints a public link with SHARE=1)
#
# Options (environment variables):
#   GPU=0                 which GPU to use (one GPU only, as the lab requires)
#   TORCH_INDEX=<url>     install torch from a CUDA-specific index, e.g. https://download.pytorch.org/whl/cu121
#                         (use when `setup` says CUDA is not available - older lab drivers)
#   HF_HOME=<dir>         where model weights are cached (~20 GB; point it at scratch space if home is small)
#   LORA_REPO=<user/repo> Part 2 LoRA on Hugging Face (Part 3);  DOMAIN_NAME=Naruto|Pokemon
set -euo pipefail
cd "$(dirname "$0")"
ROOT=$(pwd)
VENV=${VENV:-$ROOT/.venv}
export CUDA_VISIBLE_DEVICES=${GPU:-0}
mkdir -p logs report/results

act() { source "$VENV/bin/activate"; }

case "${1:-help}" in
setup)
  python3 -c 'import sys; assert sys.version_info >= (3, 10), "Python 3.10+ needed (try: module load python/3.11)"'
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv || echo "WARNING: nvidia-smi not found"
  [ -d "$VENV" ] || python3 -m venv "$VENV"
  act
  pip install -q --upgrade pip
  if [ -n "${TORCH_INDEX:-}" ]; then
    pip install -q torch torchvision --index-url "$TORCH_INDEX"
    grep -vE '^(torch|torchvision)==' requirements.txt > logs/req_no_torch.txt
    pip install -q -r logs/req_no_torch.txt
  else
    pip install -q -r requirements.txt
  fi
  python - <<'EOF'
import torch
ok = torch.cuda.is_available()
print("torch", torch.__version__, "| CUDA available:", ok, "|", torch.cuda.get_device_name(0) if ok else "-")
if not ok:
    print("CUDA is not available. Re-run with an index matching the driver, e.g.\n"
          "  TORCH_INDEX=https://download.pytorch.org/whl/cu121 bash run_lab.sh setup")
EOF
  ;;
check)
  for f in Document.pdf Document2_Supplement_A_Outturns_2011-2013.pdf Document3_Briefing_B_Labour_Prices_Policy.pdf \
           kb_manifest.json Questions.csv validate_submission.py; do
    [ -f "data/$f" ] && echo "ok       data/$f" || echo "MISSING  data/$f"
  done
  [ -f part1/eval/golden_set.json ] && echo "ok       part1/eval/golden_set.json" \
    || echo "MISSING  part1/eval/golden_set.json (copy golden_set_TEMPLATE.json, write your 25 questions, commit it)"
  echo "LORA_REPO=${LORA_REPO:-<not set - needed for Part 3 domain style>}"
  ;;
test)
  act; python -m pytest -q part1/tests part2/tests part3/tests
  ;;
p1)
  act
  python part1/build_notebook.py
  echo "running Part 1 on GPU $CUDA_VISIBLE_DEVICES (log: logs/p1.log) ..."
  DATA_DIR="$ROOT/data" jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=-1 notebooks/p1_rag.ipynb 2>&1 | tee logs/p1.log
  cp notebooks/submission.csv . 2>/dev/null || true
  cp notebooks/p1_*.csv notebooks/p1_*.json report/results/ 2>/dev/null || true
  echo "done: notebooks/p1_rag.ipynb (with outputs), ./submission.csv, report/results/p1_*"
  ;;
p2)
  act
  python part2/build_notebook.py
  echo "running Part 2 on GPU $CUDA_VISIBLE_DEVICES (log: logs/p2.log) ..."
  jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 \
    notebooks/p2_finetune.ipynb 2>&1 | tee logs/p2.log
  cp notebooks/p2_*.csv notebooks/p2_*.json notebooks/p2_grid.png report/results/ 2>/dev/null || true
  echo "done: notebooks/p2_finetune.ipynb (with outputs), report/results/p2_*"
  ;;
p3)
  act
  [ -n "${LORA_REPO:-}" ] || echo "WARNING: LORA_REPO not set - domain-style scenarios will use base SD 1.5"
  python part3/build_notebook.py
  echo "running Part 3 on GPU $CUDA_VISIBLE_DEVICES (log: logs/p3.log) ..."
  jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 \
    notebooks/p3_agents.ipynb 2>&1 | tee logs/p3.log
  cp notebooks/p3_*.csv notebooks/p3_*.json report/results/ 2>/dev/null || true
  mkdir -p traces && cp notebooks/p3_traces/*.json traces/ 2>/dev/null || true
  echo "done: notebooks/p3_agents.ipynb (with outputs), report/results/p3_*, images in notebooks/p3_images/"
  ;;
app)
  act; python part3/app.py
  ;;
*)
  sed -n '2,20p' "$0"
  ;;
esac
