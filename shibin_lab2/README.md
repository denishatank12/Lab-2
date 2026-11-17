# DATA 266 Lab 2 - Team ## (`data266-lab2-team##`)

All three parts: **Part 1** (multimodal RAG, Kaggle competition), **Part 2** (LoRA fine-tuning of Stable Diffusion 1.5
on Naruto images) and **Part 3** (multimodal multi-agent system, which loads the Part 2 LoRA for domain-style requests).

All work runs on **one 16 GB GPU** (Kaggle T4/P100 or Colab T4). **No hosted API is used**: every model is an
open-weights model downloaded from Hugging Face:

| Use | Model (Hugging Face id) | Part |
|---|---|---|
| Text-to-image base model, fine-tuned with LoRA | `stable-diffusion-v1-5/stable-diffusion-v1-5` | 2, 3 |
| KID features / diversity | Inception-v3 (torch-fidelity) / LPIPS AlexNet | 2 |
| Text embeddings | `BAAI/bge-small-en-v1.5` | 1 |
| Reranker (bonus) | `BAAI/bge-reranker-base` | 1 |
| Vision-language model, Part 1 (describes every figure/table at ingestion, then unloaded), 4-bit NF4 | `Qwen/Qwen3-VL-8B-Instruct` | 1 |
| Answer generator, Part 1 (text + table facts + figure descriptions), 4-bit NF4, no thinking mode | `Qwen/Qwen3-14B` | 1 |
| Vision-language model, Part 3 (planner probe, vision, prompt rewriting, critique), 4-bit NF4 | `Qwen/Qwen3-VL-8B-Instruct` | 3 |
| Style change / domain style (img2img, txt2img) | `stable-diffusion-v1-5/stable-diffusion-v1-5` + Part 2 LoRA (`LORA_REPO`) | 3 |
| Instruction edits | `timbrooks/instruct-pix2pix` | 3 |
| Prompt-image similarity, memorization check | `openai/clip-vit-large-patch14` | 2, 3 |
| Output NSFW filter | `stable-diffusion-v1-5/stable-diffusion-v1-5` `safety_checker` (CompVis) | 3 |

## Layout

```
part1/rag/            config, manifest, ingest, describe, index, retrieve, memory, answer, safety, llm, pipeline, evaluate
part1/eval/           golden_set.json (25 questions, written before tuning), p1_human_eval.csv (ratings)
part1/tests/          synthetic look-alike PDFs (invented numbers) + pytest suite
part1/build_notebook.py   -> notebooks/p1_rag.ipynb  (self-contained; the code is written into it with %%writefile)
part3/agents/         state (pydantic), safety, backends, agents (5 agents), graph (orchestrator + traces), evaluate
part3/eval/           inputs.json (16 COCO photos, fetched from Hugging Face), scenarios.json (15), redteam.json (10), make_images.py
traces/               Part 3 JSON traces (one file per request, copied here by run_lab p3)
part3/app.py          Gradio demo
part3/build_notebook.py   -> notebooks/p3_agents.ipynb
report/               report sections (Markdown, converted to the IEEE PDF), architecture diagrams
CONTRIBUTIONS.md  AI_USAGE.md  links.txt  requirements.txt
```

Data, weights and generated images are **not** committed (`.gitignore`): competition files go in `data/` locally
or are attached on Kaggle; model weights and generated images go to Hugging Face.

## Part 1 - Multimodal RAG (Kaggle)

Pipeline: PyMuPDF text chunks (doc_id, page) + every numbered Figure/Table cropped with its caption (separate
objects for "Figure B-5" and "Figure B-5 (corrected)") -> Qwen2.5-VL text description of each figure (values
transcribed) -> ChromaDB (metadata: doc_id, page, figure_id, trust_level, kind, section, excluded) + BM25 ->
hybrid retrieval (RRF) + exact figure lookup (+ optional reranker) -> Qwen2.5-VL JSON answer from numbered
evidence -> deterministic validation (citations only from retrieved evidence, exact labels, status rules,
<= 120 words). DOC3 Appendix C is stored but flagged `excluded` and never retrieved; instruction-like text in
any chunk or image description is removed before indexing. Each conversation has its own memory; user values
never reach the vector DB.

**Kaggle (graded run):** create a notebook from `notebooks/p1_rag.ipynb` at tag `p1-v1.0`, attach the competition
data (and a dataset containing `golden_set.json` / `p1_human_eval.csv` for sections 5-6), GPU T4, Internet on,
**Run All**. It reads `QUESTIONS_PATH` (default: `Questions.csv` of the competition data), writes
`/kaggle/working/submission.csv`, runs the format checks and `validate_submission.py`. Expected time on a T4:
ingestion + figure descriptions + 40 questions well under 90 minutes.

Local:
```bash
pip install -r requirements.txt
```
```bash
python part1/build_notebook.py && jupyter nbconvert --to notebook --execute --inplace notebooks/p1_rag.ipynb
```
Options (environment variables): `VLM_MODEL`, `ANSWER_MODEL`, `LLM_BACKEND` (`two_stage` default; `qwen_vl` = one VLM for everything), `LOAD_IN_4BIT`, `P1_RERANK_EVAL`,
`USE_RERANKER=1`, `DESCRIBE_FIGURES=0`, `DATA_DIR`, `QUESTIONS_PATH`.

Tests (CPU, synthetic look-alike PDFs, no model download except the embedder):
```bash
pytest -q part1/tests part3/tests
```

## Part 2 - Fine-tuning Stable Diffusion 1.5 (Naruto BLIP captions)

`notebooks/p2_finetune.ipynb` (built from `part2/p2/` by `part2/build_notebook.py`):
data cleaning (corrupt/blank/tiny images, exact and near duplicates) -> 90/10 split with seed 266 -> 512x512 crops;
**run 1: LoRA rank 4, run 2: LoRA rank 32** on the UNet attention projections, same data, 1,500 steps, batch 4,
LR 1e-4 (AdamW, warm-up + cosine, min-SNR 5), bf16/fp16 with float32 adapters; the 20 prompts in
`part2/eval/prompts.json` (fixed before training) x seeds 0 and 1 for the base model and both runs
(DPM-Solver, 30 steps, guidance 7.5, safety checker on); KID vs the test images, CLIP score (ViT-L/14), LPIPS
between seeds, blinded human pairs, memorization check (CLIP image similarity to the training set); adapters,
model card and all generated images pushed to Hugging Face (`huggingface-cli login` first).
```bash
bash run_lab.sh p2
```
Windows: `pwsh -ExecutionPolicy Bypass -File .\run_lab.ps1 p2`. Options: `P2_STEPS`, `P2_BATCH`, `P2_LR`, `HF_USER`,
`P2_FINAL` (force the final run), `P2_RETRAIN=1` (retrain even if an adapter exists).

## Part 3 - Multi-agent system

Planner (input safety, intent, route) -> Vision Understanding -> Prompt Engineering -> Image Generation ->
Critique (CLIP + VLM judgement + output safety) -> accept or revise (max 2). Plain-Python orchestration over one
pydantic `AgentState`; a JSON trace of every request (agent, input, output, seconds) is written to `traces/`.

Set the Part 2 adapter first: `export LORA_REPO=<hf-user>/<part2-lora-repo>` (and `LORA_WEIGHT_NAME` if needed,
`DOMAIN_NAME=Naruto` or `Pokemon`).
```bash
python part3/build_notebook.py
```
```bash
python part3/app.py
```
`SHARE=1` prints a public Gradio link (Kaggle/Colab); `MOCK=1` runs the UI with deterministic stand-in models.
The notebook `notebooks/p3_agents.ipynb` runs the 15 scenarios with and without the critique loop, the 10
red-team cases and writes the human-evaluation sheets and the results table.

## Running on a GPU lab machine from the terminal

`run_lab.sh` executes the notebooks headless (`jupyter nbconvert --execute`), saving all outputs into the `.ipynb`:
```bash
bash run_lab.sh setup        # venv + pinned packages; prints whether CUDA works
bash run_lab.sh check        # data/ files and part1/eval/golden_set.json present?
GPU=0 bash run_lab.sh p1     # Part 1 -> notebooks/p1_rag.ipynb with outputs, ./submission.csv
GPU=0 LORA_REPO=<hf-user>/<lora> bash run_lab.sh p3
SHARE=1 bash run_lab.sh app  # Gradio demo
```
On **Windows** use `run_lab.ps1` with the same commands
(`powershell -ExecutionPolicy Bypass -File .\run_lab.ps1 setup|check|test|p1|p3|app`); it installs the CUDA build
of PyTorch (PyPI's Windows torch is CPU-only) and needs Python 3.12 or 3.13.
Use `tmux` (or `nohup`) on a shared machine so a run survives a dropped SSH session. The final Part 1 graded run
must still be the Kaggle notebook at tag `p1-v1.0` (section 1.5 B).

## Reproducibility

Seed 266 everywhere (Python, NumPy, torch; diffusion seeds = 266 + attempt). Decoding is greedy. The first cell of
each notebook prints commit, library versions, GPU and seed. Tags graded: `p1-v1.0`, `p2-v1.0`, `p3-v1.0`.
