# DATA 266 Lab 2

This repository contains the three lab implementations and their run notebooks.
The latest repair keeps the completed Part 2 GPU results and replaces the older
Part 1 and Part 3 baselines with the missing model-backed components.

## Open these notebooks

For **opening just one file in Colab**, use
[Lab2_Colab_Standalone.ipynb](Lab2_Colab_Standalone.ipynb).
It contains its helper code, creates its internal files automatically and
downloads the course inputs. No clone, separate Python uploads or result ZIP
is needed. Choose a GPU runtime and Run All. A hidden Hugging Face read-token
prompt restores your original private Part 2 adapters and 120 images. Those
results stay labeled as original RTX 4090 measurements. The corrected Part 1
and Part 3 experiments then run on the GPU actually assigned to this notebook.

For the RTX 4090, use [Lab2_RTX4090_Run_All.ipynb](Lab2_RTX4090_Run_All.ipynb).
It verifies the real GPU, reuses the completed Part 2 result folder or original
ZIP, restores missing dataset images, preflights the models, then executes
Parts 1 and 3 sequentially. It saves the executed notebooks and offers one
combined result ZIP. Put the original Part 2 ZIP on that runtime before starting
if its results folder is not already present. Training is not repeated.

For the RTX 4050 overnight run, open [Lab2_Overnight_Run.ipynb](Lab2_Overnight_Run.ipynb).
It preflights both real model paths before the long work, then runs Parts 1 and 3
sequentially and autosaves their executed notebooks. It does not repeat Part 2
training. Do not leave until both real GPU preflights pass on that PC.

Afterward, use the final **Download one ZIP with all available lab results** cell.
It creates one integrity-checked ZIP for all three parts and offers a browser
download through the Colab notebook connection, including local runtimes. No
localhost file link or extra web server is needed. Missing results stay listed
in its manifest. You can run only this cell after a failure without repeating
GPU work. See the rerun guide for the contents and download details.

| Part | Notebook | What needs running now? |
| --- | --- | --- |
| 1 | [RAG](part1_rag/notebooks/Part1_RAG_Run.ipynb) | Full revised GPU run |
| 2 | [Fine-tuning](part2_diffusion/notebooks/Part2_FineTune_Run.ipynb) | Reuse completed training; refresh the last review cell when evidence changes |
| 3 | [Agents](part3_agents/notebooks/Part3_Agents_Run.ipynb) | Full revised GPU run using the existing Part 2 adapter |

Read [RERUN_GUIDE.md](RERUN_GUIDE.md) before running. The setup creates an isolated,
pinned environment. Run one notebook at a time in a fresh kernel. A first full
run needs internet and roughly 40 GB of free disk overall. Private adapter download asks
for a token securely; do not put tokens in source code or notebook outputs.

The lab specifies a single 16-GB GPU. The saved Part 2 run used a 24-GB RTX 4090.
That result is preserved honestly, not renamed as a 16-GB run. A 6-GB RTX 4050
is not the grading target and the revised large-model paths have not been fully
tested on it. The new low-VRAM profile uses CPU weight offload and a 512-pixel
image budget; it still needs an actual successful preflight and full GPU run.

## What changed

Part 1 now extracts numbered figure/table images as well as text, stores dense
and CLIP vectors in local Qdrant, combines dense search with BM25 and image
retrieval, and uses a local vision-language model for grounded JSON answers.
Memory is isolated per conversation; Appendix C is quarantined. The 25 proposed
golden references still need the team's own review and revision.

Part 2 retains the actual two 1,200-update LoRA runs, 120 images and measured
KID/CLIP/LPIPS results. The private Hugging Face upload records are tied to the
actual adapter and image-manifest hashes. Existing GPU cells are unchanged.

Part 3 now has image question answering/text transcription, prompt rewriting,
separate style and instruction-editing models, a pinned Hub adapter, visual
critique and image-based safety cases. Fifteen rubric-defined scenarios run with
and without critique. Automatic acceptance is not counted as human task success.
The default real inputs are held-out anime frames; this does not establish
performance on unrelated photographs.

## Ratings and reports

[Blank forms and rating instructions](docs/rating_forms/README.md) are provided.
The separate `sample_formats_DO_NOT_SUBMIT.json` illustrates the format only.
Its invented examples are not measurements and are rejected as an input ballot.
Real ratings must come from the required people looking at the actual outputs.

[Final_Lab_Report_DRAFT.pdf](docs/Final_Lab_Report_DRAFT.pdf) combines the three
parts and keeps unmeasured values marked pending. It is not a finished report
until the revised runs, human review and team details are complete.

See [VALIDATION.md](VALIDATION.md) for the precise scope of local checks and
[AI_USAGE.md](AI_USAGE.md) for the assistance disclosure. CPU smoke outputs in
Parts 1 and 3 are deliberately labeled and are not final GPU experiments.

## Submission status

The final audit also checks team details, grader access, the required repository
name/privacy, human ratings, final Kaggle selection, recordings, contributions,
demo rehearsal and final part tags. Those actions need real evidence; the
software does not invent them. No new final tag is created by this repair.
Weights and generated-image datasets belong on Hugging Face, not in Git.
