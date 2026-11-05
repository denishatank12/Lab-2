# DATA 266 Lab 2

This repository is for DATA 266 Lab 2 only. It contains the official Part 1 source assets,
a hybrid-retrieval baseline, a checkpointed Stable Diffusion LoRA pipeline, and a typed
multi-agent image pipeline.

## Current status

- Part 1 source PDFs, manifest, questions, sample submission, and validator: included.
- Part 1 ingestion, BM25 retrieval, optional GPU dense retrieval, submission writer, and smoke test: included.
- Part 2: `part2_diffusion/` (public Naruto captions dataset, deterministic split, two LoRA runs,
  checkpoint/resume wrapper, fixed prompt set, generation manifest, and evaluation hooks).
- Part 3: `part3_agents/` (planner, vision, prompt, generation, critique, safety routing,
  15 scenarios, 10 red-team cases, JSON traces, smoke test, and Gradio app).

## RTX 4050 notebook fixes

Open the notebook for the part you want and use a fresh kernel, then Run All:

- [Part 1 RAG](part1_rag/notebooks/Part1_RAG_Run.ipynb)
- [Part 2 fine-tuning](part2_diffusion/notebooks/Part2_FineTune_Run.ipynb)
- [Part 3 agents](part3_agents/notebooks/Part3_Agents_Run.ipynb)

The setup cell handles Windows, Linux, Colab and Kaggle, creates an isolated
environment, and selects CUDA-enabled PyTorch for Parts 2 and 3. It does not
replace packages inside your notebook kernel. Python 3.13 hosts use a managed
Python 3.12 runtime. Keep at least 20 GB of disk free for a first full GPU run.
Run one part at a time and close other GPU-heavy programs on a 6 GB RTX 4050.
The lab's required grading target is still a 16 GB GPU; this smaller-card profile
is an additional compatibility path, not proof of a completed grading run.

Part 2 now caches latents and text embeddings, trains only one LoRA UNet at a
time, checkpoints the optimizer and random generator, resumes compatible runs,
and preserves old trainer outputs in a named legacy backup folder. Inference
uses CPU offload and keeps the diffusion safety checker enabled. There is no
bitsandbytes or xformers dependency.

Part 3 now uses img2img with the input image, actual vision calls, a bounded
critique loop, separate scenario/revision outputs, and explicit error states.
The synthetic image in its notebook is a wiring example, not the final test set.

Read [VALIDATION.md](VALIDATION.md) for exactly what was tested. Local CPU smoke
outputs in the notebooks are explicitly labeled. They are not full SD training
results and must be replaced by your actual Restart & Run All GPU outputs.

## Submission status

Runtime fixes do not make the whole lab submission complete. Part 1 remains an
extractive baseline and still needs the full multimodal answerer and its golden-set
evaluation. Real GPU runs, human ratings, diverse Part 3 inputs, model/image uploads,
the combined report and team documents remain separate work. No results or human
ratings are fabricated and no final lab tags are created by this runtime repair.

Datasets, model weights and generated images are ignored by Git; put the required
artifacts on your team Hugging Face account. Notebook outputs are saved in Git.
See each part's README for additional commands.
