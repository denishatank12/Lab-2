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

See each part's README for commands. Large datasets, model weights, generated images, and
executed notebook outputs are intentionally ignored by Git and must be uploaded to the
required Hugging Face/Google Drive locations for submission.
