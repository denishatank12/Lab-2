# Lab 2 Part 1 - Multimodal RAG baseline

This folder contains the first runnable implementation for the official Part 1 assets.
It ingests the three PDFs into page-addressable passages, preserves document trust metadata,
retrieves with BM25 plus optional Sentence-Transformers embeddings, and writes the required
submission schema. The fallback path is deterministic and works without a GPU; dense retrieval
uses CUDA automatically when PyTorch reports an available GPU.

## Run

```bash
cd part1_rag
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python src/smoke_test.py
python src/generate_submission.py --data-dir data --output outputs/submission.csv
python data/validate_submission.py outputs/submission.csv data/Questions.csv
```

The generated CSV is an auditable retrieval baseline. It is not the final tuned answerer yet:
the next step is adding the local answer-generation model, figure-image extraction/description,
the 25-question golden evaluation, and the Kaggle notebook after the baseline is verified.

## Overnight safety

The current retrieval stage is short and does not require overnight training. Later GPU stages
must use checkpoints and resume from the latest completed step; no large model or dataset should
be committed to GitHub.
