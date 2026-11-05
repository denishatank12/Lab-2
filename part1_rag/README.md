# Lab 2 Part 1 - Multimodal RAG baseline

This folder contains the first runnable implementation for the official Part 1 assets.
It ingests the three PDFs into page-addressable passages, preserves document trust metadata,
retrieves with BM25 plus optional Sentence-Transformers embeddings, and writes the required
submission schema. The fallback path is deterministic and works without a GPU; dense retrieval
uses CUDA automatically when PyTorch reports an available GPU.

## Run

```bash
# From the repository root:
python tools/setup_runtime.py --device cuda
# Then use the Part1_RAG_Run.ipynb notebook, or the printed LAB2_PYTHON interpreter.
```

The generated CSV is an auditable retrieval baseline. It is not the final tuned answerer yet:
the next step is adding the local answer-generation model, figure-image extraction/description,
the 25-question golden evaluation, and the Kaggle notebook after the baseline is verified.

The smoke test is intentionally lexical and offline. The normal notebook run
requires dense embeddings to load successfully; failures are no longer silently
called hybrid retrieval. Appendix C's low-trust page is excluded. The
`QUESTIONS_PATH` environment variable is respected by both generation and
validation. A valid CSV schema does not establish semantic answer correctness.

## Overnight safety

The current retrieval stage is short and does not require overnight training. Later GPU stages
must use checkpoints and resume from the latest completed step; no large model or dataset should
be committed to GitHub.
