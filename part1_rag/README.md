# Lab 2 Part 1 - Multimodal RAG

## Revised implementation

For a standalone Colab run with no separate Python uploads, open
[Lab2_Colab_Standalone.ipynb](notebooks/Lab2_Colab_Standalone.ipynb).
It runs Parts 1 and 3 and restores the original Part 2 results.

Use `notebooks/Part1_RAG_Run.ipynb`. The revised full path now implements Qdrant
text/image indexing, exact figure/table extraction, CLIP search, BM25 fusion,
Qwen2-VL JSON answers and per-conversation memory. Run the new notebook on the
GPU; the saved CPU smoke outputs do not contain its new model evaluations.

Read the repository's `RERUN_GUIDE.md`. The golden-set file contains AI-assisted
proposals awaiting team review. Real human groundedness ratings are separate.

## Historical baseline notes

The following describes the earlier extractive fallback, retained only for
offline checks. It is not the canonical notebook's full model-backed path.

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
