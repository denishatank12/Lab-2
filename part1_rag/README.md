# Lab 2 Part 1 - Multimodal RAG

## Revised implementation

### Recover the existing v5 session first

The supplied v5 log confirms all 40 competition answers were generated and
`submission_part1_v5.csv` was published. It stopped later in the separate citation
judge, not in competition answer generation. The original attempt took 5,828.5
seconds, about 97.1 minutes, so its 90-minute requirement is not met.

Use [Lab2_Part1_Evaluation_Recovery.ipynb](notebooks/Lab2_Part1_Evaluation_Recovery.ipynb)
in the same session with the original saved run folder. Do not start a fresh
background run to recover an existing session. Missing original files cause a
safe stop rather than repeating the 40 competition answers. It checks original
source, CSV and hardware fingerprints, replaces only the citation-judge function
in memory, resumes genuine golden-answer checkpoints, and records the patch and
timing separately. Remaining golden items still need GPU inference. Original
failed notebook outputs and source files stay intact. If that session is gone,
retain its exported Results ZIP for an audited migration.

The citation judge now constrains exact source keys and boolean types from its
first call even when answers use natural JSON. Missing/invalid judgments are not
filled with fabricated support values. The metrics display no longer tries to
read an absent evaluation file. This fix does not establish a new score, timing
compliance, human ratings or a complete GPU run.

### Fresh run

For the current Part 1-only Kaggle run, import
[Lab2_Part1_Kaggle_Final.ipynb](notebooks/Lab2_Part1_Kaggle_Final.ipynb).
Select a nominal 16-GB GPU, enable Internet, attach the competition data and use
Save Version > Save & Run All. All helper code is embedded. Public model and PDF
downloads are pinned, so no separate Python upload or Hugging Face token is needed.

The v5.1 notebook retains the v4 failure fix, `Use only supplied figure labels: []`.
The GPU model loaded successfully in that log; unconstrained generation then kept
guessing a figure label despite having no figure evidence. The first call still
uses natural JSON. Invalid outputs retry with schema-constrained decoding whose
citation/figure choices come only from the current retrieved evidence. Empty
figure choices enforce `[]`. No invented label is silently deleted and no canned
answer or status replaces a failed response. Persistent invalid answers remain
errors. Regression checks include a real tiny Qwen image forward for this repair
path, not a full pretrained 7B GPU run.

The notebook uses the same 7B NF4 model for both systems. The final system adds
an evidence review for numeric, chart, revision and conversational answers.
Text observations must include a real source quote. Its result ZIP preserves the
draft, final answer and checks. If only that optional review fails output
validation, the original validated draft is retained and the failed review is
reported explicitly. Real CUDA/model errors and invalid base answers still stop
the run. This is an automatic model judgment, not proof of correctness or a human
rating. The additional calls also cost time.

Use the newly generated `submission_part1_v5.csv`, not the old submitted CSV.
It is validated and published before the separate golden-set evaluation. A later
evaluation failure does not hide the completed CSV, but does leave the full lab
evaluation incomplete. Inner errors print automatically and stay in the result ZIP.
The only measured public leaderboard score so far is the user's original 0.419.
The v5.1 score and complete GPU runtime are unverified. Neither a 0.8 nor a 0.9
score, nor completion within 90 minutes, is established by passing CPU tests.

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
