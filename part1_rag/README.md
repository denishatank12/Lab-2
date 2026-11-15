# Lab 2 Part 1 - Multimodal RAG

## Revised implementation

### New evidence-first Kaggle revision (v7)

Use [Lab2_Part1_Kaggle_v7.ipynb](notebooks/Lab2_Part1_Kaggle_v7.ipynb)
for the new experiment. Import this one file into a fresh Kaggle notebook,
enable a nominal 16-GB GPU and Internet, attach the competition questions when
available, then use Save Version > Save & Run All. No separate Python uploads,
old result folders, recovery step or Hugging Face token is needed.

The changes target the kinds of errors in the previous run: incorrect status
choices, ignored temporary assumptions, neighbouring table values and mismatched
figure references. The model now records short source readings before answering,
sees earlier user turns rather than old assistant claims, and gets row-addressed
table data without a repeated numeric matrix. Only exact object-citation metadata
can add a matching figure label. This is not an automatic factual certification.

Download `submission_part1_v7.csv`, the Results ZIP and the notebook with outputs
from Kaggle's Output panel. The CSV is validated and published before the
separate dense/hybrid evaluation, so a later evaluation failure preserves it.
Check the new evaluation, actual wall time and Kaggle score before replacing a
previous submission. The user's latest screenshot reports **0.510** for v6.
There is no measured v7 leaderboard result yet, and 0.8 is not guaranteed.

The completed v6 artifact below is preserved byte for byte. Local v7 regressions
and clean-folder CPU smoke do not establish a successful full pretrained CUDA
run, an improved score, or completion within the 90-minute limit. No reference
answers or question-ID-specific rules were added to answer generation.

### Preserved independent fresh run (v6)

For a new Kaggle session, use
[Lab2_Part1_Kaggle_Final.ipynb](notebooks/Lab2_Part1_Kaggle_Final.ipynb), titled
"independent Kaggle run (v6)". It needs only a GPU and Internet, plus attached
competition questions when available. Its executable code is embedded, its PDF
downloads are hash-checked, and its isolated Python environment is set up
automatically. It does not need the old session, a recovery notebook, a Results
ZIP, a repository clone, separate Python uploads or an HF token.

The output is `submission_part1_v6.csv`. The required dense-baseline/final-system
comparison, citation evaluation and evidence ZIP are retained. The extra optional
draft-editor pass is disabled: it repeatedly failed its checks and consumed time
in the previous profile, and it is not required by the rubric. This is a changed
measured configuration, not a claim that old outputs were improved or rerun.
It does not guarantee a particular score or a sub-90-minute full run.

Before the main stages the notebook runs regressions, a fictional-model complete
pipeline fixture, and actual GPU chart-answer/citation-judge preflights. Local
CPU smoke checks are not a full pretrained CUDA run. Timing includes setup and
the experiment, with final display/export overhead separately noted. Human
ratings, team reference review, report/team details and external submission
remain separate tasks.

### Historical v5 recovery (optional)

Historical recovery option, only for a session that still has those v5 outputs:

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

### Earlier implementation details

The current Part 1-only Kaggle run is
[Lab2_Part1_Kaggle_Final.ipynb](notebooks/Lab2_Part1_Kaggle_Final.ipynb).
Select a nominal 16-GB GPU, enable Internet, attach the competition data and use
Save Version > Save & Run All. All helper code is embedded. Public model and PDF
downloads are pinned, so no separate Python upload or Hugging Face token is needed.

The current notebook retains the v4 failure fix, `Use only supplied figure labels: []`.
The GPU model loaded successfully in that log; unconstrained generation then kept
guessing a figure label despite having no figure evidence. The first call still
uses natural JSON. Invalid outputs retry with schema-constrained decoding whose
citation/figure choices come only from the current retrieved evidence. Empty
figure choices enforce `[]`. No invented label is silently deleted and no canned
answer or status replaces a failed response. Persistent invalid answers remain
errors. Regression checks include a real tiny Qwen image forward for this repair
path, not a full pretrained 7B GPU run.

The notebook uses the same 7B NF4 model for both systems. The earlier v5 profile added
an evidence review for numeric, chart, revision and conversational answers.
Text observations must include a real source quote. Its result ZIP preserves the
draft, final answer and checks. If only that optional review fails output
validation, the original validated draft is retained and the failed review is
reported explicitly. Real CUDA/model errors and invalid base answers still stop
the run. This is an automatic model judgment, not proof of correctness or a human
rating. The additional calls also cost time.

For the current profile use `submission_part1_v6.csv`, not an old submitted CSV.
It is validated and published before the separate golden-set evaluation. A later
evaluation failure does not hide the completed CSV, but does leave the full lab
evaluation incomplete. Inner errors print automatically and stay in the result ZIP.
The user's latest screenshot reports a v6 leaderboard score of 0.510, up from
0.419. The separate v7 experiment has no measured leaderboard score yet. Neither
a 0.8 nor a 0.9 score, nor completion within 90 minutes, is established by passing
CPU tests.

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
