# Notebook repair validation

## Combined ZIP and download cell

Nine packaging/download tests passed: per-file hashes and ZIP integrity,
missing-file disclosure, correct autosaved/failed notebook selection, excluding
base weights/caches/raw data, retaining LoRA adapters, secret-pattern redaction
without changing originals, symlink rejection, bounded binary comm transfer,
native directory copying, repeatable notebook-cell insertion, and frontend
JavaScript handling complete versus interrupted transfers. Some checks share a
test method. The browser test uses a mocked Colab frontend; it does not certify a
real user-browser download. The final cell was separately executed in CPU export
mode without repeating training or modifying prior GPU outputs.

## November 6 revision: missing rubric components

### Overnight restart safeguards

Thirteen additional tests passed for protocol/hardware separation, modified
artifact invalidation, interrupted atomic writes, actual RAG/golden-evaluation
entry-point resume with synthetic answers, actual scenario CLI resume in smoke
mode, the low-VRAM placement plan, and real Jupyter-kernel autosave on success and
failure. Fixture failures remain visible; no fixture is used as a quality score.
The RTX 4050 memory plan and new real-model preflight are implemented, but their
full pretrained-model GPU execution has not been verified on this CPU-only Mac.

The sequential launcher was executed in explicitly labeled CPU smoke mode.
Actual GPU preflights, model generation and model-quality evaluations were
skipped, not reported successful. The real preflight must pass on the target PC
before leaving the long run unattended. The first-time fp16 editor variant's
required files were checked against its pinned Hugging Face file listing.

This section supersedes historical implementation descriptions below.
The revised Part 1 includes figure/table extraction, CLIP image retrieval,
Qdrant storage, a local vision-language answerer and isolated conversation state.
The revised Part 3 includes a vision-language model, distinct instruction editing,
Hub adapter provenance, image-aware safety cases and paired evaluation rubrics.
Neither revised system has a completed full pretrained-model GPU evaluation yet.

Twenty-one new offline tests passed. They check actual extraction of 179 trusted
text passages and 35 trusted numbered objects, one quarantined object, real local
Qdrant storage with fixture vectors, exact/corrected figure identity, session
isolation, failed-citation rejection, routing, image-instruction handling, and
rating/upload safeguards. A tiny randomly initialized Qwen2-VL performed an actual
multimodal CPU forward pass. It tests API wiring, not model quality. The official
pinned processor also encoded an image/text input successfully. Twenty Part 2
regression tests passed again, including generation recovery and blinded ratings.
The actual local Gradio API also processed a deceptive-evidence refusal without
loading a model or private adapter. This is a preflight/UI check, not the required
model-backed team demo or proof of broad safety.

All revised Part 1 and Part 3 notebook code cells executed in CPU smoke mode
with no saved errors. Model inference, image generation and quality measurements
were explicitly skipped. The Part 2 refresh cell executed separately on CPU while
preserving every original GPU cell and output. The machine-readable record is
`validation/revised_notebook_smoke.json`. The Part 2 upload registry now binds to
the original adapter and image-manifest hashes instead of certifying any later
replacement weights automatically.

Blank rating forms and an explicitly synthetic format example are separate from
actual results. No independent human judgments, team details, rights, grader
access, 16-GB execution or leaderboard submission are inferred. See
`RERUN_GUIDE.md` for the exact rerun and remaining-work instructions. The combined
report is a draft with pending values, not a claim that the whole lab is ready.

## Latest Part 2 evidence: completed GPU run and imported results

This section supersedes the earlier failed Part 2 upload described below. The
later `Part2_FineTune_Run_(2).ipynb` completed all ten code cells on an RTX 4090
with zero saved error outputs. Both LoRA experiments completed 1,200 updates.
The full result ZIP passed its CRC check. Its SHA-256 is
`5085f961414fa35c795233513219be4218730ac1591e271c8fca28f8696d8d7e`.

The original trained safetensors were inspected: both contain 256 finite LoRA
tensors. All 120 current image files, eight comparison grids, loss histories,
KID/CLIP/LPIPS records and six nearest-training-image figures are present.
Historical generations and optimizer pickle checkpoints were left in the
original archive rather than imported. Existing differing files were backed up.

The canonical Part 2 notebook preserves every original GPU cell and output.
An explicitly separate macOS CPU review cell was actually executed afterward;
it reads the imported evidence, shows nearest pairs and refreshes the report.
It does not claim a second GPU run. Twenty offline regression tests passed,
including incomplete ZIP rejection, path traversal rejection and backup checks.
Test fixtures are not experimental results or human judgments.

The authenticated Hugging Face account was verified as `denishatank`. Both
adapters and the 120-image dataset were uploaded privately, with model/data cards
and commit IDs recorded in `part2_diffusion/evidence/hub_links.json`. The report
includes measured tables, original figures and AI-assisted failure review.

Part 2 is not yet fully submission-ready: independent real ratings, team details,
grader access to private artifacts and redistribution-rights confirmation remain.
No human win rates or permissions were invented. This RTX 4090 run does not
establish the lab's single-16-GB-GPU execution requirement or complete the common
team deliverables and other parts of the lab.

## Generation recovery follow-up

The generation-fingerprint error is now handled without changing the training
configuration or rerunning training. An incompatible image set and its dependent
metrics, blinded packets and review files move to a timestamped history folder.
The original adapters, optimizer checkpoints and training metrics stay in place.
Matching image sets resume normally. No old image is relabeled with a new hash.

Seventeen offline regression tests passed, including the actual generation
entry point with mocked inference: 120 coherent manifest records were produced,
then a second invocation skipped all model loads. Tests also covered changed
and interrupted manifests, preserving old judgments, and keeping adapter and
checkpoint bytes unchanged. Mocked gray fixture images are not GPU generation
or image-quality evidence. See `validation/generation_recovery.json`.

## Follow-up: uploaded runs checked on November 5

The user's uploaded Part 1 notebook executed all 3 code cells, reported an RTX 4090
and passed the supplied validator with 40 rows. It has no saved error outputs.
This validates execution and CSV shape, not the full multimodal RAG rubric or
answer correctness. The saved answers remain an extractive baseline.

The uploaded Part 2 notebook executed only its setup cell, which failed with a
missing SciPy shared-library file. None of its training or evaluation cells ran.
There are no full Part 2 results to approve. The metadata warnings are consistent
with a damaged shared environment; concurrent installation is plausible but not
proven from the saved logs. Original uploaded notebooks were not edited.

The follow-up uses an OS installation lock, a single dependency resolution,
read-only reuse of healthy environments and fresh paths for damaged ones. It
adds Part 2 metric tables, a clearly labeled PDF draft, real-rating aggregation,
an opt-in private Hugging Face upload helper and an evidence checklist. The
checklist does not pass absent human judgments or missing artifact uploads.

Twelve offline regression tests passed, covering installation transaction/ready
markers, lock recovery, rating aggregation, blinded packets and PDF numeric-table
formatting. All 9 Part 2 notebook code cells also executed in explicitly labeled
CPU smoke mode with zero saved errors. The pinned model revision was checked
against the Hugging Face file listing: fp16 UNet/VAE/text-encoder/safety-checker
weights and scheduler/feature-extractor configs are present. This was a metadata
check, not a download or full inference/training test.

See `validation/uploaded_run_audit.json` for hashes and execution counts. The
regression tests include an interrupted installer lock, empty/incomplete rating
files, duplicate ballots and left/right win-rate calculation. These fixture tests
are not presented as human-evaluation findings.

Validated locally on November 5, 2026, on macOS ARM64 with Python 3.12.6.
This machine has no NVIDIA CUDA GPU. RTX 4050 and T4 full runs are not claimed.

## What passed

- The isolated pinned environment installed successfully, passed `pip check`,
  imported all core ML packages, and ran a matching torchvision NMS kernel.
- All three notebooks executed every code cell locally with zero error outputs.
  Part 1 used real dense embeddings. Parts 2 and 3 explicitly used CPU smoke mode.
- Part 1 indexed 179 trusted passages and produced a 40-row submission that passed
  the supplied schema validator. This does not establish answer accuracy.
- Real dataset preparation checked 1,221 Naruto images. The seeded split has
  1,098 training and 123 test images. No exact pixel duplicates, corrupt images
  or empty captions were found. Every prepared image is RGB at 512 by 512, and
  train/test image hashes are disjoint.
- Ten in-domain prompts match the held-out captions. The fixed prompt set also
  has five out-of-domain and five compositional prompts. It is saved before training.
- A tiny randomly initialized diffusion UNet performed actual LoRA backward
  passes. The frozen base weight check passed. Interrupted/resumed training
  matched uninterrupted training. Adapter export, loading, enable/disable and
  two-step miniature image generation passed.
- Agent tests exercised routing, refusal, missing/broken inputs, mocked generation,
  separate revision images, the two-revision limit and error recovery. All ten
  text-only red-team cases passed. These are not pretrained-model quality tests.
- The Gradio app imported and generated its API schema successfully.

The machine-readable record is `validation/cpu_smoke_results.json`. Saved smoke
outputs are not full SD 1.5 training results. Opening a notebook on the RTX 4050
and running it normally selects the full path, not the developer smoke path.

## Important limits

Full Stable Diffusion training, peak GPU memory, numerical image-quality metrics
and model-backed agents still need to run on CUDA. The new code computes real
metrics, but no unmeasured scores are supplied. Dependencies and tiny integration
tests reduce risk; they cannot prove that another PC's driver, free VRAM, network
and disk will remain available for an overnight run.

The complete graded lab is not finished by a runtime repair. Part 1 is still an
extractive baseline, the vision text reader is not verified OCR, and the automatic
image critic is limited. Independent human ratings, the diverse real input set,
the full evaluations, model/image uploads, report, team documents and final tags
remain separate requirements. None of those are replaced with invented results.

## Run on the RTX 4050

1. Update or freshly download this repository and open a new notebook kernel.
2. Run Part 1, then Part 2. Run Part 3 after Part 2 has a completed adapter.
3. Run one part at a time. Keep at least 20 GB of free disk for the first full run.
4. The first cell installs an isolated CUDA-enabled runtime and prints the actual
   device. The GPU tiny-model smoke test runs before full training starts.
5. Save the full executed notebooks and outputs. Check the actual metrics and
   failed scenarios before treating the lab as submission-ready.
