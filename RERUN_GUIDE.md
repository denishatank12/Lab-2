# What to run next

## One overnight launch on the RTX 4050

Use the latest `Lab2_Overnight_Run.ipynb` from GitHub. Do not start an old downloaded
copy. Run its setup and preflight cells while you are still at the computer.
Enter the private-adapter read token only in the hidden prompt, if requested.
Wait for **BOTH REAL GPU PREFLIGHTS PASSED**. Those checks download the actual
models and exercise image understanding, RAG, critique, style generation,
instruction editing and the selected domain adapter on this GPU.

If a preflight fails, do not leave it running and assume the long run will work.
Keep the error and diagnostic log. The code was tested locally in CPU mode;
that does not establish that the actual 4050 preflight or full inference passed.
The 6-GB profile reserves activation space and offloads VLM weights to CPU. Allow
roughly 40 GB free disk overall for a cold setup and at least 8 GiB free host RAM
at preflight. Avoid other GPU-heavy programs, connect power and disable sleep.
The local Colab/Jupyter server must stay running. Remote Colab quotas/disconnects
are outside the notebook's control.

After both preflights pass, start the overnight-work cell. It runs Parts 1 and 3
one after the other, not simultaneously. It does not repeat Part 2 training.
Executed notebooks autosave after every cell under `output/executed`; complete
result ZIPs are written under `output/exports`. On a local/Docker Colab runtime,
these are paths on that runtime, not automatic browser Downloads.

If interrupted, run the same launcher again using the same checkout, inputs and
hardware. Unchanged finished answers, evaluation turns and scenarios resume from
atomic checkpoints. Missing/modified output images invalidate only their item.
Changed code/data/hardware uses a new cache namespace; no old measurement is
relabeled. Runtime failures are retained and retried on explicit restart. A low
quality but completed result is not silently regenerated until it looks better.

This protects completed work, not a promise that no error or further review can
ever occur. Human judgments and external submission actions still need people.

## Part 1

1. Get the latest `Part1_RAG_Run.ipynb` from this repository. Use a fresh GPU kernel.
2. Read the 25 proposed questions and references in `part1_rag/eval/golden_candidate.json`.
   Both members should write or revise the set as required, check its sources,
   and only then set the corresponding `team_verified` fields to `true`.
3. Run the whole revised notebook. Leave `LAB2_SMOKE_ONLY` unset for a real run.
   The saved local smoke outputs intentionally skipped all pretrained-model work.
4. Inspect the generated answers, figures, citations and `evaluation.json`.
   A valid CSV only checks its shape, not whether its answers are correct.
5. Each member independently rates both systems using the blank forms in
   `docs/rating_forms`. Save the real forms in the evaluation folder using the
   filenames in its guide. Run the last review/export cell again.
6. Submit and select the actual CSV on Kaggle. Save the executed notebook.

If only real ratings or reference checks change, the last cell refreshes those
measurements from saved answers without model inference. If a question or its
conversation changes, run the model evaluation again; old answers cannot be
relabeled as responses to a different question.

## Part 2

The existing RTX 4090 run trained both adapters for 1,200 updates and generated
the required 120 images. These repairs do not require repeating that training.
The original GPU cells and outputs are kept intact in the canonical notebook.

Keep the original results folder or ZIP. On a machine that has those files, run
only **Completion refresh without retraining** after adding real ratings, team
details or verified sharing/access information. That cell audits files, updates
the PDF and exports the evidence; it does not train or generate images.

Use two team members and an outside rater for the 25 blinded pairs. Sample-only
values are not real ratings. See `docs/rating_forms/README.md` for the format.

Separate issue: the rubric specifies execution on one 16-GB GPU. The completed
24-GB RTX 4090 run does not prove that requirement. Obtain instructor approval
for the existing hardware or retain an actual permitted 16-GB run with its device
identity and full outputs. Repeating the same 4090 run does not resolve this.

## Part 3

1. Run the latest `Part3_Agents_Run.ipynb` in a fresh GPU kernel after Part 2.
2. Supply a Hugging Face read token only in the hidden prompt. It fetches the
   selected original adapter from its pinned private Hub commit. Training is not
   repeated. The run stops rather than quietly using the base model if access fails.
3. Run all evaluation cells. They execute 15 scenarios for each configuration,
   record captions/answers/images/traces, and run 10 image-aware safety cases for
   both configurations. Keep failures visible; do not replace failed outputs.
4. Both members independently rate all 30 scenario outcomes using the saved
   rubrics. Save `human_ratings.json` in the run folder and document any resolved
   disagreement. Rerun the review/report/export cell, not image generation.
5. Review rights and safety before enabling the opt-in private image upload.
   Arrange grader access separately. No upload is claimed while this stays off.
6. Enable the local Gradio demo cell and rehearse the required scenarios. A local
   safe-refusal smoke test is not a full model-backed team demo.

## Before the final submission

Run `tools/final_submission_audit.py`, inspect each pending item, update the real
team details and review confirmations, and review the combined PDF together.
Keep the executed notebooks and final measured evidence. The report and final
tags should only be finalized once the remaining real work is complete.

The updated code passed local regression and CPU notebook smoke tests, not a
full pretrained-model run on the user's GPU. Driver, memory and network problems
can still occur. Preserve any full traceback and the stage diagnostic log.
