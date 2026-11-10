# What to run next

## Open only the standalone Colab notebook

Use [Lab2_Colab_Standalone.ipynb](Lab2_Colab_Standalone.ipynb) when you have no
repository folder or Python files on the runtime. The RTX 4090 launcher is for
an existing project checkout; it is not the one-file Colab entry point.

1. Open the standalone notebook in Colab. Choose Runtime, Change runtime type,
   then GPU. T4 is the course's nominal 16-GB option. Hosted Colab does not use
   the RTX 4090 in your PC unless you explicitly connect a local runtime.
2. Run All. Embedded helper code is verified and created inside the runtime.
   The course inputs download automatically from a pinned revision with hash
   checks. No Python source is fetched from GitHub or uploaded by you.
3. Supply a Hugging Face read token in the hidden prompt, or a Colab secret
   named `HF_TOKEN`. It must read your two private Part 2 models and image
   dataset. Never put it in a code cell or send it in chat.
4. Keep the runtime connected and wait until both real GPU model preflights
   pass. Part 1 and Part 3 then run sequentially. Their executed notebooks save
   after each cell, and unchanged completed items resume while the runtime
   files still exist. Colab can disconnect or expire; these local checkpoints
   do not survive deletion of that runtime.
5. Download the final combined ZIP and save your Colab notebook with outputs.
   If a cell fails, run only the final ZIP cell to collect partial evidence.
   Its missing-file list is not replaced with made-up results.

Allow roughly 40 GB of disk for a cold setup and at least 8 GiB available host
RAM. The isolated installer supports the newer Colab kernel by bootstrapping
Python 3.12 when needed. Missing `ensurepip` does not require you to repair the
running Colab environment or supply scripts manually.

The token restores your actual original Part 2 files from pinned Hub commits.
Adapter and image bytes are checked against their original hashes. Previously
measured metrics and loss histories retain the supplied archive's provenance;
only their display panels are recreated. Training is not repeated and old
4090 timings are not called new T4 measurements. Part 2 training on the required
16-GB setup therefore still needs separate evidence or instructor approval.

Real human ratings, both members' reference review, team information, permitted
sharing, grader access, the demo and external submissions are still separate.
A complete developer smoke test does not establish pretrained-model quality or
submission readiness.

## One notebook for the RTX 4090

Open the latest [Lab2_RTX4090_Run_All.ipynb](Lab2_RTX4090_Run_All.ipynb).
This is the entry point for the next run on your 4090. The individual Part 1
and Part 3 notebooks are called automatically; do not start them alongside it.

1. Connect the notebook to the RTX 4090 PC's local runtime, or open it in
   Jupyter on that PC. Opening Colab does not connect it to the PC's GPU by itself.
2. Keep the completed Part 2 results folder on this runtime. If it is absent,
   copy your original `Part2_results_*.zip` here. The notebook checks Downloads,
   the repository, `/content` and `/app`; otherwise set `PART2_RESULTS_ZIP` in
   its first cell to the actual runtime path. A container cannot read an
   unmounted Windows Downloads path.
3. Close other GPU notebooks, connect power and disable sleep. Allow roughly
   40 GB free disk for a cold installation and at least 8 GiB available host RAM.
4. Run All with `LAB2_SMOKE_ONLY` unset. Setup uses a separate pinned Python
   environment. The early GPU check rejects a CPU or another GPU rather than
   silently running the expensive work there.
5. Stay at the PC until both actual model preflights pass. The notebook then
   runs the full Part 1 and Part 3 experiments in order and autosaves the
   executed notebooks after every cell. It does not retrain Part 2.
6. Keep the kernel connected through the final ZIP download, then send that
   ZIP for review. If an earlier cell fails, run only the final export cell to
   collect the saved evidence. Do not delete completed outputs or edit errors
   out of an executed notebook.

The Part 2 reuse check validates the two original adapter hashes, training
settings, all 120 image combinations, image shapes, quality metrics and
memorization figures. It can seed Part 3's adapter from the verified local
export without asking for a Hub token. This is not a new Hub download or a
claim that the instructor can access the private repositories.

A result ZIP may contain dataset metadata but no source images. The revised
preparation checks actual files and restores the original pinned dataset and
split when needed. Downloading those missing images does not repeat training.

The final computation check also inspects saved scenario errors and confirms
that the real model paths were enabled. Successful computation is not the same
as good answers or submission readiness. Real ratings, reference review, team
details, permitted sharing, demo, external submissions and the rubric's GPU
requirement remain separate. A 4090 run is not a 16-GB-card execution record.

## After the uploaded Part 1 citation error

Download the current notebooks again. The latest ZIP showed a later failure:
the model copied a paragraph into `text` instead of returning the required
answer object. Decoding is now constrained to the required fields, allowed status
values and actual retrieved source labels. Part 3 and the citation judge also
use structured decoding. This controls format, not whether every answer is true.
Unsupported sources and unsafe outputs are still rejected.

Use the fresh Part 1 notebook, not the edited copy whose generation and evaluation
cells print logs. Run All with a fresh kernel. If the old checkout contains local
validator patches, setup preserves it and uses a clean sibling folder named
`Lab-2-structured-v1`. Model downloads in the shared Hugging Face cache can still
be reused. Changed source/protocol fingerprints intentionally do not reuse the
three earlier answers as measurements of this new version.

Run Parts 1 and 3 sequentially, preferably with the overnight launcher below.
The uploaded Part 3 copy did not reach its GPU/model evaluation. Do not repeat
Part 2 training for this repair: its two trained adapters and generated images
are already saved. Finish its real-rating and access checks separately.

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

## Download the ZIP to send for review

The launcher's final cell collects all available Parts 1, 2 and 3 results into
`Lab2_All_Results_<timestamp>.zip`. Run it after the job, or run only that cell
if an earlier stage failed. It never trains or generates images.

The ZIP includes matching autosaved executed notebooks (including a partial
failed notebook when present), generated images, LoRA adapters, saved metrics,
evaluation traces, reports, configuration/source snapshots and diagnostic logs.
It excludes Python environments, downloaded base models, original datasets,
optimizer pickle checkpoints and old ZIPs. Its `BUNDLE_MANIFEST.json` records
each included file's SHA-256 and lists expected files missing on this runtime.
It cannot recover files from another PC or an expired runtime automatically.

In Colab, the cell transfers the ZIP over the notebook connection and requests
a browser save. This also supports a local Colab kernel without installing the
`google.colab` package. Keep that kernel connected while it transfers. If the
browser blocks automatic downloads, click the visible Download/Save ZIP link.
The destination follows your browser's download settings, usually Downloads.

A native Windows or Mac kernel also copies the ZIP into its accessible local
Downloads folder. For a Docker-mounted host folder, you may set
`LAB2_DOWNLOAD_DIR` to that existing mounted path. A container's `/root` is not
your Windows Downloads folder. Other notebook frontends can use that copy or
the exact runtime path printed by the cell; no unreachable localhost link is
presented as a completed download.

The archive is local and is not pushed to GitHub. Common credential patterns are
redacted from copied text without editing your originals; still review the ZIP
before sharing. Upload the resulting ZIP in this chat for the next audit. It is
a review bundle, not an automatic declaration that the Canvas submission is ready.

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
