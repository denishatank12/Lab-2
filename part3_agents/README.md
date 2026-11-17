# Lab 2 Part 3 - Multimodal multi-agent system

## Partner review and Denisha's baseline

Use [Part3_Partner_Baseline_Standalone.ipynb](notebooks/Part3_Partner_Baseline_Standalone.ipynb)
for Denisha's baseline. Import only that file into a fresh Kaggle GPU session,
enable Internet and Run All. Shibin made
[`Shibin22/sd15-naruto-run1-lora-r4`](https://huggingface.co/Shibin22/sd15-naruto-run1-lora-r4)
public. Anonymous access returned HTTP 200 at revision
`88fa0cf4426c3950d1d8a7378c43a20a23f6cdb5`, with the expected safetensors weights.
No token or PowerShell command is needed for the default run. No separate Python
uploads or repository clone are needed. Do not run the older all-parts launcher
for this baseline.

Leave `USE_HF_TOKEN = False` for public access; an old invalid token is ignored.
Only if deliberately switching to a private repository should it be enabled.
That optional mode uses a notebook secret or hidden interactive input, never a
hard-coded token. A background job cannot respond to a hidden prompt.

The November 16 setup repair bypasses Debian/Kaggle `ensurepip`: it creates the
environment without pip and uses the host pip's
[`--python` option](https://pip.pypa.io/en/stable/topics/python-option/) to install
into that environment. The final repair uses **no system site packages or parent
package bridge**. It installs PyTorch 2.11.0+cu128 with torchvision 0.26.0+cu128
in a separate, profile-hashed environment, then requires `pip check`, isolated
imports, a compiled torchvision operation and a CUDA NF4 forward-pass smoke test
to pass. Host Colab, nbconvert, cuDF and torchao packages are not inherited.
The compatible Hub/Tokenizers pair is pinned to 1.32.0/0.23.1. All resolved
versions are saved in the result ZIP. Older environments and saved results are
preserved. A separate CPU profile supports local software verification only.

The baseline disables the revision loop, not the diagnostic critic or safety
filters. It runs the same 15 scenarios and 10 red-team cases, saves every
request's typed state and trace, keeps the executed notebook, and exports one ZIP
under `/kaggle/working`. Download the newest `Lab2_Part3_Baseline_*.zip` from the
Output panel. Interrupted runs keep partial files; do not delete them. Model and
input revisions are recorded so a changed run is not silently merged.

### What the partner actually ran

The [unchanged executed notebook](partner_snapshot/Partner_Part3_Executed.ipynb)
comes from [Shibin's Part 3](https://github.com/Shibin-1020/data266-6945/tree/a79529d1d59a6efb05bf0524ce6d60bda9bf6c09/GPU_LAB%202/part3).
All 24 code cells have execution counts and no saved error outputs. Its first
cell identifies an RTX 4090, Python 3.13.3 and Torch 2.11.0+cu128; its printed git
commit was blank. The external source commit and file hashes are preserved in
[PROVENANCE.json](partner_snapshot/PROVENANCE.json). This is Shibin's execution,
not Denisha's.

| Saved automatic result | No revision loop | Critique loop |
| --- | --- | --- |
| Scenario proxy checks | 15/15 | 15/15 |
| Median request time | 20.671 s | 20.402 s |
| Mean revisions | 0 | about 0.67 |
| Red-team attack success | Not measured separately | 0/10 |
| Human task success and critic agreement | Not rated | Not rated |

These automatic checks mostly test routes, status, files and text. They do not
prove that a sketch, snow edit or style transfer looks correct. The small timing
difference does not establish that the critique loop is faster.

### Source and runtime qualifications

Two latest partner modules differ from the source embedded in the executed
notebook. Later source adds sketch/snow processing and mock-test support. The
new baseline uses the executed snapshot instead of claiming that newer source
produced the older results. The latest-source test files are kept separately;
they do not certify the older snapshot.

The baseline runner adds declared safeguards without editing the snapshot:

- All 16 real COCO photos must download. Placeholders are allowed only in mock tests.
- The exact adapter must exist and load. A missing adapter cannot fall back to the base model.
- Safety-checker exceptions block acceptance, and generated images are checked before saving.
- Model revisions, input hashes, actual hardware and per-case checkpoints are saved.
- Diffusion CPU offload is enabled for memory headroom. It can change timing relative to the partner run.

The original partner code disables the diffusion pipeline's built-in checker
and instead calls an external checker. Its nodes catch checker failures and
continue. The wrapper closes that failure path; it does not claim the original
run used this safeguard. Exact partner input bytes and raw per-request files
were not in the supplied Git tree, so byte-identical image reproduction is not
established by the imported notebook alone.

### Still needed before final submission

Denisha must run the real baseline. Both members must complete genuine ratings
and resolve disagreements, then integrate the comparison into the team report.
The assignment's single-16-GB setup still needs actual compatibility evidence;
the existing partner run used a 24-GB RTX 4090. Private input/output links,
grader access, team details and the required demo/submission records also need
their final review. A local mock smoke test is not any of those measurements.

## Older local implementation

The remaining sections describe this repository's earlier agent implementation,
not the newly selected partner snapshot or Denisha's new baseline notebook.

## Revised model-backed system

Use `notebooks/Part3_Agents_Run.ipynb` and the root `RERUN_GUIDE.md`. The new runtime
uses Qwen2-VL for visual questions, text transcription, rewriting and critique;
SD img2img for styles; InstructPix2Pix for edits; and the pinned Part 2 Hub adapter
for domain requests. It unloads the VLM before diffusion inference.

The new notebook runs both configurations on held-out anime images, saves full
traces and outputs, and tests actual image-pixel attacks. Independent rubric
ratings stay pending until people supply them. Generated-image Hub upload is an
explicit opt-in action after rights review. The local smoke check is not the
full GPU experiment or a completed live team demo.

## Historical baseline notes

The description below refers to the older BLIP/ViLT implementation. It is kept
for context only; the revised notebook uses the new model runtime above.

The implementation uses plain typed Python functions so it is easy to audit:
Planner -> Vision -> Prompt -> Generation -> Critique, with at most two revisions.
Unsafe and deceptive text requests are refused before generation. Broken images
produce a recorded error instead of a crash. Text guardrails are heuristic, not
a complete image-safety detector. Vision uses BLIP captioning and ViLT visual QA
on the CPU; Diffusers img2img offloads components for a small CUDA card.

```bash
cd part3_agents
python -m pip install -r requirements.txt
python src/smoke_test.py
python src/run_scenarios.py --output outputs/scenario_traces.json
python src/app.py
```

Use the shared runtime created by `tools/setup_runtime.py`, or simply Run All in
the corrected notebook. Part 2 must have a completed run1 adapter for domain-style
requests. Do not run Parts 2 and 3 simultaneously on a 6 GB card.

With `--enable-models`, the generation agent really consumes the input image.
The critic uses a CLIP alignment score and a generated-image caption. It can
request at most two revisions. This is a weak automatic critic, not a substitute
for the two members' judgments. Output safety relies in part on the checkpoint's
enabled safety checker. Visible text is a VQA guess, not verified OCR.

The notebook runs scenarios with and without critique. Each scenario/revision
gets a distinct output path, and the Gradio app saves a separate trace directory
per request. Model-disabled tests are labeled `models_disabled`, never successful
generated answers. Generation exceptions are saved as errors.

The default synthetic picture only checks wiring. The required diverse inputs,
scenario rubrics and independent human judgments remain necessary for the lab.
The ten text red-team checks do not establish robustness to unseen image attacks.
