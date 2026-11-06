# Lab 2 Part 3 - Multimodal multi-agent system

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
