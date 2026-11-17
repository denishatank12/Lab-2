# Part 3 system card

Team 09: Denisha Ketan Tank and Shibin Biji Thomas Lnu.

## Purpose and workflow

This course prototype takes an image and a request. Five stages plan the task,
answer visual questions/read visible text, rewrite a grounded prompt, generate
or edit an image, and critique the result. Typed state and JSON traces retain
the route, tool outputs, safety decision and errors. The final partner system
allows at most two revisions; Denisha's independent baseline allows none.

The VLM is Qwen/Qwen3-VL-8B-Instruct, with 4-bit language weights. Generation uses
SD 1.5 for style/domain transformations and InstructPix2Pix for instruction
editing. Critique combines visual judgment and CLIP ViT-L/14 similarity.
The domain adapter is Shibin22/sd15-naruto-run1-lora-r4. It is Shibin's separate
Part 2 model, not Denisha's rank-4 checkpoint. Recorded model revisions are in
`evidence/denisha_baseline/baseline_outputs/model_revisions.json`.

## Evidence and intended use

Fifteen frozen scenarios cover understanding, style, edits, Naruto-domain tasks
and ambiguous/invalid inputs. Ten additional cases test image prompt injection,
unsafe/deceptive requests and malformed inputs. Denisha completed the baseline
on one Tesla T4; the partner paired run used RTX 4090. The saved automatic
checks do not establish human task success. Human ratings remain blank.

Use is limited to academic, non-commercial exploration. This is not a reliable
OCR service, authenticity detector, safety judge or autonomous editing service.
No claim is made that generated images depict real events or that Naruto
source-image rights have been granted. Respect the base-model and input-image
licenses. The source photos are COCO evaluation inputs with recorded provenance;
their original creator licenses still apply.

## Safety and failures

Image text is treated as data, not executable instructions. Routing guardrails
reject unsafe requests. Diffusion safety checks stay enabled and unchecked
outputs are withheld. Invalid images get a clear error. Each request has its
own state/trace, and revisions are bounded. No secrets are embedded in source.
The Gradio interface should stay private unless the team reviews any sharing.

These protections have limits. The saved pencil-sketch case remains a colour
photo even though its automatic check passes. VLM critique and CLIP can miss
attributes, counts and subtle edits. Image injection tests are a small fixed
set, not proof of broad attack resistance. Expected invalid-input errors are
not evidence of a model crash, but an unexpected exception must not be counted
as a safe refusal. Humans must inspect both actual image quality and refusals.

## Remaining review

Collect both members' independent pass/fail judgments, calculate agreement and
resolve disagreements with reasons. Upload the input/output artifact package
to Hugging Face and verify grader access. Complete a live Gradio rehearsal,
including safe failure. Do not call the whole lab complete before these steps.
