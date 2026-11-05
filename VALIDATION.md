# Notebook repair validation

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
