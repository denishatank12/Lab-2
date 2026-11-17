# Part 3: image agents

Denisha's genuine T4 baseline is in `notebooks/Part3_Baseline_Executed.ipynb`.
The metrics, hardware identity, 25 states and 25 traces are in
`evidence/denisha_baseline/`; `IMPORT.json` binds them to the original ZIP.
The notebook has two executed code cells and no error outputs. The baseline
does not run a revision loop. Safety and diagnostic critique remain enabled.

Shibin's unchanged executed notebook and its source are in `partner_snapshot/`.
This contains the controlled baseline/final comparison on his RTX 4090.
Both configurations passed 15/15 automatic checks; final red-team results were
0/10 attacks. His results are attributed to Shibin Biji Thomas Lnu.

Denisha's baseline used one Tesla T4 with 14.56 GiB visible VRAM. Its peak
PyTorch allocation was 10.635 GiB and median request time 39.665 seconds.
This verifies the baseline on a nominal 16-GB GPU, not all lab workloads.
Different hardware and declared runtime patches prevent a causal speed
comparison with the partner's final run.

`notebooks/Part3_Partner_Baseline_Standalone.ipynb` is the one-file reproduction
entry point. It uses the public partner rank-4 adapter without a token. The
older unrelated agent implementation has been removed from the active tree.

The automatic proxy is permissive: S06 retained a colour photo instead of a
pencil sketch. Human success and critic-agreement scores need both members'
independent ratings. A nonfatal host `sitecustomize` warning about `wrapt` is
preserved in the original run log; it did not stop the completed evaluation.
See `SYSTEM_CARD.md` for limits and `../SUBMISSION_STATUS.md` for pending items.
