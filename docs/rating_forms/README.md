# Real rating forms

These files are blank templates, not measured results. No sample scores belong
in the submission. Denisha and Shibin must rate independently before resolving
disagreements.

Part 1: review each of the 25 actual answers and cited passages for both systems.
Give groundedness 1 (unsupported) through 5 (fully supported), with a short reason.
Use the original selected run, not a later experiment. Save completed forms next
to that run's `evaluation.json` and retain both raters' original judgments.

Part 2: give the 25 blinded image pairs and their prompts to both members and one
outside rater. Each chooses A, B or tie separately for domain style and prompt
match. The private rater packet is in the original Part 2 Results ZIP under
`outputs/human_run1`; keep its blinding key away from raters. Merge the completed
ballot lists into `part2_diffusion/outputs/human_run1/ratings.json`. The validator
in `part2_diffusion/src/submission.py` requires 25 pairs from each real rater.

Part 3: the JSON template uses Shibin's actual 15 COCO scenario rubrics, not the
old synthetic demo. Both members rate baseline and critique outputs as pass/fail,
retain comments and record a justified resolved outcome for disagreements.
The original partner run is the paired baseline/final comparison. Denisha's T4
baseline is additional independently executed evidence; its timing is not a
controlled comparison with a different GPU. The baseline ZIP also contains a
blank CSV form. Human task success and critic agreement stay pending until rated.

Add completed real forms under `submission/ratings/` with their run identifier.
The report must use calculated values from those ballots, not automatic proxy
passes or AI visual observations. Refresh the audit after adding them.
