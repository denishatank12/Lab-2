# DATA 266 Lab 2 · Team 09

Denisha Ketan Tank and Shibin Biji Thomas Lnu

This is the submission working tree. The selected notebooks and genuine results
are listed below. Older launchers and experiments are recoverable from Git
history; they are not alternative submissions.

| Part | Files to review | Current evidence |
| :--- | :--- | :--- |
| 1: multimodal RAG | [Selected v6 notebook](part1_rag/notebooks/Lab2_Part1_Kaggle_Final.ipynb) | User's leaderboard screenshot: **0.510**. Original executed v6 notebook/ZIP still needs restoring. |
| 2: diffusion LoRA | [Executed notebook](part2_diffusion/notebooks/Part2_FineTune_Run.ipynb) | Both 1,200-step RTX 4090 trainings, 120 images and measured KID/CLIP/LPIPS. No retraining needed for those results. |
| 3: agents baseline | [Denisha's executed T4 notebook](part3_agents/notebooks/Part3_Baseline_Executed.ipynb) | Full real baseline: 15 scenarios, 10 red-team cases, no saved cell errors. |
| 3: agents final | [Shibin's executed notebook](part3_agents/partner_snapshot/Partner_Part3_Executed.ipynb) | Paired baseline/final automatic results on RTX 4090, with attribution. |
| 3: reproduction | [Standalone baseline](part3_agents/notebooks/Part3_Partner_Baseline_Standalone.ipynb) | One-file Kaggle entry point. The public partner adapter needs no token. |

The 15/15 Part 3 automatic check is not a human quality score. For example, the
saved baseline pencil-sketch request still looks like a colour photograph.
Both members must independently rate the actual results before human success
and critic-agreement scores can be reported.

## Submission files

* [Findings](REPORT.md), [requirement audit](SUBMISSION_STATUS.md) and [audit data](submission/audit.json)
* [Contributions and recording log](CONTRIBUTIONS.md), [AI disclosure](AI_USAGE.md) and [links](links.txt)
* [Blank rating forms](docs/rating_forms/README.md), [Part 2 model cards](part2_diffusion/model_cards/run1.md)
* The combined report PDF and `Pair_Programming_Team_09.zip` are generated locally
  by `python tools/build_combined_report.py`. PDFs, image datasets and weights
  are excluded from new Git commits as the assignment requests.

## What still needs real evidence

1. Restore the original selected Part 1 executed v6 notebook and Results ZIP.
   The later file with the same notebook name is v8; its results are not used as v6.
2. Finish the team's golden-reference review and real ratings: both members for
   Parts 1 and 3, and three people for Part 2, including someone outside the team.
3. Add actual meeting recording links, confirm grader access to private Hub
   artifacts and publish the Part 3 input/output artifact package.
4. Confirm the final Kaggle selection/team name, GPU compatibility evidence for
   the remaining workloads, report review and live demo. Create the three part
   tags after the missing evidence is integrated.

The repository keeps the name `Lab-2` and stays public at Denisha's request.
The assignment asks for a private `data266-lab2-team09` repository, so this remains
an explicit requirement difference. There is no claim that the whole submission
is ready while these items remain open.

## Reproducing and checking

Do not rerun completed GPU work just to rebuild the report. Part 1's selected
source is entirely inside its frozen notebook; it unpacks its own exact helper
files. Part 2 uses the root pinned Python 3.12 requirements. The Part 3 standalone
uses its own clean, pinned CUDA environment; do not mix these dependency sets.

On an existing development environment:

```bash
python tools/check_submission_evidence.py
python tools/test_part2_submission.py
python tools/test_partner_baseline.py
python tools/final_submission_audit.py
python tools/build_combined_report.py
```

An audit reporting pending human or access items is not a Python failure.
It distinguishes completed computation from remaining submission obligations.
