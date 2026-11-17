# What is done and what the team still has to do (Parts 1 and 3)

Deadline: **Tue Nov 17, 2026, 11:59 PM PT** (tags + Canvas zip + Kaggle selection). Demo: Nov 18-24.

## Done in this repository
- Part 1 code for every task (1.1-1.5) + self-contained Kaggle notebook `notebooks/p1_rag.ipynb`.
- Part 3 code for every task (3.1-3.3): 5 agents, orchestrator with JSON traces, failure handling, Gradio app,
  15 scenarios + 10 red-team cases (committed before any run), evaluation notebook `notebooks/p3_agents.ipynb`.
- 36 CPU tests (`pytest -q part1/tests part3/tests`) on synthetic look-alike PDFs / mock models.
- Both pipelines were also run with the real models on CPU (Qwen2.5-VL-3B, bge-small, SD 1.5, InstructPix2Pix,
  CLIP, safety checker) on the synthetic inputs, to check the integration. These are **not** results for the report.
- README, requirements (pinned), CONTRIBUTIONS / AI_USAGE / links templates, IEEE report skeleton with diagrams,
  system card and safety sections; `report/make_tables.py` builds the tables from the notebook outputs.

## Team to-do (in this order)
1. **Repo**: create private GitHub repo `data266-lab2-team##`, push this folder, share with instructor + ISAs.
   Replace `##` / `teamXX` everywhere with your team number. Re-author the commits if you want your names on them.
2. **Part 1 golden set (before any tuning!)**: copy `part1/eval/golden_set_TEMPLATE.json` to
   `part1/eval/golden_set.json`, write the 25 questions **yourselves from the three PDFs** (not from Questions.csv),
   fill gold answers/status/numbers/sources/figures, **commit and push it** (git timestamp is checked).
3. **Part 1 Kaggle run**: new Kaggle notebook from `notebooks/p1_rag.ipynb`, attach the competition data + a small
   dataset with `golden_set.json`, GPU T4, Internet on, Run All. Submit `submission.csv` from the Output tab.
   Read the per-category errors in section 5 and the leaderboard score; improve **generic** behaviour only
   (no question-specific rules), re-run.
4. **Part 1 human evaluation**: both members independently fill `rater_A` / `rater_B` (1-5) in `p1_human_eval.csv`
   from the output, add it to the dataset, re-run sections 6-8.
5. **Part 3**: set `LORA_REPO` (your Part 2 LoRA on Hugging Face) in the config cell of `notebooks/p3_agents.ipynb`,
   run it on Kaggle/Colab T4. Both members fill the two human sheets (task success pass/fail, critic accept/revise),
   re-run sections 6-7.
6. **Report**: copy the CSV/JSON outputs into `report/results/`, run `python report/make_tables.py`, compile
   `report/main.tex` (Overleaf, IEEEtran), complete every *[bracketed]* part in your own words.
7. **CONTRIBUTIONS.md** (who did what + meeting recordings), **AI_USAGE.md** (check it is complete), **links.txt**.
8. **Tags**: `git tag p1-v1.0 && git tag p3-v1.0 && git push origin --tags` (and `p2-v1.0`). Regenerate the
   notebooks right before tagging so cell 1 prints the tagged commit.
9. Canvas zip `Pair_Programming_Team_##.zip`: report PDF, CONTRIBUTIONS.md, AI_USAGE.md, links.txt.
10. Demo prep: be able to explain every module (Part 1: `ingest.py`, `retrieve.py`, `answer.py`, `memory.py`;
    Part 3: `agents.py`, `graph.py`, `safety.py`).
