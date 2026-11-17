# report/results

Files written by the executed notebooks; the report's tables are built from them (`report/make_tables.py`).

- `p1_kaggle_run1_results.json`, `p1_kaggle_run1_submission.csv`: output of the first complete Kaggle run of
  `notebooks/p1_rag.ipynb` (Tesla T4, code commit `5d030d6`). The submission is the file the notebook wrote
  (`/kaggle/working/submission.csv`), unedited; it passed the official `validate_submission.py`. It is kept as a
  record of that run: the answers are produced by the pipeline at run time, nothing in the code refers to these
  questions or answers.
- `p1_kaggle_run1_log.txt`, `p1_kaggle_run1_answer_log.json`: the Kaggle log of that run (downloaded from the
  version's Logs; 40/40 answered in 38 minutes) and the per-question answer log the notebook wrote (evidence, raw
  model JSON, final status). `notebooks/p1_rag.ipynb` is the executed notebook of this run (Kaggle notebook
  `notebookec4b2aff1d`, Version 2, downloaded with its outputs). Kaggle public score of its submission: 0.805.
  The golden-set section did not run on Kaggle (golden_set.json was not attached). Later code changes in
  `part1/rag/` (commits after 5d030d6) have not been run on Kaggle.
- `p2_*`: Part 2 results tables, results JSON, image grid, failure-mode and memorization figures.
- `p3_*`: Part 3 results, red-team results and the human rating sheets.
