# report/results

Files written by the executed notebooks; the report's tables are built from them (`report/make_tables.py`).

- `p1_kaggle_run1_results.json`, `p1_kaggle_run1_submission.csv`: output of the first complete Kaggle run of
  `notebooks/p1_rag.ipynb` (Tesla T4, code commit `5d030d6`). The submission is the file the notebook wrote
  (`/kaggle/working/submission.csv`), unedited; it passed the official `validate_submission.py`. It is kept as a
  record of that run: the answers are produced by the pipeline at run time, nothing in the code refers to these
  questions or answers.
- `p2_*`: Part 2 results tables, results JSON, image grid, failure-mode and memorization figures.
- `p3_*`: Part 3 results, red-team results and the human rating sheets.
