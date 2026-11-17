# Which files to run

No new full run is needed for the completed Part 2 RTX 4090 results or the
completed Part 3 T4 baseline. Their executed notebooks are preserved unchanged.

Part 1: the selected independent v6 source is
`part1_rag/notebooks/Lab2_Part1_Kaggle_Final.ipynb`. Recover its original executed
copy before considering a new run. A new run would be a new measured result,
not proof of the earlier 0.510 submission. Import the single file into Kaggle,
enable Internet and one GPU, and run all cells only if a new run is necessary.

Part 3: the standalone baseline is for reproducibility, not an instruction to
repeat the completed baseline. It embeds the attributed source and downloads
the public rank-4 adapter, so there is no token or separate Python upload needed.

Run `tools/final_submission_audit.py` and `tools/build_combined_report.py` after
adding genuine rating/access/recording evidence. These only audit/package files;
they do not train, perform inference, submit to Kaggle or upload to Canvas.
