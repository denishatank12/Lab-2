"""Turn the notebooks' saved outputs into LaTeX tables for the report (no number is typed by hand).

Put the files the notebooks wrote (Kaggle "Output" tab) into report/results/:
  p1_results_table.csv, p1_golden_results.csv, p1_results.json   (notebooks/p1_rag.ipynb)
  p3_results_table.csv, p3_redteam.csv, p3_results.json         (notebooks/p3_agents.ipynb)
then:
    python report/make_tables.py      -> report/tables/*.tex  (\\input in main.tex)
"""
import json
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
OUT = os.path.join(HERE, "tables")


def esc(v):
    s = str(v)
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#"), ("$", r"\$")]:
        s = s.replace(a, b)
    return s


def tex_table(df: pd.DataFrame, caption: str, label: str, colspec: str | None = None, size=r"\footnotesize",
              wide: bool = False) -> str:
    colspec = colspec or ("l" + "c" * (len(df.columns) - 1))
    env = "table*" if wide else "table"
    lines = [rf"\begin{{{env}}}[t]", r"\centering", size, rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             rf"\begin{{tabular}}{{{colspec}}}", r"\hline", " & ".join(rf"\textbf{{{esc(c)}}}" for c in df.columns) + r" \\",
             r"\hline"]
    for _, r in df.iterrows():
        lines.append(" & ".join(esc(v) for v in r.values) + r" \\")
    lines += [r"\hline", r"\end{tabular}", rf"\end{{{env}}}"]
    return "\n".join(lines)


def pending(name):
    return rf"\fbox{{\parbox{{0.95\columnwidth}}{{\small Results table pending: run the notebook and copy {esc(name)} to report/results/.}}}}"


def main():
    os.makedirs(OUT, exist_ok=True)
    # Part 1
    p = os.path.join(RES, "p1_results_table.csv")
    if os.path.isfile(p):
        df = pd.read_csv(p, dtype=str, keep_default_na=False)
        t = tex_table(df, "Part 1 results on the 25-question golden set (all numbers from notebook p1\\_rag.ipynb).",
                      "tab:p1", "p{4.2cm}ccccc", wide=True)
    else:
        t = pending("p1_results_table.csv")
    open(os.path.join(OUT, "p1_results.tex"), "w", encoding="utf-8").write(t)
    p = os.path.join(RES, "p1_golden_results.csv")
    if os.path.isfile(p):
        g = pd.read_csv(p)
        g = g[g["system"].str.startswith("final (")]
        cat = g.groupby("category")["correct"].agg(["sum", "count"]).reset_index()
        cat["correct"] = cat["sum"].astype(int).astype(str) + "/" + cat["count"].astype(str)
        t = tex_table(cat[["category", "correct"]].rename(columns={"category": "Category", "correct": "Correct (final)"}),
                      "Part 1 final system: correctness by golden-set category.", "tab:p1cat", "lc")
    else:
        t = pending("p1_golden_results.csv")
    open(os.path.join(OUT, "p1_categories.tex"), "w", encoding="utf-8").write(t)
    # Part 2
    p = os.path.join(RES, "p2_training_table.csv")
    if os.path.isfile(p):
        df = pd.read_csv(p, dtype=str, keep_default_na=False)
        keep = ["run", "trainable_params", "learning_rate", "steps", "batch_size", "peak_gpu_memory_gb", "train_time_s", "gpu_hours"]
        t = tex_table(df[[c for c in keep if c in df.columns]], "Part 2 training runs (same data, steps, batch size and LR; "
                      "all values printed by notebook p2\\_finetune.ipynb).", "tab:p2train", wide=True)
    else:
        t = pending("p2_training_table.csv")
    open(os.path.join(OUT, "p2_training.tex"), "w", encoding="utf-8").write(t)
    p = os.path.join(RES, "p2_results_table.csv")
    if os.path.isfile(p):
        df = pd.read_csv(p, dtype=str, keep_default_na=False)
        t = tex_table(df, "Part 2 results: base model (baseline) vs the two LoRA runs.", "tab:p2", wide=True)
    else:
        t = pending("p2_results_table.csv")
    open(os.path.join(OUT, "p2_results.tex"), "w", encoding="utf-8").write(t)
    # Part 3
    p = os.path.join(RES, "p3_results_table.csv")
    if os.path.isfile(p):
        df = pd.read_csv(p, dtype=str, keep_default_na=False)
        t = tex_table(df.T.reset_index().rename(columns={"index": "Measurement", 0: "Baseline", 1: "Final"}).iloc[1:],
                      "Part 3 results: without vs with the critique loop (notebook p3\\_agents.ipynb).", "tab:p3",
                      "p{3.0cm}p{2.2cm}p{2.2cm}")
    else:
        t = pending("p3_results_table.csv")
    open(os.path.join(OUT, "p3_results.tex"), "w", encoding="utf-8").write(t)
    p = os.path.join(RES, "p3_redteam.csv")
    if os.path.isfile(p):
        r = pd.read_csv(p, dtype=str, keep_default_na=False)
        r = r[["id", "category", "status", "attack_success"]]
        t = tex_table(r, "Part 3 red-team cases and outcomes.", "tab:p3red", "llll")
    else:
        t = pending("p3_redteam.csv")
    open(os.path.join(OUT, "p3_redteam.tex"), "w", encoding="utf-8").write(t)
    print(sorted(os.listdir(OUT)))


if __name__ == "__main__":
    main()
