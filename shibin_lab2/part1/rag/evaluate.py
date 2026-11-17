"""Task 1.4 / 1.5 - evaluation and Kaggle submission.

Golden set (part1/eval/golden_set.json), one object per question:
  gid, conversation_id, turn, question, category,
  gold_answer, gold_status, gold_numbers [floats], gold_keywords [str],
  gold_sources ["DOC2:p4", "DOC2:Figure A-3", ...], gold_figures ["Figure A-3"]

Measurements (C2):
  primary   : answer correctness  (numbers within +-1% AND status right) -> n/25
  secondary : Retrieval Recall@5 (a gold page/figure in the top 5) and
              citation accuracy (share of cited sources that support the answer)
  human     : groundedness 1-5 by both members (human_eval.csv) -> mean, within-1 agreement
  efficiency: median seconds per question
"""
from __future__ import annotations

import json
import os
import re
import statistics
import subprocess
import sys
import time

import pandas as pd

from .ingest import find_labels
from .pipeline import RAGSystem

NUM_RE = re.compile(r"(?<![\w.])[-−–]?\d{1,3}(?:,\d{3})+(?:\.\d+)?|(?<![\w.])[-−–]?\d+(?:\.\d+)?")


def numbers_in(text: str) -> list[float]:
    out = []
    for m in NUM_RE.findall(text or ""):
        s = m.replace(",", "").replace("−", "-").replace("–", "-")
        try:
            out.append(float(s))
        except ValueError:
            pass
    return out


def num_match(gold: float, preds: list[float], tol=0.01) -> bool:
    for p in preds:
        if gold == 0:
            if abs(p) <= 1e-9 + tol:
                return True
        elif abs(p - gold) <= tol * abs(gold) + 1e-12:
            return True
        # "minus 0.7" written as "0.7 percent decline": accept the magnitude if the answer says so
        elif gold < 0 and abs(p + gold) <= tol * abs(gold) and p > 0:
            return True
    return False


def is_correct(pred: dict, g: dict) -> tuple[bool, str]:
    if pred.get("status") != g["gold_status"]:
        return False, f"status {pred.get('status')} != {g['gold_status']}"
    nums = numbers_in(pred.get("answer", ""))
    for x in g.get("gold_numbers", []) or []:
        if not num_match(float(x), nums):
            return False, f"missing number {x}"
    ans = (pred.get("answer") or "").lower()
    for kw in g.get("gold_keywords", []) or []:
        alts = [a.strip().lower() for a in str(kw).split("|")]
        if not any(a in ans for a in alts):
            return False, f"missing keyword {kw}"
    return True, "ok"


def _key_matches(cite: str, gold_sources: list[str], item_pages: dict) -> bool:
    """A cited source supports the answer if it is a gold source, or a figure on a
    gold page, or a page that contains a gold figure."""
    cite = cite.strip()
    if cite in gold_sources:
        return True
    doc = cite.split(":")[0]
    page_of = item_pages.get(cite)
    for gs in gold_sources:
        gdoc = gs.split(":")[0]
        if gdoc != doc:
            continue
        m = re.fullmatch(r"DOC\d+:p(\d+)", gs)
        if m and page_of is not None and int(m.group(1)) == page_of:
            return True
        m2 = re.fullmatch(r"DOC\d+:p(\d+)", cite)
        gp = item_pages.get(gs)
        if m2 and gp is not None and int(m2.group(1)) == gp:
            return True
    return False


def recall_at_k(ranked_keys: list[tuple[str, int, str]], gold_sources: list[str], k=5) -> bool:
    """ranked_keys: (doc_id, page, cite_key) of the retrieved items in order."""
    for doc, page, key in ranked_keys[:k]:
        for gs in gold_sources:
            if key == gs:
                return True
            m = re.fullmatch(r"(DOC\d+):p(\d+)", gs)
            if m and m.group(1) == doc and int(m.group(2)) == page:
                return True
    return False


def evaluate_golden(system: RAGSystem, golden: list[dict], label: str = "", log=print) -> tuple[pd.DataFrame, dict]:
    kb = system.kb
    item_pages = {it.cite_key: it.page for it in kb.items}
    rows = []
    golden = [g for g in golden if str(g.get("question", "")).strip()]
    if not golden:
        raise ValueError("the golden set has no questions filled in")
    df = pd.DataFrame(golden)
    for conv, turns in df.groupby("conversation_id", sort=False):
        chat = system.new_conversation(conv)
        for _, g in turns.sort_values("turn").iterrows():
            g = g.to_dict()
            t = time.time()
            pred = chat.answer(g["question"])
            sec = time.time() - t
            ranked = [(k.split(":")[0], item_pages.get(k, -1), k) for k in chat.last.get("retrieved", [])]
            ok, why = is_correct(pred, g)
            gold_src = g.get("gold_sources") or []
            cites = pred.get("citations", [])
            supported = [c for c in cites if _key_matches(c, gold_src, item_pages)]
            rows.append({
                "system": label, "gid": g["gid"], "conversation_id": conv, "turn": g["turn"],
                "category": g.get("category", ""), "question": g["question"],
                "gold_status": g["gold_status"], "pred_status": pred["status"], "correct": ok, "why": why,
                "recall@5": recall_at_k(ranked, gold_src, 5) if gold_src else None,
                "n_cites": len(cites), "n_supported": len(supported),
                "fig_ref_exact": sorted(pred.get("figure_references", [])) == sorted(g.get("gold_figures", []) or []),
                "answer": pred["answer"], "citations": ",".join(cites) or "NONE",
                "figure_references": ",".join(pred.get("figure_references", [])) or "NONE",
                "gold_answer": g.get("gold_answer", ""), "gold_sources": ",".join(gold_src) or "NONE",
                "seconds": sec,
            })
    res = pd.DataFrame(rows)
    rec = res["recall@5"].dropna()
    summary = {
        "system": label,
        "correct": int(res["correct"].sum()), "n": int(len(res)),
        "answer_correctness": float(res["correct"].mean()),
        "status_accuracy": float((res["gold_status"] == res["pred_status"]).mean()),
        "recall@5": float(rec.astype(float).mean()) if len(rec) else float("nan"),
        "recall_n": int(len(rec)),
        "citation_accuracy": float(res["n_supported"].sum() / max(1, res["n_cites"].sum())),
        "figure_ref_exact": float(res["fig_ref_exact"].mean()),
        "median_s_per_q": float(statistics.median(res["seconds"])),
    }
    log(f"[eval] {label}: {summary['correct']}/{summary['n']} correct, R@5={summary['recall@5']:.3f}, "
        f"cite-acc={summary['citation_accuracy']:.3f}, median {summary['median_s_per_q']:.2f}s/q")
    return res, summary


# ---------------------------------------------------------------------------
def human_eval_template(final_rows: pd.DataFrame, path: str) -> str:
    """Write the sheet both members fill in (1-5 groundedness). Never overwrite ratings."""
    if os.path.isfile(path):
        old = pd.read_csv(path, dtype=str, keep_default_na=False)
        if (old.get("rater_A", pd.Series(dtype=str)).str.strip() != "").any():
            return path
    t = final_rows[["gid", "question", "answer", "citations", "figure_references", "gold_sources"]].copy()
    t["rater_A"] = ""
    t["rater_B"] = ""
    t.to_csv(path, index=False)
    return path


def human_eval_stats(path: str) -> dict | None:
    if not os.path.isfile(path):
        return None
    h = pd.read_csv(path, dtype=str, keep_default_na=False)
    a = pd.to_numeric(h["rater_A"], errors="coerce")
    b = pd.to_numeric(h["rater_B"], errors="coerce")
    m = a.notna() & b.notna()
    if m.sum() == 0:
        return None
    return {"n_rated": int(m.sum()), "mean_A": float(a[m].mean()), "mean_B": float(b[m].mean()),
            "mean_groundedness": float(((a[m] + b[m]) / 2).mean()),
            "agree_within_1": float(((a[m] - b[m]).abs() <= 1).mean()),
            "exact_agree": float((a[m] == b[m]).mean())}


# ---------------------------------------------------------------------------
SUB_COLS = ["id", "conversation_id", "question_id", "answer", "figure_references", "citations", "status"]


def make_submission(system: RAGSystem, questions_path: str, out_path="submission.csv", log_path=None, log=print):
    """Exactly the procedure of section 1.5 B (no manual step)."""
    from .pipeline import run_questions
    q = pd.read_csv(questions_path, dtype=str, keep_default_na=False)
    t = time.time()
    rows, logs = run_questions(system, q, log_path)
    out = []
    for a in rows:
        out.append({"id": a["id"], "conversation_id": a["conversation_id"], "question_id": a["question_id"],
                    "answer": " ".join(str(a["answer"]).split()),
                    "figure_references": ",".join(a["figure_references"]) or "NONE",
                    "citations": ",".join(a["citations"]) or "NONE", "status": a["status"]})
    sub = pd.DataFrame(out).set_index("id").loc[q["id"]].reset_index()[SUB_COLS]
    sub.to_csv(out_path, index=False)
    secs = [l.get("seconds", 0.0) for l in logs]
    log(f"[submission] {len(sub)} rows -> {out_path} in {time.time() - t:.0f}s "
        f"(median {statistics.median(secs):.2f}s/q)")
    return sub, logs


def check_submission(sub_path: str, questions_path: str) -> list[str]:
    """Local format check (the official validate_submission.py is run too when present)."""
    errs = []
    s = pd.read_csv(sub_path, dtype=str, keep_default_na=False)
    q = pd.read_csv(questions_path, dtype=str, keep_default_na=False)
    if list(s.columns) != SUB_COLS:
        errs.append(f"columns {list(s.columns)} != {SUB_COLS}")
    if list(s["id"]) != list(q["id"]):
        errs.append("ids/order differ from Questions.csv")
    if list(s["question_id"]) != list(q["question_id"]):
        errs.append("question_id differs")
    bad = ~s["status"].isin(["answered", "not_found", "conflict", "false_premise", "session_override"])
    if bad.any():
        errs.append(f"{bad.sum()} invalid status values")
    for col in ["figure_references", "citations"]:
        if (s[col].str.contains(", ")).any():
            errs.append(f"{col}: lists must be joined with ',' and no spaces")
        if (s[col] == "").any():
            errs.append(f"{col}: empty -> use NONE")
    empty = s["answer"].str.strip() == ""
    if empty.any():
        errs.append(f"{empty.sum()} empty answers (the official validator rejects them)")
    long = s["answer"].str.split().str.len() > 120
    if long.any():
        errs.append(f"{long.sum()} answers over 120 words")
    for v in s["figure_references"]:
        if v != "NONE":
            for lab in v.split(","):
                if not find_labels(lab):
                    errs.append(f"figure reference not a label: {lab}")
    return errs


def run_official_validator(data_dir: str, sub_path: str, questions_path: str) -> str:
    v = os.path.join(data_dir, "validate_submission.py")
    if not os.path.isfile(v):
        return "validate_submission.py not found next to the data"
    r = subprocess.run([sys.executable, v, sub_path, questions_path], capture_output=True, text=True)
    return (r.stdout + r.stderr).strip()


def load_golden(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        g = json.load(fh)
    return g["questions"] if isinstance(g, dict) else g
