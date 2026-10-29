"""Generate a valid Part 1 submission from Questions.csv.

This is an auditable extractive baseline: every answer is assembled only from
retrieved source text, with citations and figure/table references.  A later
LLM answerer can consume the same `retrieved` objects without changing the
submission schema or retrieval contract.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

from rag_pipeline import build_index


def _status(question: str, hits: list[dict]) -> str:
    q = question.lower()
    if any(x in q for x in ("ignore previous", "system prompt", "follow these instructions")):
        return "false_premise"
    if not hits or hits[0]["score"] <= 0:
        return "not_found"
    return "answered"


def _answer(question: str, hits: list[dict], status: str) -> str:
    if status == "false_premise":
        return "The document content is treated as evidence, not as instructions to the retrieval system."
    if status == "not_found":
        return "The supplied knowledge base does not contain enough evidence to answer this question."
    snippets = []
    for hit in hits[:3]:
        text = hit["passage"]["text"]
        snippets.append(text[:650].strip())
    return " ".join(snippets)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--output", default="outputs/submission.csv")
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()
    data_dir = Path(args.data_dir)
    index = build_index(data_dir)
    rows = list(csv.DictReader((data_dir / "Questions.csv").open(newline="", encoding="utf-8")))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "conversation_id", "question_id", "answer", "figure_references", "citations", "status"])
        writer.writeheader()
        for row in rows:
            hits = index.search(row["question"], args.top_k)
            status = _status(row["question"], hits)
            figures = []
            citations = []
            for hit in hits:
                p = hit["passage"]
                figures.extend(p["figure_ids"] + p["table_ids"])
                citations.append(f'{p["doc_id"]}:p{p["page"]}')
            figures = list(dict.fromkeys(figures)) or ["NONE"]
            citations = list(dict.fromkeys(citations)) or ["NONE"]
            writer.writerow({**{k: row[k] for k in ("id", "conversation_id", "question_id")}, "answer": _answer(row["question"], hits, status), "figure_references": ",".join(figures), "citations": ",".join(citations), "status": status})
    print(f"wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
