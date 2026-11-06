"""Generate a valid Part 1 submission from Questions.csv.

This is an auditable extractive baseline: every answer is assembled only from
retrieved source text, with citations and figure/table references.  A later
LLM answerer can consume the same `retrieved` objects without changing the
submission schema or retrieval contract.
"""
from __future__ import annotations

import argparse
import os
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
    ap.add_argument("--questions-path", default=os.environ.get("QUESTIONS_PATH"))
    ap.add_argument("--lexical-only", action="store_true")
    ap.add_argument("--require-dense", action="store_true")
    ap.add_argument("--dense-only", action="store_true", help="Required baseline: dense retrieval without BM25/CLIP fusion.")
    args = ap.parse_args()
    data_dir = Path(args.data_dir)
    if not args.lexical_only:
        from multimodal import MultimodalIndex, Conversation
        import time
        index = MultimodalIndex(data_dir, Path(args.output).parent / "index")
        questions = Path(args.questions_path) if args.questions_path else data_dir / "Questions.csv"
        with questions.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        conversations, answers, traces = {}, {}, []
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        groups = {}
        for row in rows:
            groups.setdefault(row["conversation_id"], []).append(row)
        try:
            for conv_id, turns in groups.items():
                chat = Conversation(index, dense_only=args.dense_only)
                # Share the model, never the conversation state.
                if conversations:
                    chat.model = next(iter(conversations.values())).model
                conversations[conv_id] = chat
                for row in sorted(turns, key=lambda r: int(r["turn"])):
                    started = time.perf_counter()
                    answer = chat.answer(row["question"])
                    answers[row["id"]] = dict(id=row["id"], conversation_id=conv_id, question_id=row["question_id"],
                        answer=answer["answer"], status=answer["status"],
                        figure_references=",".join(answer["figure_references"]) or "NONE",
                        citations=",".join(answer["citations"]) or "NONE")
                    traces.append(dict(**row, result=answer, retrieved=chat.last_hits,
                                       seconds=time.perf_counter()-started))
                    output.with_suffix(".traces.json").write_text(json.dumps(traces, indent=2))
                    print(row["id"], answer["status"], flush=True)
            with output.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["id", "conversation_id", "question_id", "answer", "figure_references", "citations", "status"])
                writer.writeheader()
                writer.writerows(answers[row["id"]] for row in rows)
            print(f"wrote {len(rows)} model-generated rows to {output}")
        finally:
            if conversations:
                next(iter(conversations.values())).model.unload()
            index.db.close()
        return
    index = build_index(data_dir, use_dense=not args.lexical_only, require_dense=args.require_dense)
    questions = Path(args.questions_path) if args.questions_path else data_dir / "Questions.csv"
    rows = list(csv.DictReader(questions.open(newline="", encoding="utf-8")))
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
