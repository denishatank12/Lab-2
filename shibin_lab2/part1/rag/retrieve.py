"""Task 1.2.1-1.2.2 - retrieval.

baseline : dense (bge-small + Chroma) only.
final    : dense + BM25 fused with Reciprocal Rank Fusion, plus exact figure /
           table lookup when the question names a label ("Figure A-5"), plus an
           optional cross-encoder reranker (bonus task).

Exact lookup: a named label returns exactly that object. Look-alike figures
that the question did not name are dropped from the evidence so their values
cannot be mixed in; the "(corrected)" version of a named figure is kept as a
labelled correction, because a stated correction replaces the earlier value.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .config import Config
from .index import KnowledgeBase
from .ingest import Item, find_labels

DOC_ALIASES = {
    "DOC1": [r"\bdoc(ument)?\s*1\b", r"\bdoc1\b", r"\bblanchard\b", r"\btext ?book\b", r"\bmacroeconomics\b"],
    "DOC2": [r"\bdoc(ument)?\s*2\b", r"\bdoc2\b", r"\bsupplement\s*a\b", r"\boutturns?\b"],
    "DOC3": [r"\bdoc(ument)?\s*3\b", r"\bdoc3\b", r"\bbriefing\s*b\b"],
}


def docs_mentioned(q: str) -> set[str]:
    ql = (q or "").lower()
    return {d for d, pats in DOC_ALIASES.items() if any(re.search(p, ql) for p in pats)}


@dataclass
class Retrieval:
    ranked: list[Item]  # full ranked list (for Recall@k)
    evidence: list[Item]  # what the generator sees
    named_labels: list[str] = field(default_factory=list)
    pinned: list[Item] = field(default_factory=list)
    blocked: list[Item] = field(default_factory=list)  # named but untrusted (excluded) objects
    missing_labels: list[str] = field(default_factory=list)


def rrf(rank_lists: list[list[str]], k: int = 60, weights=None) -> list[str]:
    weights = weights or [1.0] * len(rank_lists)
    score = {}
    for w, lst in zip(weights, rank_lists):
        for r, i in enumerate(lst):
            score[i] = score.get(i, 0.0) + w / (k + r + 1)
    return [i for i, _ in sorted(score.items(), key=lambda x: -x[1])]


class Retriever:
    def __init__(self, kb: KnowledgeBase, cfg: Config):
        self.kb = kb
        self.cfg = cfg
        self._reranker = None

    @property
    def reranker(self):
        if self._reranker is None:
            from sentence_transformers import CrossEncoder
            import torch
            self._reranker = CrossEncoder(self.cfg.reranker_model, device="cuda" if torch.cuda.is_available() else "cpu")
        return self._reranker

    # ------------------------------------------------------------------
    def lookup(self, labels: list[str], question: str) -> tuple[list[Item], list[Item], list[str]]:
        """Exact figure/table objects for the labels named in the question."""
        want_docs = docs_mentioned(question)
        pinned, blocked, missing = [], [], []
        for lab in labels:
            cands = list(self.kb.figures.get(lab, []))
            if not lab.endswith("(corrected)"):
                corr = self.kb.figures.get(lab + " (corrected)", [])
            else:
                corr = []
            if not cands and not corr:
                missing.append(lab)
                continue
            if want_docs:
                c2 = [c for c in cands if c.doc_id in want_docs]
                cands = c2 or cands
                corr = [c for c in corr if c.doc_id in want_docs] or corr
            for c in corr + cands:  # a corrected version first: its values replace the original's
                (blocked if c.excluded else pinned).append(c)
        # de-duplicate, keep order
        seen, out = set(), []
        for p in pinned:
            if p.id not in seen:
                seen.add(p.id)
                out.append(p)
        return out, blocked, missing

    def retrieve(self, question: str, context_query: str = "", carry_labels: list[str] | None = None,
                 mode: str | None = None, use_reranker: bool | None = None) -> Retrieval:
        cfg = self.cfg
        mode = mode or cfg.mode
        use_reranker = cfg.use_reranker if use_reranker is None else use_reranker
        n = cfg.candidates
        dense = [i for i, _ in self.kb.dense(question, n)]
        if mode == "baseline":
            ranked_ids = dense
            if context_query:
                ranked_ids = rrf([dense, [i for i, _ in self.kb.dense(context_query, n)]], cfg.rrf_k, [1.0, 0.5])
            ranked = [self.kb.by_id[i] for i in ranked_ids]
            return Retrieval(ranked=ranked, evidence=ranked[:cfg.top_k])

        lists = [dense, [i for i, _ in self.kb.sparse(question, n)]]
        weights = [1.0, 1.0]
        if context_query:  # follow-up turn: also search with the earlier turn's wording
            lists += [[i for i, _ in self.kb.dense(context_query, n)], [i for i, _ in self.kb.sparse(context_query, n)]]
            weights += [0.5, 0.5]
        fused = rrf(lists, cfg.rrf_k, weights)

        want_docs = docs_mentioned(question)
        if want_docs:  # soft preference for a document the user named
            fused = [i for i in fused if self.kb.by_id[i].doc_id in want_docs] + \
                    [i for i in fused if self.kb.by_id[i].doc_id not in want_docs]

        if use_reranker:
            pool = fused[:cfg.rerank_pool]
            q = question if not context_query else f"{context_query} {question}"
            scores = self.reranker.predict([(q, self.kb.by_id[i].text[:2000]) for i in pool], show_progress_bar=False)
            pool = [i for _, i in sorted(zip(scores, pool), key=lambda x: -x[0])]
            fused = pool + fused[cfg.rerank_pool:]

        labels = find_labels(question) or list(carry_labels or [])
        pinned, blocked, missing = self.lookup(labels, question) if labels else ([], [], [])
        pinned_ids = {p.id for p in pinned}
        ranked = pinned + [self.kb.by_id[i] for i in fused if i not in pinned_ids]

        if labels:
            # never show look-alike figures the user did not name
            named_bases = {lab.replace(" (corrected)", "") for lab in labels}
            evidence = pinned + [it for it in ranked[len(pinned):]
                                 if not it.figure_id or it.figure_id.replace(" (corrected)", "") in named_bases]
        else:
            evidence = ranked
        evidence = self.cover_documents(evidence, cfg.top_k, cfg.per_doc_min, keep=len(pinned))
        if not labels:
            evidence = evidence + self.referenced_objects(evidence, cfg.follow_refs)
        return Retrieval(ranked=ranked, evidence=evidence, named_labels=labels, pinned=pinned,
                         blocked=blocked, missing_labels=missing)

    def referenced_objects(self, evidence: list[Item], n: int) -> list[Item]:
        """Tables/figures that the retrieved text explicitly points to ("Table A-2 reports ...")."""
        if n <= 0:
            return []
        have = {it.id for it in evidence}
        found = []
        objects = [o for objs in self.kb.figures.values() for o in objs if not o.excluded]
        for it in evidence:
            if it.figure_id:
                continue
            cands = [o for lab in find_labels(it.text) for o in self.kb.figures.get(lab, []) if o.doc_id == it.doc_id]
            # tables printed on the same page as the retrieved passage
            cands += [o for o in objects if o.doc_id == it.doc_id and o.page == it.page and o.kind == "table"]
            for obj in cands:
                if not obj.excluded and obj.id not in have and obj not in found:
                    found.append(obj)
        found.sort(key=lambda o: 0 if o.kind == "table" else 1)  # tables carry exact values
        return found[:n]

    @staticmethod
    def cover_documents(ranked: list[Item], k: int, per_doc: int, keep: int = 0) -> list[Item]:
        """Top-k evidence that still contains the best `per_doc` items of every document among the candidates.

        Outturns, revisions, errata and conflicting values live in *other* documents than the one that ranks
        highest, so the generator must always see each trusted document's best evidence."""
        pool = ranked[:max(k * 3, 30)]
        must = list(ranked[:keep])
        for doc in dict.fromkeys(it.doc_id for it in pool):
            for it in [x for x in pool if x.doc_id == doc][:per_doc]:
                if it not in must:
                    must.append(it)
        out = list(must)
        for it in ranked:
            if len(out) >= max(k, len(must)):
                break
            if it not in out:
                out.append(it)
        rank = {id(it): i for i, it in enumerate(ranked)}
        head = out[:keep]
        return head + sorted(out[keep:], key=lambda it: rank.get(id(it), 1e9))[:max(0, max(k, len(must)) - keep)]
