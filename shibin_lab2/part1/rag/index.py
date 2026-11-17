"""Task 1.1.4 - storage: ChromaDB (dense vectors + metadata) and a BM25 index.

Metadata stored per item: doc_id, page, figure_id, trust_level, kind, section,
excluded, has_injection, cite_key. User-supplied conversation values are never
written here (see memory.py).
"""
from __future__ import annotations

import json
import os
import re

import numpy as np

from .config import Config
from .ingest import Item, build_figure_text

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "  # bge convention
_TOK = re.compile(r"[a-z0-9]+(?:[-.][a-z0-9]+)*")
_STOP = set("""a an the of in on at to for from by and or is are was were be been being what which who whom
whose how why when where this that these those it its as with about into than then there their they them
do does did according value values give state report reported""".split())


def _stem(t: str) -> str:
    """Tiny plural stemmer so 'economies'/'economy' and 'rates'/'rate' match (numbers untouched)."""
    if t[:1].isdigit() or len(t) <= 3:
        return t
    if t.endswith("ies") and len(t) > 4:
        return t[:-3] + "y"
    if t.endswith("s") and not t.endswith(("ss", "us", "is")):
        return t[:-1]
    return t


def tokenize(text: str) -> list[str]:
    toks = _TOK.findall((text or "").lower())
    out = []
    for t in toks:
        if t in _STOP:
            continue
        out.append(_stem(t))
        if "-" in t:  # "a-3" also matches "a 3"; "advanced-economy" also matches "advanced economy"
            out.extend(_stem(p) for p in t.split("-") if p and p not in _STOP)
    return out


def item_text(it: Item) -> str:
    if it.figure_id:
        return build_figure_text(it)
    prefix = f"[{it.doc_id} page {it.page}{' ' + it.section if it.section else ''}] "
    return prefix + it.text


class KnowledgeBase:
    def __init__(self, cfg: Config, items: list[Item], embedder=None, log=print):
        import chromadb
        from rank_bm25 import BM25Okapi
        from sentence_transformers import SentenceTransformer

        self.cfg = cfg
        self.items = items
        self.by_id = {it.id: it for it in items}
        for it in items:
            if it.figure_id:
                it.text = build_figure_text(it)
        self.texts = [item_text(it) for it in items]
        dev = "cuda" if _cuda() else "cpu"
        self.embedder = embedder or SentenceTransformer(cfg.embed_model, device=dev)
        log(f"[index] embedding {len(items)} items with {cfg.embed_model} on {dev}")
        self.emb = self.embedder.encode(self.texts, batch_size=32, normalize_embeddings=True,
                                        show_progress_bar=False).astype(np.float32)

        db_dir = os.path.join(cfg.work_dir, "chroma")
        os.makedirs(db_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=db_dir)
        try:
            self.client.delete_collection("lab2_kb")
        except Exception:
            pass
        self.col = self.client.create_collection("lab2_kb", metadata={"hnsw:space": "cosine"})
        metas = [{
            "doc_id": it.doc_id, "page": int(it.page), "figure_id": it.figure_id, "trust_level": it.trust_level,
            "kind": it.kind, "section": it.section, "excluded": bool(it.excluded),
            "has_injection": bool(it.has_injection), "cite_key": it.cite_key, "image_path": it.image_path,
        } for it in items]
        B = 500
        for s in range(0, len(items), B):
            self.col.add(ids=[it.id for it in items[s:s + B]], embeddings=self.emb[s:s + B].tolist(),
                         documents=self.texts[s:s + B], metadatas=metas[s:s + B])
        self.bm25 = BM25Okapi([tokenize(t) for t in self.texts])
        self.allowed = np.array([not it.excluded for it in items])
        self.figures = {}
        for it in items:
            if it.figure_id:
                self.figures.setdefault(it.figure_id, []).append(it)
        log(f"[index] chroma count={self.col.count()}, figures/tables={sum(len(v) for v in self.figures.values())}, "
            f"excluded (untrusted)={int((~self.allowed).sum())}, injection-flagged={sum(it.has_injection for it in items)}")

    # ------------------------------------------------------------------
    def dense(self, query: str, k: int) -> list[tuple[str, float]]:
        q = self.embedder.encode([QUERY_PREFIX + query], normalize_embeddings=True)[0].tolist()
        res = self.col.query(query_embeddings=[q], n_results=min(k, self.col.count()),
                             where={"excluded": False})
        ids = res["ids"][0]
        d = res["distances"][0]
        return [(i, 1.0 - float(x)) for i, x in zip(ids, d)]

    def sparse(self, query: str, k: int) -> list[tuple[str, float]]:
        scores = self.bm25.get_scores(tokenize(query))
        scores = np.where(self.allowed, scores, -1e9)
        order = np.argsort(-scores)[:k]
        return [(self.items[i].id, float(scores[i])) for i in order if scores[i] > 0]

    def save_items(self, path=None):
        path = path or os.path.join(self.cfg.work_dir, "items.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump([it.to_dict() for it in self.items], fh, indent=1)
        return path


def _cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False
