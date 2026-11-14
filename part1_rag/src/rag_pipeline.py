"""Lightweight hybrid retriever for DATA 266 Lab 2 Part 1.

The implementation is deliberately dependency-light so it runs on a laptop,
RTX 4050, Kaggle, or Colab.  Sentence-Transformers is used when installed;
the TF-IDF fallback keeps the pipeline runnable without model downloads.
"""
from __future__ import annotations

import json
import math
import re
import warnings
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass
class Passage:
    doc_id: str
    page: int
    text: str
    trust_level: str
    figure_ids: list[str]
    table_ids: list[str]

    @property
    def citation(self) -> str:
        return f"{self.doc_id}:p{self.page}"


def _tokens(text: str) -> list[str]:
    # Treat dotted initialisms and their usual plain spelling identically.
    text = re.sub(r'\b(?:[A-Za-z]\.){2,}', lambda m: m.group(0).replace('.', ''), text)
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", text.lower())


def _labels(text: str, kind: str) -> list[str]:
    pattern = rf"{kind}\s+([12]|[A-C])-\d+(?:\s+\(corrected\))?"
    return [m.group(0).title().replace("(Corrected)", "(corrected)") for m in re.finditer(pattern, text, flags=re.I)]


def load_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def extract_passages(data_dir: Path, chunk_words: int = 180, overlap_words: int = 35) -> list[Passage]:
    """Extract page-addressable passages and preserve figure/table labels."""
    from pypdf import PdfReader

    manifest = load_manifest(data_dir / "kb_manifest.json")
    passages: list[Passage] = []
    for doc in manifest["documents"]:
        reader = PdfReader(str(data_dir / doc["file"]))
        for page_no, page in enumerate(reader.pages, start=1):
            trust = doc.get("trust_level", "medium")
            for override in doc.get("section_trust_overrides", []):
                if page_no in override.get("pages", []):
                    trust = override["trust_level"]
            if trust == "low":
                continue
            text = re.sub(r"\s+", " ", page.extract_text() or "").strip()
            if not text:
                continue
            words = text.split()
            step = max(1, chunk_words - overlap_words)
            for start in range(0, len(words), step):
                chunk = " ".join(words[start : start + chunk_words]).strip()
                if not chunk:
                    continue
                passages.append(
                    Passage(
                        doc_id=doc["doc_id"],
                        page=page_no,
                        text=chunk,
                        trust_level=trust,
                        figure_ids=_labels(chunk, "figure"),
                        table_ids=_labels(chunk, "table"),
                    )
                )
                if start + chunk_words >= len(words):
                    break
    return passages


class HybridIndex:
    def __init__(self, passages: list[Passage], dense_model: str = "all-MiniLM-L6-v2",
                 use_dense: bool = True, require_dense: bool = False):
        if not passages:
            raise ValueError("No PDF passages were extracted. Check the data folder.")
        self.passages = passages
        self._terms = [_tokens(p.text) for p in passages]
        self._idf: dict[str, float] = {}
        self._avgdl = sum(map(len, self._terms)) / max(1, len(self._terms))
        df: dict[str, int] = {}
        for terms in self._terms:
            for term in set(terms):
                df[term] = df.get(term, 0) + 1
        n = len(self._terms)
        self._idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self._embedder = None
        self._vectors = None
        self.dense_error = None
        if not use_dense:
            return
        try:
            from sentence_transformers import SentenceTransformer

            self._embedder = SentenceTransformer(dense_model, device="cuda" if _cuda_available() else "cpu")
            self._vectors = self._embedder.encode(
                [p.text for p in passages], normalize_embeddings=True, show_progress_bar=False
            )
            if not np.isfinite(self._vectors).all():
                raise RuntimeError("Embedding model returned nonfinite vectors.")
        except Exception as exc:
            self._embedder = None
            self.dense_error = f"{type(exc).__name__}: {exc}"
            if require_dense:
                raise RuntimeError("Dense retrieval failed: " + self.dense_error) from exc
            warnings.warn("Using BM25 only; dense retrieval failed: " + self.dense_error, RuntimeWarning)

    def _bm25(self, query: str) -> np.ndarray:
        q = _tokens(query)
        scores = np.zeros(len(self.passages), dtype=np.float32)
        k1, b = 1.5, 0.75
        for i, terms in enumerate(self._terms):
            counts = Counter(terms)
            dl = len(terms)
            for term in q:
                if term not in counts:
                    continue
                tf = counts[term]
                scores[i] += self._idf.get(term, 0.0) * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / max(1, self._avgdl)))
        return scores

    def _dense(self, query: str) -> np.ndarray:
        if self._embedder is None or self._vectors is None:
            return np.zeros(len(self.passages), dtype=np.float32)
        q = self._embedder.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
        if not np.isfinite(q).all():
            raise RuntimeError("Invalid query embedding; refusing to write misleading results.")
        return np.asarray(self._vectors @ q, dtype=np.float32)

    def search(self, query: str, k: int = 5) -> list[dict]:
        bm25 = self._bm25(query)
        dense = self._dense(query)
        bm25 = bm25 / max(1e-8, float(bm25.max())) if bm25.max() else bm25
        if dense.max() > dense.min():
            dense = (dense - dense.min()) / (dense.max() - dense.min())
        scores = 0.65 * bm25 + 0.35 * dense
        requested = set(re.findall(r"(?:figure|table)\s+\S+", query.lower()))
        for i, p in enumerate(self.passages):
            labels = {x.lower() for x in p.figure_ids + p.table_ids}
            if requested & labels:
                scores[i] += 0.25
        order = np.argsort(-scores)[:k]
        return [{"score": float(scores[i]), "passage": asdict(self.passages[i])} for i in order]


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def build_index(data_dir: str | Path, use_dense=True, require_dense=False) -> HybridIndex:
    data_dir = Path(data_dir)
    return HybridIndex(extract_passages(data_dir), use_dense=use_dense, require_dense=require_dense)


def save_index(index: HybridIndex, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(p) for p in index.passages], ensure_ascii=False, indent=2), encoding="utf-8")
