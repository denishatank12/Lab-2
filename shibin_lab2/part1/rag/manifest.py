"""Read kb_manifest.json (document IDs, file names, trust levels).

The loader is tolerant of the exact JSON layout: it walks the structure and
collects every object that names a document (doc_id / id) together with a file
name and a trust level. Section-level trust (e.g. an appendix marked "low") is
picked up from nested objects that carry a section/title/name and a trust.
"""
from __future__ import annotations

import json
import os
import re

from .config import DEFAULT_TRUST, DOC_FILES, MANIFEST_FILE, UNTRUSTED_SECTIONS

_TRUST_WORDS = ("high", "medium", "low")


def _norm_trust(v) -> str | None:
    if v is None:
        return None
    s = str(v).strip().lower()
    for w in _TRUST_WORDS:
        if w in s:
            return w
    return None


def _first(d: dict, *keys):
    lower = {k.lower(): v for k, v in d.items()}
    for k in keys:
        if k in lower:
            return lower[k]
    return None


def load_manifest(data_dir: str) -> dict:
    """Return {doc_id: {"file": str, "trust": str, "sections": {name: trust}, "raw": dict}}."""
    docs = {d: {"file": f, "trust": DEFAULT_TRUST[d], "sections": {}, "raw": {}}
            for d, f in DOC_FILES.items()}
    path = os.path.join(data_dir, MANIFEST_FILE)
    raw = None
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)

    def visit(node, parent_doc=None):
        if isinstance(node, dict):
            doc_id = _first(node, "doc_id", "docid", "document_id", "id")
            fname = _first(node, "file", "file_name", "filename", "path", "name")
            trust = _norm_trust(_first(node, "trust", "trust_level", "trustlevel"))
            doc_key = None
            if isinstance(doc_id, str) and re.fullmatch(r"DOC\d+", doc_id.strip().upper()):
                doc_key = doc_id.strip().upper()
            elif isinstance(fname, str) and fname.lower().endswith(".pdf"):
                for d, f in DOC_FILES.items():
                    if os.path.basename(fname) == f:
                        doc_key = d
            if doc_key:
                entry = docs.setdefault(doc_key, {"file": "", "trust": "medium", "sections": {}, "raw": {}})
                if isinstance(fname, str) and fname.lower().endswith(".pdf"):
                    entry["file"] = os.path.basename(fname)
                if trust:
                    entry["trust"] = trust
                entry["raw"] = node
                parent_doc = doc_key
            else:
                sect = _first(node, "section", "title", "name", "part", "appendix")
                if parent_doc and isinstance(sect, str) and trust:
                    docs[parent_doc]["sections"][sect.strip()] = trust
                    pages = _first(node, "pages", "page")
                    if trust == "low" and pages is not None:  # page-level override, e.g. "pages": [7]
                        for pg in (pages if isinstance(pages, list) else [pages]):
                            try:
                                docs[parent_doc].setdefault("low_pages", set()).add(int(pg))
                            except (TypeError, ValueError):
                                pass
            for k, v in node.items():
                # keys like {"DOC1": {...}} or {"Appendix C": "low"}
                if isinstance(k, str) and re.fullmatch(r"DOC\d+", k.strip().upper()):
                    if isinstance(v, (dict, list)):
                        if isinstance(v, dict) and not _first(v, "doc_id", "id"):
                            v = dict(v, doc_id=k)
                        visit(v, k.strip().upper())
                    elif _norm_trust(v):
                        docs.setdefault(k.upper(), {"file": "", "trust": "medium", "sections": {}, "raw": {}})["trust"] = _norm_trust(v)
                    continue
                if parent_doc and isinstance(k, str) and re.match(r"(appendix|section)\b", k.strip(), re.I) and _norm_trust(v):
                    docs[parent_doc]["sections"][k.strip()] = _norm_trust(v)
                    continue
                if isinstance(v, (dict, list)):
                    visit(v, parent_doc)
        elif isinstance(node, list):
            for x in node:
                visit(x, parent_doc)

    if raw is not None:
        visit(raw)
    # The assignment states Appendix C of DOC3 is low trust / untrusted.
    for d, s in UNTRUSTED_SECTIONS:
        docs[d]["sections"].setdefault(s, "low")
    return {"docs": docs, "raw": raw}


def untrusted_sections(manifest: dict) -> dict[str, list[str]]:
    """doc_id -> list of section names whose trust is low."""
    out = {}
    for d, e in manifest["docs"].items():
        out[d] = [s for s, t in e["sections"].items() if t == "low"]
    return out
