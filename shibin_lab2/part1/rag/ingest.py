"""Task 1.1 - ingestion: text chunks + numbered figures/tables from the PDFs.

Outputs a list of *items*. Every item carries doc_id, page (1-based),
figure_id ("" for text), trust_level, section and an `excluded` flag
(True for untrusted sections such as DOC3 Appendix C - stored, never used).

Figures/tables:
  * a caption line that starts with "Figure X-n" / "Table X-n" (optionally
    "(corrected)") defines one object; "Figure B-5" and
    "Figure B-5 (corrected)" become two different objects;
  * the graphic is found from the page's raster images, vector drawings and
    detected tables, and cropped *together with its caption* to a PNG;
  * text printed inside the graphic (axis/data labels of vector charts) and
    table cells are extracted as well, so numbers can be retrieved by text.
"""
from __future__ import annotations

import os
import re
import statistics
from dataclasses import dataclass, field, asdict

import pymupdf

from .config import Config
from .manifest import load_manifest, untrusted_sections
from .safety import sanitize

LABEL_RE = re.compile(
    r"^\s*(Figure|Fig\.?|Table|Chart|Exhibit)\s+([A-Z]?\s?[-‐-―]?\s?\d+(?:\s?[-‐-―.]\s?\d+)?[a-z]?)"
    r"(\s*\((?:corrected|revised|updated)\))?",
    re.I)
TOC_RE = re.compile(r"(\.\s?){3,}|…|\s\.{2,}\s*\d+\s*$")
APPENDIX_HEAD_RE = re.compile(r"^\s*Appendix\s+([A-Z])\b", re.I)
DASHES = dict.fromkeys(map(ord, "‐‑‒–—―−"), "-")


def norm_label(kind: str, num: str, corrected: bool = False) -> str:
    kind = kind.lower().rstrip(".")
    kind = {"fig": "Figure", "figure": "Figure", "table": "Table", "chart": "Chart", "exhibit": "Exhibit"}[kind]
    num = re.sub(r"\s+", "", num.translate(DASHES))
    num = re.sub(r"^([A-Z])(\d)", r"\1-\2", num.upper())  # "A3" -> "A-3"
    return f"{kind} {num}" + (" (corrected)" if corrected else "")


def find_labels(text: str) -> list[str]:
    """All figure/table labels mentioned anywhere in a string (e.g. a question)."""
    out = []
    pat = re.compile(
        r"\b(Figure|Fig\.?|Table|Chart|Exhibit)\s+([A-Z]?\s?[-‐-―]?\s?\d+(?:\s?[-‐-―.]\s?\d+)?)"
        r"(\s*\((?:corrected|revised|updated)\))?", re.I)
    for m in pat.finditer(text or ""):
        lab = norm_label(m.group(1), m.group(2), bool(m.group(3)))
        if lab not in out:
            out.append(lab)
    # plural lists: "Figures A-6 and A-7", "Tables 1-2, 1-3 and 1-4"
    num = r"[A-Z]?\s?[-\u2010-\u2015]\s?\d+|\d+[-\u2010-\u2015.]\d+"
    for m in re.finditer(rf"\b(Figures|Figs\.?|Tables)\s+((?:(?:{num})(?:\s*\((?:corrected)\))?(?:\s*,\s*|\s+and\s+|\s*&\s*)?)+)", text or "", re.I):
        kind = "Table" if m.group(1).lower().startswith("tab") else "Figure"
        for n in re.finditer(rf"({num})(\s*\(corrected\))?", m.group(2), re.I):
            lab = norm_label(kind, n.group(1), bool(n.group(2)))
            if lab not in out:
                out.append(lab)
    return out


@dataclass
class Item:
    id: str
    kind: str  # "text" | "figure" | "table"
    doc_id: str
    page: int
    text: str  # text that is embedded / BM25-indexed
    trust_level: str
    section: str = ""
    figure_id: str = ""
    excluded: bool = False
    has_injection: bool = False
    removed_text: list = field(default_factory=list)
    caption: str = ""
    figure_text: str = ""
    table_text: str = ""
    description: str = ""
    image_path: str = ""
    bbox: list = field(default_factory=list)
    crop_method: str = ""
    source_file: str = ""

    @property
    def cite_key(self) -> str:
        return f"{self.doc_id}:{self.figure_id}" if self.figure_id else f"{self.doc_id}:p{self.page}"

    def to_dict(self):
        d = asdict(self)
        d["cite_key"] = self.cite_key
        return d


# --------------------------------------------------------------------------
def _lines(page):
    """Text lines with font info: list of dict(text, bbox, size, bold, block)."""
    out = []
    d = page.get_text("dict")
    for bi, b in enumerate(d["blocks"]):
        if b.get("type") != 0:
            continue
        for ln in b["lines"]:
            spans = [s for s in ln["spans"] if s["text"].strip()]
            if not spans:
                continue
            text = "".join(s["text"] for s in ln["spans"]).strip()
            size = max(s["size"] for s in spans)
            bold = any((s["flags"] & 16) or "bold" in s["font"].lower() for s in spans)
            out.append({"text": text, "bbox": pymupdf.Rect(ln["bbox"]), "size": size, "bold": bold, "block": bi})
    return out


def _body_size(doc) -> float:
    sizes = []
    for p in doc:
        for ln in _lines(p):
            sizes.append(round(ln["size"], 1))
    return statistics.median(sizes) if sizes else 10.0


def _is_heading(ln, body):
    words = len(ln["text"].split())
    return (ln["size"] >= body + 0.8 or ln["bold"]) and words <= 18 and not TOC_RE.search(ln["text"])


def _graphic_rects(page) -> list[pymupdf.Rect]:
    rects = []
    for info in page.get_image_info():
        r = pymupdf.Rect(info["bbox"])
        if r.width > 20 and r.height > 20:
            rects.append(r)
    try:
        clusters = page.cluster_drawings()
    except Exception:  # older PyMuPDF
        clusters = [pymupdf.Rect(d["rect"]) for d in page.get_drawings()]
    for r in clusters:
        r = pymupdf.Rect(r)
        if r.width > 40 and r.height > 25:
            rects.append(r)
    try:
        for t in page.find_tables().tables:
            rects.append(pymupdf.Rect(t.bbox))
    except Exception:
        pass
    # merge rectangles that overlap (a table grid + its cluster, chart parts)
    merged = []
    for r in sorted(rects, key=lambda r: (r.y0, r.x0)):
        for i, m in enumerate(merged):
            if m.intersects(r) or (abs(m.y1 - r.y0) < 4 and m.x0 < r.x1 and r.x0 < m.x1):
                merged[i] = m | r
                break
        else:
            merged.append(r)
    # a second pass, since unions can create new overlaps
    changed = True
    while changed:
        changed = False
        for i in range(len(merged)):
            for j in range(i + 1, len(merged)):
                if merged[i].intersects(merged[j]):
                    merged[i] = merged[i] | merged[j]
                    merged.pop(j)
                    changed = True
                    break
            if changed:
                break
    page_area = page.rect.width * page.rect.height
    return [m for m in merged if m.get_area() < 0.9 * page_area]


def _v_overlap(a, b) -> float:
    return max(0.0, min(a.y1, b.y1) - max(a.y0, b.y0))


def _h_overlap(a, b) -> float:
    return max(0.0, min(a.x1, b.x1) - max(a.x0, b.x0))


_NUM_CELL = re.compile(r"[-\u2212\u2013]?\$?\d[\d.,]*\*?%?(\s*\([-\u2212\u2013]?\d[\d.,]*\*?\))?")


def _cell_text(page, rect) -> str:
    if rect is None:
        return ""
    t = page.get_text("text", clip=pymupdf.Rect(rect)).strip()
    t = re.sub(r"\s+", " ", t).replace("\u2212", "-").replace("\u2013", "-")
    return t.replace("**", "").strip()


def _is_year(c: str) -> bool:
    return bool(re.fullmatch(r"(19|20)\d\d\*?|(19|20)\d\d\s*-\s*(19|20)\d\d\*?", c.strip()))


def _is_num(c: str) -> bool:
    return bool(_NUM_CELL.fullmatch(c.strip()))


def table_facts(grid: list[list[str]]) -> list[str]:
    """Turn a table grid into self-describing lines: 'Row label: Column = value; ...'.

    Header rows (mostly non-numeric) are merged column by column; leading text columns are the row labels and
    are forward-filled (group rows such as 'United States' followed by 'Output growth', 'Unemployment rate')."""
    grid = [[c or "" for c in r] for r in grid if any((c or "").strip() for c in r)]
    if not grid:
        return []
    ncol = max(len(r) for r in grid)
    grid = [r + [""] * (ncol - len(r)) for r in grid]
    header, data = [], []
    for r in grid:
        vals = [c for c in r[1:] if c.strip()]
        values = [c for c in vals if _is_num(c) and not _is_year(c)]  # years are header labels, not data
        if not data and (not vals or len(values) < 0.5 * len(vals)):
            header.append(r)
        else:
            data.append(r)
    names = []
    for j in range(ncol):
        parts = [h[j] for h in header if h[j].strip() and not re.fullmatch(r"Col\d+", h[j].strip())]
        names.append(" ".join(parts).strip() or f"column {j + 1}")
    lines = []
    if header and names[0] and not names[0].startswith("column"):
        lines.append(f"(first column: {names[0]})")
    fill = [""] * ncol
    for r in data:
        # column 0 is always a row label (it may be a year, e.g. "2013"); values start after the text labels
        first_num = next((j for j in range(1, ncol) if _is_num(r[j])), ncol)
        if first_num == ncol:  # a group header row (only labels)
            for j in range(ncol):
                if r[j].strip():
                    fill[j] = r[j].strip()
                    for k in range(j + 1, ncol):
                        fill[k] = ""
                    break
            continue
        labels = []
        for j in range(first_num):
            if r[j].strip():
                fill[j] = r[j].strip()
                for k in range(j + 1, first_num):
                    if not r[k].strip():
                        fill[k] = ""
            if fill[j]:
                labels.append(fill[j])
        cells = [f"{names[j]} = {r[j].strip()}" for j in range(first_num, ncol) if r[j].strip()]
        if cells:
            lines.append(f"{', '.join(labels) or 'row'}: " + "; ".join(cells))
    return lines


def _table_text(page, clip) -> str:
    try:
        tabs = page.find_tables(clip=clip).tables
    except Exception:
        return ""
    out = []
    for t in tabs:
        grid = []
        try:
            for row in t.rows:
                grid.append([_cell_text(page, c) for c in row.cells])
        except Exception:
            grid = [[re.sub(r"\s+", " ", c or "").strip() for c in row] for row in t.extract()]
        out += table_facts(grid)
    return "\n".join(out)


# --------------------------------------------------------------------------
def ingest(cfg: Config, log=print) -> list[Item]:
    data_dir = cfg.resolved_data_dir()
    manifest = load_manifest(data_dir)
    low_sections = untrusted_sections(manifest)
    fig_dir = os.path.join(cfg.work_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    items: list[Item] = []

    for doc_id, meta in sorted(manifest["docs"].items()):
        path = os.path.join(data_dir, meta["file"])
        if not os.path.isfile(path):
            log(f"[ingest] WARNING missing {path}")
            continue
        doc = pymupdf.open(path)
        body = _body_size(doc)
        low = [re.match(r"(appendix\s+[A-Z])", s, re.I) for s in low_sections.get(doc_id, [])]
        low_names = {m.group(1).title() for m in low if m}
        section = ""
        seen_labels: dict[str, int] = {}
        n_text = n_fig = 0

        for pno, page in enumerate(doc, start=1):
            lines = _lines(page)
            # ---- section tracking (per line, so a page can switch section) ----
            line_section = []
            for ln in lines:
                m = APPENDIX_HEAD_RE.match(ln["text"])
                if m and _is_heading(ln, body):
                    section = f"Appendix {m.group(1).upper()}"
                line_section.append(section)

            def trust_for(sec: str) -> tuple[str, bool]:
                if pno in meta.get("low_pages", set()):
                    return "low", True
                if sec and sec in low_names:
                    return "low", True
                st = meta["sections"].get(sec)
                return (st or meta["trust"]), (st == "low")

            # ---- figures & tables ----
            graphics = _graphic_rects(page)
            captions = []
            for li, ln in enumerate(lines):
                m = LABEL_RE.match(ln["text"])
                if not m or TOC_RE.search(ln["text"]):
                    continue
                rest = ln["text"][m.end():].strip()
                # caption, not a sentence that starts with a reference
                # ("Figure A-3 shows ..." in body text is skipped unless bold / punctuated)
                looks_caption = ln["bold"] or rest[:1] in {":", ".", "-", "–", "—", "|", ""} \
                    or bool(m.group(3)) or (rest[:1].isupper() and not re.match(r"(shows|plots|presents|reports|gives|is|was)\b", rest, re.I))
                if not looks_caption:
                    continue
                corrected = bool(m.group(3)) or bool(re.match(r"[\s:.\-–—]*\(?(corrected|revised)\b", rest, re.I))
                label = norm_label(m.group(1), m.group(2), corrected)
                # caption continuation = rest of the same text block
                block_lines = [x["text"] for x in lines if x["block"] == ln["block"] and x["bbox"].y0 >= ln["bbox"].y0 - 1]
                cap_rect = pymupdf.Rect(ln["bbox"])
                for x in lines:
                    if x["block"] == ln["block"]:
                        cap_rect |= x["bbox"]
                if len(block_lines) == 1 and not rest:  # label alone: the title is the next block below
                    nxt = [x for x in lines if x["block"] != ln["block"] and 0 <= x["bbox"].y0 - cap_rect.y1 <= 14
                           and abs(x["bbox"].x0 - cap_rect.x0) <= 12]
                    if nxt:
                        nb = nxt[0]["block"]
                        for x in lines:
                            if x["block"] == nb:
                                block_lines.append(x["text"])
                                cap_rect |= x["bbox"]
                captions.append({"label": label, "kind": "table" if label.startswith("Table") else "figure",
                                 "line": ln, "rect": cap_rect, "text": " ".join(block_lines), "sec": line_section[li]})

            claimed = set()
            pairs = []
            for ci, c in enumerate(captions):
                for gi, g in enumerate(graphics):
                    cr = c["rect"]
                    if _h_overlap(cr, g) >= 0.3 * min(cr.width, g.width):  # caption above / below
                        if g.y1 <= cr.y0 + 2:
                            dist = cr.y0 - g.y1
                        elif g.y0 >= cr.y1 - 2:
                            dist = g.y0 - cr.y1
                        else:
                            dist = 0.0
                    elif _v_overlap(cr, g) > 0 or abs(cr.y0 - g.y0) < 40:  # caption in the side margin
                        dist = (cr.x0 - g.x1) if cr.x0 >= g.x1 - 2 else (g.x0 - cr.x1)
                        dist = max(0.0, dist) + 5  # prefer above/below matches at equal distance
                    else:
                        continue
                    if dist <= 140:
                        pairs.append((dist, ci, gi))
            assign = {}
            for dist, ci, gi in sorted(pairs):
                if ci in assign or gi in claimed:
                    continue
                assign[ci] = gi
                claimed.add(gi)

            for ci, c in enumerate(captions):
                if ci in assign:
                    g = graphics[assign[ci]]
                    region = g | c["rect"]
                    # axis / tick / legend labels sit just outside the drawn lines: take short text nearby
                    near = g + (-45, -12, 45, 40)
                    for x in lines:
                        in_margin = x["bbox"].y0 > page.rect.y1 - 45 or x["bbox"].y1 < page.rect.y0 + 25  # running header/footer
                        if len(x["text"]) <= 60 and near.intersects(x["bbox"]) and not LABEL_RE.match(x["text"]) \
                                and not in_margin:
                            region |= x["bbox"]
                    method = "graphic+caption"
                else:  # fallback: a band above the caption (or below if near the top)
                    r = c["rect"]
                    if r.y0 > 200:
                        region = pymupdf.Rect(page.rect.x0 + 36, max(page.rect.y0, r.y0 - 260), page.rect.x1 - 36, r.y1)
                    else:
                        region = pymupdf.Rect(page.rect.x0 + 36, r.y0, page.rect.x1 - 36, min(page.rect.y1, r.y1 + 260))
                    method = "fallback-band"
                region = (region + (-6, -6, 6, 6)) & page.rect
                label = c["label"]
                if label in seen_labels:  # same label twice in one document
                    seen_labels[label] += 1
                    label = f"{label} [{seen_labels[label]}]"
                else:
                    seen_labels[label] = 1
                safe = re.sub(r"[^A-Za-z0-9]+", "_", f"{doc_id}_{label}").strip("_")
                img_path = os.path.join(fig_dir, f"{safe}.png")
                page.get_pixmap(matrix=pymupdf.Matrix(cfg.figure_zoom, cfg.figure_zoom), clip=region).save(img_path)
                inner = page.get_text("text", clip=region)
                inner = inner.replace(c["text"], " ") if c["text"] in inner else inner
                inner = re.sub(r"[ \t]+", " ", inner).strip()
                ttext = _table_text(page, region)
                trust, excluded = trust_for(c["sec"])
                cap_clean, rm1 = sanitize(c["text"])
                inner_clean, rm2 = sanitize(inner)
                tt_lines = [sanitize(l) for l in ttext.splitlines()]
                tt_clean, rm3 = "\n".join(c for c, _ in tt_lines), [x for _, r in tt_lines for x in r]
                it = Item(id=f"{doc_id}:{label}", kind=c["kind"], doc_id=doc_id, page=pno, text="",
                          trust_level=trust, section=c["sec"], figure_id=label, excluded=excluded,
                          has_injection=bool(rm1 or rm2 or rm3), removed_text=rm1 + rm2 + rm3,
                          caption=cap_clean, figure_text=inner_clean, table_text=tt_clean, image_path=img_path,
                          bbox=[round(v, 1) for v in region], crop_method=method, source_file=meta["file"])
                items.append(it)
                n_fig += 1
            fig_regions = [pymupdf.Rect(i.bbox) for i in items if i.doc_id == doc_id and i.page == pno and i.figure_id]

            # ---- text chunks (outside figure regions), grouped by section ----
            paras: list[tuple[str, str]] = []
            cur_block, cur_text, cur_sec = None, [], None
            for li, ln in enumerate(lines):
                if any(r.contains(ln["bbox"]) or (r & ln["bbox"]).get_area() > 0.6 * ln["bbox"].get_area() for r in fig_regions) \
                        and not LABEL_RE.match(ln["text"]):
                    continue
                if ln["block"] != cur_block or line_section[li] != cur_sec:
                    if cur_text:
                        paras.append((cur_sec, " ".join(cur_text)))
                    cur_block, cur_text, cur_sec = ln["block"], [], line_section[li]
                cur_text.append(ln["text"])
            if cur_text:
                paras.append((cur_sec, " ".join(cur_text)))

            groups: list[tuple[str, list[str]]] = []
            for sec, p in paras:
                if groups and groups[-1][0] == sec:
                    groups[-1][1].append(p)
                else:
                    groups.append((sec, [p]))
            ci = 0
            for sec, ps in groups:
                words = " ".join(ps).split()
                step = max(1, cfg.chunk_words - cfg.chunk_overlap)
                for start in range(0, max(1, len(words)), step):
                    chunk = " ".join(words[start:start + cfg.chunk_words])
                    if len(chunk) < 20:
                        continue
                    clean, removed = sanitize(chunk)
                    trust, excluded = trust_for(sec)
                    items.append(Item(id=f"{doc_id}:p{pno}:c{ci}", kind="text", doc_id=doc_id, page=pno,
                                      text=clean, trust_level=trust, section=sec, excluded=excluded,
                                      has_injection=bool(removed), removed_text=removed, source_file=meta["file"]))
                    ci += 1
                    n_text += 1
                    if start + cfg.chunk_words >= len(words):
                        break
        log(f"[ingest] {doc_id} ({meta['file']}, trust={meta['trust']}): {len(doc)} pages, "
            f"{n_text} text chunks, {n_fig} figures/tables")
        doc.close()

    return items


def build_figure_text(it: Item) -> str:
    """Text used to index a figure/table (caption + labels/cells + VLM description)."""
    parts = [f"{it.figure_id} ({it.doc_id}, page {it.page}).", it.caption]
    if it.table_text:
        parts.append("Table cells: " + it.table_text)
    if it.figure_text:
        parts.append("Text inside the graphic: " + it.figure_text)
    if it.description:
        parts.append("Description: " + it.description)
    return "\n".join(p for p in parts if p)
