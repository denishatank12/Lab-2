"""Build the two-column report PDF from REPORT.md.

Usage: python tools/build_report_pdf.py [REPORT.md] [Lab2_Team09_Report.pdf]
Supports the subset of Markdown used in REPORT.md: '#', '##', '###' headings, paragraphs,
pipe tables, images written as ![caption](path), **bold** and `code`.
"""
import re
import sys
from pathlib import Path
from xml.sax.saxutils import escape

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (BaseDocTemplate, Frame, Image, KeepTogether, NextPageTemplate,
                                PageTemplate, Paragraph, Spacer, Table, TableStyle)

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "REPORT.md"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "Lab2_Team09_Report.pdf"

PAGE_W, PAGE_H = letter
MARGIN_X, MARGIN_TOP, MARGIN_BOTTOM, GAP = 0.65 * inch, 0.75 * inch, 0.8 * inch, 0.25 * inch
COL_W = (PAGE_W - 2 * MARGIN_X - GAP) / 2
TITLE_H = 1.25 * inch

S = {
    "title": ParagraphStyle("title", fontName="Times-Roman", fontSize=20, leading=24, alignment=TA_CENTER),
    "author": ParagraphStyle("author", fontName="Times-Italic", fontSize=10.5, leading=13, alignment=TA_CENTER),
    "h2": ParagraphStyle("h2", fontName="Times-Bold", fontSize=10, leading=12, alignment=TA_CENTER,
                         spaceBefore=8, spaceAfter=4),
    "h3": ParagraphStyle("h3", fontName="Times-Italic", fontSize=10, leading=12, spaceBefore=6, spaceAfter=2),
    "p": ParagraphStyle("p", fontName="Times-Roman", fontSize=9.5, leading=11.4, alignment=TA_JUSTIFY,
                        firstLineIndent=10, spaceAfter=3),
    "cap": ParagraphStyle("cap", fontName="Times-Roman", fontSize=8, leading=9.5, alignment=TA_JUSTIFY,
                          spaceBefore=2, spaceAfter=6),
    "cell": ParagraphStyle("cell", fontName="Times-Roman", fontSize=7.2, leading=8.4),
    "cellh": ParagraphStyle("cellh", fontName="Times-Bold", fontSize=7.2, leading=8.4),
}


def inline(text):
    """Escape for reportlab and convert **bold** and `code`."""
    parts = re.split(r"(`[^`]*`)", text)
    out = []
    for part in parts:
        if part.startswith("`") and part.endswith("`") and len(part) > 1:
            out.append('<font face="Courier" size="8">' + escape(part[1:-1]) + "</font>")
        else:
            e = escape(part)
            e = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", e)
            out.append(e)
    return "".join(out)


def roman(n):
    vals = [(10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]
    r = ""
    for v, s in vals:
        while n >= v:
            r += s
            n -= v
    return r


def make_table(rows, caption_next):
    head, body = rows[0], rows[1:]
    data = [[Paragraph(inline(c), S["cellh"]) for c in head]]
    data += [[Paragraph(inline(c), S["cell"]) for c in r] for r in body]
    n = len(head)
    # first column a little wider
    widths = [COL_W * (1.5 / (n + 0.5))] + [COL_W * (1.0 / (n + 0.5))] * (n - 1)
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("LINEABOVE", (0, 0), (-1, 0), 0.8, colors.black),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.black),
        ("LINEBELOW", (0, -1), (-1, -1), 0.8, colors.black),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 2), ("RIGHTPADDING", (0, 0), (-1, -1), 2),
        ("TOPPADDING", (0, 0), (-1, -1), 1.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5),
    ]))
    return t


def make_image(path):
    p = (ROOT / path) if not Path(path).is_absolute() else Path(path)
    with PILImage.open(p) as im:
        w, h = im.size
    width = COL_W
    height = width * h / w
    max_h = 4.2 * inch
    if height > max_h:
        width, height = width * max_h / height, max_h
    return Image(str(p), width=width, height=height)


def build():
    lines = SRC.read_text(encoding="utf-8").splitlines()
    title_flow, body = [], []
    sec, fig, i = 0, 0, 0
    para = []

    def flush():
        if para:
            body.append(Paragraph(inline(" ".join(para)), S["p"]))
            para.clear()

    while i < len(lines):
        line = lines[i].rstrip()
        if line.startswith("# "):
            title_flow.append(Paragraph(inline(line[2:]), S["title"]))
            # author line = next non-empty line
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            title_flow.append(Spacer(1, 6))
            title_flow.append(Paragraph(inline(lines[j].strip()), S["author"]))
            i = j + 1
            continue
        if line.startswith("## "):
            flush()
            sec += 1
            body.append(Paragraph(f"{roman(sec)}. " + inline(line[3:]).upper(), S["h2"]))
        elif line.startswith("### "):
            flush()
            body.append(Paragraph(inline(re.sub(r"^[IVX]+\.\s*", "", line[4:])), S["h3"]))
        elif line.startswith("|"):
            flush()
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-{3,}:?", c) for c in cells):
                    rows.append(cells)
                i += 1
            cap = None
            while i < len(lines) and not lines[i].strip():
                i += 1
            if i < len(lines) and re.match(r"Table [IVX]+\.", lines[i]):
                cap = lines[i].strip()
                i += 1
            items = [make_table(rows, cap)]
            if cap:
                items.insert(0, Paragraph(inline(cap), S["cap"]))
            body.append(KeepTogether(items))
            continue
        elif line.startswith("!["):
            flush()
            m = re.match(r"!\[(.*)\]\((.*)\)", line)
            fig += 1
            body.append(KeepTogether([Spacer(1, 3), make_image(m.group(2)),
                                      Paragraph(f"Fig. {fig}. " + inline(m.group(1)), S["cap"])]))
        elif not line.strip():
            flush()
        else:
            para.append(line.strip())
        i += 1
    flush()

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Times-Roman", 8)
        canvas.drawCentredString(PAGE_W / 2, 0.45 * inch, f"DATA 266 Lab 2 · Pair_Programming_Team_09 · page {doc.page}")
        canvas.restoreState()

    col_h_first = PAGE_H - MARGIN_TOP - MARGIN_BOTTOM - TITLE_H
    col_h = PAGE_H - MARGIN_TOP - MARGIN_BOTTOM
    first = PageTemplate(id="first", onPage=footer, frames=[
        Frame(MARGIN_X, PAGE_H - MARGIN_TOP - TITLE_H, PAGE_W - 2 * MARGIN_X, TITLE_H, id="title"),
        Frame(MARGIN_X, MARGIN_BOTTOM, COL_W, col_h_first, id="c1"),
        Frame(MARGIN_X + COL_W + GAP, MARGIN_BOTTOM, COL_W, col_h_first, id="c2")])
    later = PageTemplate(id="later", onPage=footer, frames=[
        Frame(MARGIN_X, MARGIN_BOTTOM, COL_W, col_h, id="l1"),
        Frame(MARGIN_X + COL_W + GAP, MARGIN_BOTTOM, COL_W, col_h, id="l2")])
    doc = BaseDocTemplate(str(OUT), pagesize=letter, pageTemplates=[first, later],
                          title="DATA 266 Lab 2 Report - Pair_Programming_Team_09",
                          author="Denisha Ketan Tank; Shibin Biji Thomas Lnu")
    from reportlab.platypus import FrameBreak
    story = title_flow + [FrameBreak(), NextPageTemplate("later")] + body
    doc.build(story)
    print("wrote", OUT)


if __name__ == "__main__":
    build()
