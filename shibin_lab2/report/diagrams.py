"""Architecture diagrams for the report (Part 1 and Part 3) -> report/fig_p1_arch.png, report/fig_p3_arch.png

    python report/diagrams.py
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def box(ax, x, y, w, h, text, color="#e8eef7", fs=6.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.04", fc=color, ec="#33415c", lw=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, wrap=True)
    return (x, y, w, h)


def arrow(ax, a, b, text="", side="h", fs=6.5, rad=0.0):
    ax_, ay, aw, ah = a
    bx, by, bw, bh = b
    if side == "h":
        p, q = (ax_ + aw, ay + ah / 2), (bx, by + bh / 2)
    elif side == "v":
        p, q = (ax_ + aw / 2, ay), (bx + bw / 2, by + bh)
    else:  # "up"
        p, q = (ax_ + aw / 2, ay + ah), (bx + bw / 2, by)
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=9, lw=0.9, color="#33415c",
                                 connectionstyle=f"arc3,rad={rad}"))
    if text:
        ax.text((p[0] + q[0]) / 2, (p[1] + q[1]) / 2 + 0.05, text, ha="center", va="bottom", fontsize=fs, color="#7a1f1f")


def part1():
    fig, ax = plt.subplots(figsize=(7.4, 3.9))
    ax.set_xlim(0, 10.4); ax.set_ylim(0, 5.2); ax.axis("off")
    pdf = box(ax, 0.05, 3.8, 1.7, 1.1, "3 PDFs +\nkb_manifest.json\n(trust levels)", "#fdf3d8")
    ing = box(ax, 2.05, 3.8, 2.45, 1.1, "PyMuPDF ingestion\ntext chunks (doc, page)\nFigure/Table crops + caption\nAppendix C -> excluded")
    vlm = box(ax, 4.8, 3.8, 2.1, 1.1, "Qwen2.5-VL\nfigure descriptions\n(values transcribed)\ninjection filter")
    db = box(ax, 7.2, 3.8, 3.15, 1.1, "ChromaDB (bge-small) + BM25\nmeta: doc_id, page, figure_id,\ntrust_level, kind, excluded")
    q = box(ax, 0.05, 1.9, 1.7, 1.0, "Question +\nconversation_id", "#fdf3d8")
    mem = box(ax, 0.05, 0.1, 2.2, 1.1, "Conversation memory\n(turns, user values)\nnever written to the DB", "#eaf6ea")
    ret = box(ax, 2.05, 1.9, 2.45, 1.0, "Hybrid retrieval\ndense + BM25 (RRF)\nexact figure lookup\n(+ reranker, bonus)")
    gen = box(ax, 4.8, 1.9, 2.1, 1.0, "Qwen2.5-VL generator\nnumbered evidence\n+ named figure image\nanswering rules")
    val = box(ax, 7.2, 1.9, 3.15, 1.0, "Validation\ncitations from evidence ids\nexact labels, status rules\n<=120 words")
    out = box(ax, 7.2, 0.1, 3.15, 1.0, '{"answer", "status",\n"citations", "figure_references"}', "#fdf3d8")
    arrow(ax, pdf, ing); arrow(ax, ing, vlm); arrow(ax, vlm, db)
    arrow(ax, q, ret); arrow(ax, ret, gen); arrow(ax, gen, val); arrow(ax, val, out, side="v")
    ax.add_patch(FancyArrowPatch((8.0, 3.8), (3.6, 2.9), arrowstyle="-|>", mutation_scale=9, lw=0.9, color="#33415c"))
    ax.text(6.0, 3.4, "top-k evidence", fontsize=6.5, color="#7a1f1f")
    ax.add_patch(FancyArrowPatch((1.6, 1.2), (2.6, 1.9), arrowstyle="-|>", mutation_scale=9, lw=0.9, color="#33415c"))
    ax.text(2.3, 1.35, "follow-up context", fontsize=6.5, color="#7a1f1f")
    ax.add_patch(FancyArrowPatch((2.25, 0.6), (5.6, 1.9), arrowstyle="-|>", mutation_scale=9, lw=0.9, color="#33415c",
                                 connectionstyle="arc3,rad=0.15"))
    ax.text(4.3, 0.75, "history + user values", fontsize=6.5, color="#7a1f1f")
    fig.savefig(os.path.join(HERE, "fig_p1_arch.png"), dpi=200, bbox_inches="tight")


def part3():
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.set_xlim(0, 10.15); ax.set_ylim(0.6, 3.75); ax.axis("off")
    u = box(ax, 0.05, 2.6, 1.3, 1.0, "User request\n+ image", "#fdf3d8")
    p = box(ax, 1.55, 2.6, 1.75, 1.0, "Planner\nimage check,\nrequest/image safety,\nintent -> route")
    v = box(ax, 3.5, 2.6, 1.65, 1.0, "Vision\nUnderstanding\ncaption, objects,\nvisible text, Q&A")
    pe = box(ax, 5.35, 2.6, 1.6, 1.0, "Prompt Eng.\nrewrite; drops\nimage text")
    g = box(ax, 7.15, 2.6, 2.85, 1.0, "Image Generation\nSD1.5 img2img (style)\nSD1.5 + Part 2 LoRA (domain)\npix2pix (edit) | PIL (enhance)")
    c = box(ax, 7.15, 0.9, 2.85, 1.1, "Critique\nCLIP score + VLM judge\nSD safety checker + VLM\naccept / revise (<=2)")
    f = box(ax, 3.45, 0.9, 2.6, 1.1, "Finalize\nbest safe attempt, message,\nJSON trace per request", "#eaf6ea")
    arrow(ax, u, p); arrow(ax, p, v); arrow(ax, v, pe); arrow(ax, pe, g); arrow(ax, g, c, side="v")
    arrow(ax, c, pe, "revise", side="up", rad=0.25)
    ax.add_patch(FancyArrowPatch((7.15, 1.45), (6.05, 1.45), arrowstyle="-|>", mutation_scale=9, lw=0.9, color="#33415c"))
    ax.text(6.6, 1.5, "accept/reject", fontsize=6.5, color="#7a1f1f", ha="center")
    ax.add_patch(FancyArrowPatch((2.4, 2.6), (3.6, 2.0), arrowstyle="-|>", mutation_scale=9, lw=0.9, color="#7a1f1f",
                                 connectionstyle="arc3,rad=0.2"))
    ax.text(2.2, 2.1, "refuse / clarify /\nbroken image", fontsize=6.5, color="#7a1f1f")
    ax.add_patch(FancyArrowPatch((4.25, 2.6), (4.6, 2.0), arrowstyle="-|>", mutation_scale=9, lw=0.9, color="#7a1f1f"))
    ax.text(4.55, 2.25, "caption /\nquestion", fontsize=6.5, color="#7a1f1f")
    ax.text(0.05, 0.75, "Shared typed state: AgentState (pydantic)", fontsize=7, style="italic")
    fig.savefig(os.path.join(HERE, "fig_p3_arch.png"), dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    part1(); part3(); print("ok")
