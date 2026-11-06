"""Render a measured Part 2 report, clearly marked draft while evidence is missing."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer,
                                Table, TableStyle, Image, NextPageTemplate, PageBreak, FrameBreak)
from reportlab.graphics.shapes import Drawing, Rect, String, Line, Polygon
from reportlab.graphics.charts.lineplots import LinePlot
import yaml
from submission import audit, read_json

ROOT = Path(__file__).resolve().parents[1]
STYLES = getSampleStyleSheet()
STYLES.add(ParagraphStyle(name="BodySmall", fontName="Times-Roman", fontSize=9, leading=11,
                         spaceAfter=6))
STYLES.add(ParagraphStyle(name="SectionSmall", fontName="Times-Bold", fontSize=10,
                         leading=12, spaceBefore=9, spaceAfter=5, keepWithNext=True))
STYLES.add(ParagraphStyle(name="CaptionSmall", fontName="Times-Italic", fontSize=8, leading=10,
                         spaceAfter=6))


def p(text, style="BodySmall"):
    # Plain ASCII hyphens avoid fonts with missing Unicode punctuation.
    text = str(text).replace("\u2014", " - ").replace("\u2013", "-").replace("\u2011", "-")
    return Paragraph(escape(text).replace("\n", "<br/>"), STYLES[style])


def table(rows, width):
    n = len(rows[0])
    grid = [[p(value, "CaptionSmall") for value in row] for row in rows]
    t = Table(grid, colWidths=[width / n] * n, repeatRows=1, hAlign="LEFT")
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eeeeee")),
        ("LINEBELOW", (0, 0), (-1, 0), .6, colors.black),
        ("LINEBELOW", (0, -1), (-1, -1), .4, colors.black),
        ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4)]))
    return t


def picture(path, width, caption):
    if not path.is_file():
        return [p("Not yet available: " + caption)]
    from PIL import Image as PILImage
    with PILImage.open(path) as im:
        height = width * im.height / im.width
    return [Image(str(path), width=width, height=height), p(caption, "CaptionSmall")]


def architecture(width):
    d = Drawing(width, 148)
    lines = [("512px RGB images / captions", 118), ("Frozen VAE / frozen CLIP text encoder", 80),
             ("Cached latents + text features", 42), ("Noisy latent -> UNet + trainable LoRA -> MSE", 4)]
    for label, y in lines:
        d.add(Rect(2, y, width - 4, 25, strokeColor=colors.black, fillColor=colors.white))
        d.add(String(width / 2, y + 9, label, textAnchor="middle", fontName="Times-Roman", fontSize=8))
        if y > 4:
            d.add(Line(width / 2, y, width / 2, y - 13))
            d.add(Polygon([width / 2 - 3, y - 9, width / 2, y - 13, width / 2 + 3, y - 9], fillColor=colors.black))
    return d


def loss_chart(output, width):
    histories = [read_json(output / run / "loss_history.json", []) for run in ["run1", "run2"]]
    if not all(histories):
        return [p("Training loss curves are not available. The uploaded Part 2 run stopped before its first optimizer update.")]
    d = Drawing(width, 150)
    chart = LinePlot()
    chart.x, chart.y, chart.width, chart.height = 25, 25, width - 40, 105
    # Average fixed blocks for a readable curve; retain original logs unchanged.
    chart.data = [[(h[min(i + 24, len(h) - 1)]["step"],
                    sum(v["loss"] for v in h[i:i + 25]) / len(h[i:i + 25]))
                   for i in range(0, len(h), 25)] for h in histories]
    chart.lines[0].strokeColor = colors.black
    chart.lines[1].strokeColor = colors.gray
    chart.lines[1].strokeDashArray = [3, 2]
    chart.xValueAxis.labelTextFormat = "%.0f"
    chart.yValueAxis.labelTextFormat = "%.2f"
    d.add(chart)
    d.add(String(width / 2, 5, "Optimizer updates; solid: rank 4, dashed: rank 8", textAnchor="middle", fontSize=7))
    return [d, p("Training noise MSE, averaged in blocks of 25 updates. This is not a validation quality score.", "CaptionSmall")]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "output/pdf/Part2_Findings.pdf")
    args = parser.parse_args()
    result = audit()
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    output = ROOT / cfg["output_root"]
    details = read_json(ROOT / "submission_details.json", {})
    width, margin, gap = 612, 45, 18
    full = width - 2 * margin
    col = (full - gap) / 2
    story = []
    title_style = ParagraphStyle("TitlePart2", parent=STYLES["Title"], fontName="Times-Bold", fontSize=17, leading=20)
    story += [Paragraph("Domain-Specific Image Generation with LoRA", title_style),
              p(details.get("student_name", "Student") + (" and " + details["partner_name"] if details.get("partner_name") else "; partner details pending")),
              p("Part 2 findings - " + ("measured evidence available" if result["ready"] else "DRAFT: full experiment or submission evidence is still missing")),
              p("Abstract", "SectionSmall"),
              p("This experiment compares Stable Diffusion 1.5 with two attention-only LoRA adapters, ranks 4 and 8. "
                "Both use the same cleaned dataset, split and update budget. The report reads saved measurement files; it does not invent missing scores. "
                + ("The uploaded Part 2 notebook failed during setup and contains no training results." if not result["training"]
                   else "See the measured tables and pending-evidence checklist before treating this draft as final.")),
              NextPageTemplate("columns"), FrameBreak()]
    story += [p("I. Data and experimental setup", "SectionSmall"),
              p("I use Naruto BLIP Captions, an anime-frame dataset with automatically generated captions. "
                "Exact duplicate pixels, corrupt files and empty captions are removed before the seed-266 90/10 split. "
                "Images are converted to RGB and resized to 512 x 512. Test images are reserved for evaluation."),
              p("Locally verified preparation: " + json.dumps(result["data"].get("removed", {})) + "; "
                + str(result["data"].get("train", "not measured")) + " train and "
                + str(result["data"].get("test", "not measured")) + " test images. These local data checks are separate from the failed uploaded training run."),
              p("Two important limits are narrow anime/character coverage and noisy captions. In the prepared data, some captions repeat words or describe clothing vaguely. "
                "These issues can weaken prompt following and limit generalization to photographs or other domains."),
              architecture(col), p("Fig. 1. Training architecture. Frozen components are run before the UNet to reduce peak memory.", "CaptionSmall"),
              p("II. Controlled training comparison", "SectionSmall"),
              p(f"Both ranks use {cfg['max_train_steps']} AdamW updates at learning rate {cfg['learning_rate']}, "
                f"microbatch {cfg['train_batch_size']} and accumulation {cfg['gradient_accumulation_steps']}. "
                "Only attention LoRA weights update. Base weights stay frozen, gradient checkpointing is enabled, and fp16 is used for the frozen UNet. "
                "The lab requires a working single-16-GB-GPU path. A successful 24-GB RTX 4090 setup alone does not demonstrate that memory target.")]
    if result["training"]:
        story += [table([["Run", "Rank", "Trainable", "LR", "Updates"], *[[m["model"], m["rank"], m["trainable_parameters"], m["learning_rate"], m["steps"]] for m in result["training"]]], col), Spacer(1, 6),
                  table([["Run", "Batch/acc.", "Peak MiB", "Seconds", "GPU-h"], *[[m["model"], f"{m['batch_size']}/{m['gradient_accumulation_steps']}", f"{m['peak_allocated_mib']:.1f}" if m.get('peak_allocated_mib') is not None else "CPU", f"{m['training_seconds']:.1f}", f"{m['gpu_hours']:.4f}" if m.get('gpu_hours') is not None else "N/A"] for m in result["training"]]], col),
                  p("Table I. Measured optimizer-loop cost, including checkpoint writes but excluding shared VAE/text caching and evaluation.", "CaptionSmall")]
    else:
        story += [p("No completed CUDA training measurements are present. Parameter counts, final loss, peak VRAM and GPU-hours must come from the new run; they cannot be inferred from the installation log.")]
    story += loss_chart(output, col)
    story += [p("III. Generation and evaluation", "SectionSmall"),
              p("Twenty prompts are frozen before training: ten held-out captions, five out-of-domain requests and five compositional requests. "
                "Base, rank 4 and rank 8 each generate seeds 0 and 1 using the same sampler, step count and guidance settings: 120 images in total. "
                "The safety checker remains enabled. Eight grids compare the models side by side."),
              p("Sampling protocol: " + json.dumps(result["evaluation"].get("sampling", {
                  "scheduler": "DDIMScheduler", "steps": 20, "guidance_scale": 7.5, "eta": 0.0,
                  "status": "planned; no generated-image results yet"}))),
              p("KID compares the twenty in-domain generations per model with the test images; lower is better, but the estimate is noisy at this size and may be negative. "
                "CLIP score measures image-prompt alignment. LPIPS measures diversity across the two seeds for each prompt. "
                "None of these replaces checking whether characters, counts and relationships are actually correct.")]
    metrics = result["evaluation"].get("metrics", [])
    if metrics:
        story += [table([["Model", "KID", "CLIP", "LPIPS"], *[[r["model"], f"{r['kid_mean']:.4f}", f"{r['clip_score_mean']:.2f}", f"{r['lpips_seed_pair_mean_256px']:.4f}"] for r in metrics]], col),
                  p("Table II. Actual saved automatic metrics. KID uses in-domain images only; CLIP and LPIPS cover all twenty prompts.", "CaptionSmall")]
    else:
        story += [p("Automatic quality metrics are pending. No image-quality conclusion can be made from the failed uploaded Part 2 notebook.")]
    story += [p("IV. Human evaluation and model choice", "SectionSmall"),
              p("The program creates 25 blinded comparisons with randomized left/right placement. Two team members and at least one person outside the team must independently judge domain style and prompt match. "
                "The reported win rate is final-run wins divided by all judgments, with ties reported separately. Human judgments are not generated by an LLM.")]
    if result["human"]["complete"]:
        story += [p(json.dumps(result["human"]["criteria"], indent=2))]
    else:
        story += [p("Human evaluation is pending, as requested. Do not submit an invented win rate.")]
    review = result["review"]
    story += [p(review.get("selection_reason") or "Final-run selection is pending inspection of the actual generated images and metric trade-offs.")]
    story += [p("V. Failure analysis", "SectionSmall")]
    failures = review.get("failure_modes", [])
    if len(failures) < 2:
        story += [p("At least two confirmed failure modes still need to be recorded after viewing the outputs. "
                    "Low alignment, ignored object counts, repeated poses and out-of-domain style drift are candidates to inspect, not findings from a run that never trained.")]
    for index, f in enumerate(failures[:4]):
        name = f"{f['model']}_{f['prompt_index']:02d}_{f['seed']}.png"
        story += picture(output / "generations" / name, col, f"Failure example {index + 1}: {name}")
        story += [p(f.get("observation", "") + " " + f.get("explanation", ""))]
    story += [p("VI. Safety, licensing and memorization", "SectionSmall"),
              p("The Naruto dataset card does not include an explicit license field. Instructor approval of a dataset option does not itself grant copyright or redistribution rights. "
                "Anime frames may contain copyrighted characters and designs, and generated images may reproduce similar features. Before uploading adapters or samples, "
                "confirm the intended educational sharing is permitted. The base SD 1.5 model has its own CreativeML Open RAIL-M conditions; they do not resolve source-image rights."),
              p("The diffusion pipeline keeps its NSFW checker and records its flag for every generation. A filtered output remains counted rather than being silently replaced. "
                "The checker can make mistakes, so manually inspect outputs before sharing. Avoid unsafe, deceptive or real-person-targeted prompts. "
                "Use private artifact repositories by default and share access with the graders as required. Private sharing reduces exposure but does not create a license."),
              p("The memorization check embeds ten seed-0 in-domain generations from each adapter with CLIP, compares them with all training images, and shows the three highest-similarity pairs. "
                "High cosine similarity is a warning for inspection, not proof that an image was copied. Low similarity is not proof that memorization is absent. "
                "The check is limited by ten samples, a single embedding model and a small training set. A stronger check would use more prompts, perceptual hashes and direct visual review."),
              p("Do not publish rater identities or authentication tokens. The human-rating form uses anonymous IDs, and the token is entered using a hidden prompt. "
                "Both team members still need to inspect the code, run the experiment and explain the modeling and safety decisions during the demo.")]
    final = review.get("final_run")
    if final in ["run1", "run2"]:
        for i in range(3):
            story += picture(output / f"memorization_{final}_{i}.png", col, f"Closest training pair {i + 1} for {final}; generated image left, training image right.")
    if metrics:
        story += [NextPageTemplate("wide"), PageBreak(), p("Appendix: controlled visual comparisons", "SectionSmall")]
        for i in range(8):
            story += picture(output / "generations" / f"comparison_{i:02d}.png", full, f"Prompt {i + 1}, seed 0: base, rank 4 and rank 8, left to right.")
    story += [p("Submission status", "SectionSmall"), p("This is a Part 2 section draft, not the required final combined report for all three parts. "
                "The final submission also needs team contributions, AI disclosure, discussion recordings, the tag commit links, and the Kaggle link. "
                "Pending items are listed below so incomplete work is not mistaken for a finished submission.")]
    for item in result["pending"]:
        story.append(p("Pending: " + item.replace("_", " ")))
    for name, url in result["hub_links"].items():
        if isinstance(url, str) and url.startswith("https://"):
            story.append(p(name + ": " + url))
    story += [p("References", "SectionSmall"), p("[1] DATA 266 Lab 2 requirements, pp. 8-9 and 11-12, provided by the course."),
              p("[2] Rombach et al. High-Resolution Image Synthesis with Latent Diffusion Models, 2022. arxiv.org/abs/2112.10752."),
              p("[3] Hu et al. LoRA: Low-Rank Adaptation of Large Language Models, 2021. arxiv.org/abs/2106.09685."),
              p("[4] Carlini et al. Extracting Training Data from Diffusion Models, 2023. arxiv.org/abs/2301.13188.")]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    doc = BaseDocTemplate(str(args.output), pagesize=(612, 792), leftMargin=margin, rightMargin=margin,
                          topMargin=40, bottomMargin=40, title="Lab 2 Part 2 Findings", author=details.get("student_name", "Student"))
    def footer(canvas, document):
        canvas.setFont("Times-Roman", 8)
        canvas.drawString(margin, 25, "Lab 2 | Part 2 | Evidence-based draft" if not result["ready"] else "Lab 2 | Part 2")
        canvas.drawRightString(612 - margin, 25, str(document.page))
    doc.addPageTemplates([
        PageTemplate(id="title", frames=[Frame(margin, 610, full, 142, leftPadding=0, rightPadding=0),
                   Frame(margin, 42, col, 550, leftPadding=0, rightPadding=0),
                   Frame(margin + col + gap, 42, col, 550, leftPadding=0, rightPadding=0)], onPage=footer),
        PageTemplate(id="wide", frames=[Frame(margin, 42, full, 710, leftPadding=0, rightPadding=0)], onPage=footer),
        PageTemplate(id="columns", frames=[Frame(margin, 42, col, 710, leftPadding=0, rightPadding=0),
                                            Frame(margin + col + gap, 42, col, 710, leftPadding=0, rightPadding=0)], onPage=footer)])
    doc.build(story)
    print("Created", args.output)
    print("Report status:", "Part 2 evidence complete; verify common submission requirements" if result["ready"] else "DRAFT with pending evidence")


if __name__ == "__main__":
    main()
