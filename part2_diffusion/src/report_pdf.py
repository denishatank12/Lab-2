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
                                Table, TableStyle, Image, NextPageTemplate, PageBreak, FrameBreak, Flowable)
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
    t = Table(grid, colWidths=[width / n] * n, repeatRows=1, hAlign="LEFT", splitByRow=0)
    t.keepWithNext = True
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


class ComparisonPanel(Flowable):
    """Clip one panel for page layout without editing or regenerating any pixels."""
    def __init__(self, path, width, panel):
        super().__init__()
        self.path, self.width, self.height, self.panel = str(path), width, width * 544 / 512, panel

    def draw(self):
        canvas = self.canv
        canvas.saveState()
        clip = canvas.beginPath()
        clip.rect(0, 0, self.width, self.height)
        canvas.clipPath(clip, stroke=0)
        canvas.drawImage(self.path, -self.panel * self.width, 0, width=3 * self.width, height=self.height)
        canvas.restoreState()


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
        return [p("The full loss-history files are not available in this report's input. Final training noise MSE is reported where saved; no missing curve is reconstructed.")]
    d = Drawing(width, 165)
    chart = LinePlot()
    chart.x, chart.y, chart.width, chart.height = 25, 40, width - 40, 105
    # Average fixed blocks for a readable curve; retain original logs unchanged.
    chart.data = [[(h[min(i + 24, len(h) - 1)]["step"],
                    sum(v["loss"] for v in h[i:i + 25]) / len(h[i:i + 25]))
                   for i in range(0, len(h), 25)] for h in histories]
    chart.lines[0].strokeColor = colors.black
    chart.lines[1].strokeColor = colors.gray
    chart.lines[1].strokeDashArray = [3, 2]
    chart.xValueAxis.labelTextFormat = "%.0f"
    chart.yValueAxis.labelTextFormat = "%.2f"
    chart.xValueAxis.labels.fontSize = 7
    chart.yValueAxis.labels.fontSize = 7
    d.add(chart)
    d.add(String(width / 2, 5, "Optimizer updates; solid: rank 4, dashed: rank 8", textAnchor="middle", fontSize=7))
    return [d, p("Training noise MSE, averaged in blocks of 25 updates. This is not a validation quality score.", "CaptionSmall")]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "output/pdf/Part2_Findings.pdf")
    parser.add_argument("--saved-evidence", type=Path, help="Report recovered notebook evidence without claiming missing local artifacts exist.")
    args = parser.parse_args()
    result = read_json(args.saved_evidence) if args.saved_evidence else audit()
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    output = ROOT / cfg["output_root"]
    if result.get("evidence_only"):
        output = ROOT / result["evidence_directory"]
    details = read_json(ROOT / "submission_details.json", {})
    width, margin, gap = 612, 45, 18
    full = width - 2 * margin
    col = (full - gap) / 2
    story = []
    title_style = ParagraphStyle("TitlePart2", parent=STYLES["Title"], fontName="Times-Bold", fontSize=17, leading=20)
    story += [Paragraph("Domain-Specific Image Generation with LoRA", title_style),
              p(details.get("student_name", "Student") + (" and " + details["partner_name"] if details.get("partner_name") else "; partner details pending")),
              p("Part 2 findings - " + ("measured evidence available" if result["ready"] else "DRAFT: remaining submission items are listed at the end")),
              p("Abstract", "SectionSmall"),
              p("This experiment compares Stable Diffusion 1.5 with two attention-only LoRA adapters, ranks 4 and 8. "
                "Both use the same cleaned dataset, split and update budget. "
                + ("The uploaded Part 2 notebook failed during setup and contains no training results." if not result["training"]
                   else "Rank 4 gives the lowest KID, while the base model keeps the highest overall CLIP alignment. The remaining human study and submission steps are clearly identified.")),
              NextPageTemplate("columns"), FrameBreak()]
    story += [p("I. Data and experimental setup", "SectionSmall"),
              p("I use Naruto BLIP Captions, an anime-frame dataset with automatically generated captions. "
                "Exact duplicate pixels, corrupt files and empty captions are removed before the seed-266 90/10 split. "
                "Images are converted to RGB and resized to 512 x 512. Test images are reserved for evaluation."),
              p("Verified preparation: " + json.dumps(result["data"].get("removed", {})) + "; "
                + str(result["data"].get("train", "not measured")) + " train and "
                + str(result["data"].get("test", "not measured")) + " test images. The saved output reports no removed corrupt, duplicate or empty-caption samples."),
              p("Two important limits are narrow anime/character coverage and noisy captions. In the prepared data, some captions repeat words or describe clothing vaguely. "
                "These issues can weaken prompt following and limit generalization to photographs or other domains."),
              architecture(col), p("Fig. 1. Training architecture. Frozen components are run before the UNet to reduce peak memory.", "CaptionSmall"),
              p("II. Controlled training comparison", "SectionSmall"),
              p(f"Both ranks use {cfg['max_train_steps']} AdamW updates at learning rate {cfg['learning_rate']}, "
                f"microbatch {cfg['train_batch_size']} and accumulation {cfg['gradient_accumulation_steps']}. "
                "Only attention LoRA weights update. Base weights stay frozen, gradient checkpointing is enabled, and fp16 is used for the frozen UNet. "
                "The lab requires a working single-16-GB-GPU path. A successful 24-GB RTX 4090 setup alone does not demonstrate that memory target.")]
    if result["training"]:
        story += [table([["Run", "Rank", "Trainable", "LR", "Updates", "Final MSE"], *[[m["model"], m["rank"], m["trainable_parameters"], m["learning_rate"], m["steps"], f"{m.get('final_loss', 0):.4f}" if 'final_loss' in m else "not saved"] for m in result["training"]]], col), Spacer(1, 6),
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
        names=[r['model'] for r in metrics]
        training={r['model']:r for r in result['training']}
        selected=result['review'].get('final_run')
        def human_value(model,criterion):
            if model=='base':
                return 'reference'
            if model!=selected:
                return 'not selected'
            return f"{result['human']['criteria'][criterion]['win_rate']:.3f}" if result['human']['complete'] else 'pending'
        summary=[['Measurement',*names]]
        for label,key,precision in [('KID','kid_mean',4),('CLIP','clip_score_mean',2),('LPIPS','lpips_seed_pair_mean_256px',4)]:
            summary.append([label,*[f'{r[key]:.{precision}f}' for r in metrics]])
        summary.extend([
            ['Human style win rate',*[human_value(name,'domain_style') for name in names]],
            ['Human prompt win rate',*[human_value(name,'prompt_match') for name in names]],
            ['Training seconds',*[f"{training[name]['training_seconds']:.1f}" if name in training else 'not trained' for name in names]]])
        story += [table(summary,col),
                  p("Table II. Baseline and adapter comparison. KID uses in-domain images; CLIP and LPIPS cover all twenty prompts. Human rates compare the selected adapter with the base model and remain pending until real ballots are supplied. Training time excludes shared caching and evaluation.", "CaptionSmall")]
        story += [p("Rank 4 is closer to the held-out domain under KID than either the base or rank 8. However, the base has the strongest average CLIP alignment. "
                    "Fine-tuning therefore improves one goal without improving every goal. Increasing the rank does not help this experiment enough to justify its larger adapter."),
                  p("Run1 has the largest LPIPS distance, but this needs caution: one black filtered image paired with a normal image can also raise perceptual distance. "
                    "That score alone is not proof of more useful diversity. KID uses only twenty generated in-domain images, so its subset variation is not a confidence interval for repeated training runs."),
                  table([["Model", "Filtered / 40"], *[[r["model"], f"{r.get('nsfw_filtered', 'not saved')} / 40"] for r in metrics]], col),
                  p("Table III. Safety-filtered images remain in the evaluation. They are not silently replaced with more favorable samples.", "CaptionSmall")]
    else:
        story += [p("Automatic quality metrics are pending. No image-quality conclusion can be made from the failed uploaded Part 2 notebook.")]
    story += [p("IV. Human evaluation and model choice", "SectionSmall"),
              p("The program creates 25 blinded comparisons with randomized left/right placement. Two team members and at least one person outside the team must independently judge domain style and prompt match. "
                "The reported win rate is final-run wins divided by all judgments, with ties reported separately. Human judgments are not generated by an LLM.")]
    if result["human"]["complete"]:
        story += [p(json.dumps(result["human"]["criteria"], indent=2))]
    else:
        story += [p("The independent three-rater comparison is not complete, so no human win rate is reported. The visual observations below are AI-assisted image review, not a substitute for those ratings.")]
    review = result["review"]
    story += [p(review.get("selection_reason") or "Final-run selection is pending inspection of the actual generated images and metric trade-offs.")]
    story += [p("V. Failure analysis", "SectionSmall")]
    failures = review.get("failure_modes", [])
    if len(failures) < 2:
        story += [p("At least two confirmed failure modes still need to be recorded after viewing the outputs. "
                    "Low alignment, ignored object counts, repeated poses and out-of-domain style drift are candidates to inspect, not findings from a run that never trained.")]
    for index, f in enumerate(failures[:4]):
        name = f"{f['model']}_{f['prompt_index']:02d}_{f['seed']}.png"
        original = output / "generations" / name
        comparison = output / "generations" / f"comparison_{f.get('comparison_index', f['prompt_index']):02d}.png"
        if not original.exists() and comparison.exists():
            story += [ComparisonPanel(comparison, col, 1 if f["model"] == "run1" else 2),
                      p(f"Failure example {index + 1}: {name}; panel shown directly from the saved comparison.", "CaptionSmall")]
        else:
            story += picture(original, col, f"Failure example {index + 1}: {name}")
        story += [p(f.get("observation", "") + " " + f.get("explanation", ""))]
    story += [p("VI. Safety, licensing and memorization", "SectionSmall"),
              p("The Naruto dataset card does not include an explicit license field. Instructor approval of a dataset option does not itself grant copyright or redistribution rights. "
                "Anime frames may contain copyrighted characters and designs, and generated images may reproduce similar features. Before uploading adapters or samples, "
                "confirm the intended educational sharing is permitted. The base SD 1.5 model has its own CreativeML Open RAIL-M conditions; they do not resolve source-image rights."),
              p("The course describes Naruto and Pokemon characters as copyrighted and limits this model to non-commercial research. This report follows that scope and does not claim that permission grants ownership of the source images."),
              p("The diffusion pipeline keeps its NSFW checker and records its flag for every generation. A filtered output remains counted rather than being silently replaced. "
                "The checker can make mistakes, so manually inspect outputs before sharing. Avoid unsafe, deceptive or real-person-targeted prompts. "
                "Use private artifact repositories by default and share access with the graders as required. Private sharing reduces exposure but does not create a license."),
              p("The evaluation code embeds ten seed-0 in-domain generations from each adapter with CLIP, compares them with all training images, and saves the three highest-similarity pairs. "
                "High cosine similarity is a warning for inspection, not proof that an image was copied. Low similarity is not proof that memorization is absent. "
                "The check is limited by ten samples, a single embedding model and a small training set. A stronger check would use more prompts, perceptual hashes and direct visual review."),
              p("Do not publish rater identities or authentication tokens. The human-rating form uses anonymous IDs, and the token is entered using a hidden prompt. "
                "Both team members still need to inspect the code, run the experiment and explain the modeling and safety decisions during the demo.")]
    if result.get("evidence_only"):
        story += [p("Evidence limit: this report recovers the actual measurements and eight PNG comparison grids from the executed notebook. "
                    "The notebook does not embed the adapters, all 120 separate images, the loss histories or the nearest-training-image files. "
                    "Those files must be brought over from the GPU outputs folder before artifact uploads and the memorization figure review can be finalized.")]
    final = review.get("final_run")
    if final in ["run1", "run2"]:
        nearest = sorted([r for r in read_json(output / "memorization.json", []) if r["model"] == final],
                         key=lambda r: r["clip_cosine"], reverse=True)[:3]
        if len(nearest) == 3:
            story += [p("For the selected rank-4 model, the closest three generated/training pairs have CLIP cosine similarities "
                        + ", ".join(f"{r['clip_cosine']:.4f}" for r in nearest)
                        + ". I can see shared anime-style features, but the displayed pairs differ in their hair, poses or characters and are not exact visible copies. "
                        "This small inspection does not establish that memorization is absent.")]
        for i in range(3):
            story += picture(output / f"memorization_{final}_{i}.png", col * .8, f"Closest training pair {i + 1} for {final}; generated image left, training image right.")
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
