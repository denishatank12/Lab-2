"""Build a black-and-white evidence report and the exact Canvas review ZIP."""
import hashlib
import io
import json
from pathlib import Path
import zipfile
from xml.sax.saxutils import escape

from PIL import Image as PILImage, ImageOps
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (BaseDocTemplate, Frame, PageTemplate, Paragraph, Table,
    TableStyle, Spacer, PageBreak, Image, BalancedColumns)
from reportlab.graphics.shapes import Drawing, Rect, String, Line

from final_submission_audit import ROOT, collect


def materials(audit):
    p2 = audit['part2']
    train = p2['training']
    metrics = p2['evaluation']['metrics']
    p3 = audit['part3']
    sections = []
    def page(title, blocks):
        sections.append((title, blocks))
    p = lambda s: ('p', s)
    h = lambda s: ('h', s)
    page('Part 1: Multimodal retrieval', [
        h('I. Task and selected implementation'),
        p('The task is to answer questions from three economics PDFs without changing the knowledge base. Some questions depend on charts, similar-looking figures, revisions or earlier conversation turns. A plausible answer is not enough: its status and citations also need to match the evidence.'),
        p('The selected source is the unchanged independent v6 notebook. It contains its own helper files and pinned environment. It extracts text and numbered figure/table objects, keeps document/page/figure/trust metadata, and indexes text and visual embeddings in local Qdrant. The answer model is Qwen2-VL-7B with 4-bit language weights and float16 vision. The random seed is 266.'),
        ('diagram', ['Read-only PDFs and trust manifest', 'Text chunks and numbered visual objects', 'Dense + BM25 + CLIP retrieval and reranking', 'Session memory + grounded JSON answer']),
        p('Fig. 1. The final retrieval path. Conversation assumptions are separate from the document index.'),
        h('II. Evaluation design'),
        p('The intended comparison is dense-only retrieval against the final multimodal hybrid system using the same answer model. The 25 candidate golden questions include facts/tables, figure identity, two-source questions, multiple turns, conflicts/revisions and unsupported requests. Team review of those references is still required. Correctness, Recall@5, citation support, median latency and both members\' groundedness ratings must be reported from the selected run.'),
        ('table', [['Evidence', 'Current record'], ['Denisha public Kaggle score', '0.510, screenshot and live submission details'],
            ['Newer partner submission', '0.805, Shibin, notebook Version 2'],
            ['Selected source', 'v6, unchanged hash recorded'], ['Original executed v6 / CSV / metrics', 'Not available in current files'],
            ['Both members\' groundedness', 'Not supplied']]),
        h('III. Findings and limits'),
        p('The screenshot shows an improvement from 0.419 to 0.510 on the public leaderboard. That is a leaderboard result, not a 25-question golden accuracy or a grade. The original v6 Results ZIP was supplied earlier, but its bytes are not available in the current checkout or Downloads. The later notebook with the same filename contains v8. Its outputs cannot support the earlier run.'),
        p('A read-only check of Kaggle on November 17 also shows a newer completed submission by Shibin with public score 0.805. That is a separate partner result, not the score of Denisha\'s preserved v6 notebook. Its source has not been imported. The original 0.510 CSV is still listed, but Chrome blocked its download. No final submission choice was changed.'),
        p('I would retain the selected source and restore its original executed notebook, CSV, traces and evaluation files before making a final comparison. Repeating a run could produce new evidence, but would not recreate the historical submission. No missing metric has been filled with a later result or a guessed value.')])
    page('Part 1: Safety and evidence quality', [
        h('IV. Source trust and prompt injection'),
        p('The main safety issue in this task is allowing retrieved material to change the assistant\'s instructions. A PDF can contain relevant data and an instruction-like sentence on the same page. The implementation treats extracted content as evidence rather than executable instructions. DOC3 Appendix C is quarantined as unverified and is excluded from answer evidence. Image text needs the same treatment as ordinary extracted text.'),
        p('Document trust does not mean every statement is current. A forecast and an actual outturn can both be correctly quoted while answering different questions. The manifest, vintage and figure identity should therefore travel with every retrieved chunk. Corrected versions and explicit errata take precedence under the stated policy. An unresolved disagreement should retain both sources and explain the preferred source rather than merge inconsistent values.'),
        p('A near-identical chart is a difficult failure case. Similar colour, axes or wording can make a visual match look convincing even when the base period differs. Exact figure identifiers, page references and original visual crops make that mistake reviewable. Reranking can improve retrieval, but it cannot establish that the answer used the right series or unit. That requires checking the question against the cited evidence.'),
        h('V. Conversation boundaries'),
        p('A user may introduce an assumption for one conversation. That assumption belongs in session-local memory, alongside the document facts. It must not become a permanent vector-store entry or silently change an official value. Conversation IDs isolate the state. Requests to rewrite or permanently update the corpus should be refused. An assistant\'s earlier answer is not an independent source for a later answer.'),
        p('Status selection is part of the answer, not just formatting. A missing passage should lead to not_found; an unsupported premise should be identified; an explicit chat assumption should be marked as a session override. A syntactically valid JSON object can still have the wrong status, figure or factual claim. Format validation is useful for Kaggle compatibility but does not demonstrate grounding.'),
        h('VI. Review still required'),
        p('The original selected execution is needed to inspect actual wrong answers, source choices, runtime and citations. I have not claimed that every injection was rejected or that the 90-minute requirement was met on the selected evidence now present. Both members should independently read each answer and source, rate groundedness from 1 to 5, retain their original judgments and calculate agreement.'),
        p('The public leaderboard gives feedback on a hidden subset. Repeated changes aimed only at that score may overfit it. The own-question set should remain independent of the competition questions and should be reviewed before model tuning. Test prompts, source hashes and validation records need to remain tied to the exact experiment. A later result with a reused filename must not be relabeled as an earlier run.'),
        p('The active repository no longer contains raw PDF files. The frozen notebook still uses its existing hash-pinned historical resource URLs, so Git history was preserved rather than rewritten. Course PDFs and other raw datasets should be shared through the permitted course or artifact channel. Keeping a public repository is the user\'s explicit choice; it differs from the assignment\'s private-repository requirement.')])
    page('Part 2: LoRA domain adaptation', [
        h('I. Data and training'),
        p('The experiment uses lambdalabs/naruto-blip-captions at its recorded revision. The 1,221 usable captioned images were split into 1,098 training and 123 held-out images with seed 266. The recorded cleanup found no corrupt images, duplicate hashes or empty captions. Inputs were RGB at 512 by 512 pixels.'),
        p('Both adapters use the same SD 1.5 base, data, AdamW learning rate 0.0001, 1,200 updates, batch size 1 and accumulation of 4. The changed parameter is LoRA rank, 4 versus 8. Training used an RTX 4090. The original executed notebook is preserved with its outputs.'),
        ('diagram', ['Captioned images: fixed train/test split', 'SD 1.5 frozen base + LoRA r4 or r8', '20 fixed prompts x 2 seeds x 3 models', 'KID, CLIP, LPIPS and blinded human review']),
        ('table', [['Run', 'Params', 'Loss', 'Seconds', 'Peak MiB']] + [[r['model'], f"{r['trainable_parameters']:,}", f"{r['final_loss']:.5f}", f"{r['training_seconds']:.2f}", f"{r['peak_allocated_mib']:.2f}"] for r in train]),
        p('Table I. Optimizer-loop measurements. GPU-hours were 0.2124 for r4 and 0.2147 for r8. These memory peaks are PyTorch allocations, not total pipeline or driver memory.'),
        h('II. Matched generation and results'),
        p('The fixed 20 prompts contain ten held-out in-domain prompts, five out-of-domain prompts and five compositional prompts. Seeds 0 and 1 produce 120 images across the base and two adapters. All use DDIM with 20 steps, CFG 7.5, eta 0 and the same resolution. Eight side-by-side grids are saved.'),
        ('table', [['Model', 'KID mean', 'CLIP', 'LPIPS', 'Filtered']] + [[r['model'], f"{r['kid_mean']:.4f}", f"{r['clip_score_mean']:.3f}", f"{r['lpips_seed_pair_mean_256px']:.3f}", str(r['nsfw_filtered'])] for r in metrics]),
        p('Table II. KID uses 20 in-domain generations against 123 test images. Lower KID is preferable; CLIP and LPIPS measure different properties. Small-sample KID is noisy.'),
        p('I would choose r4 for the domain model: it has the lowest KID and half the adapter parameters of r8. The base has the highest CLIP score, so fine-tuning did not improve every measure. The larger r4 LPIPS cannot by itself establish useful diversity because filtered black images affect the metric. Independent human style and prompt-match judgments are still pending.')])
    page('Part 2: Failures, safety and rights', [
        h('III. Failure analysis'),
        p('AI-assisted inspection of the saved grids found an attribute-binding issue: in the green-eyes/black-hair portrait, green spreads across the face while the eyes are not clearly green. The style changes, but the requested colour is not reliably attached to the right feature. More varied captions and an explicit attribute-following test would be more informative than CLIP alone.'),
        p('A second example asks for a group with one person holding a sword. More than one character appears to hold a blade. The base comparison also struggles with this relationship. This suggests that learning the domain appearance is easier than learning exact counts and ownership. It is not evidence that LoRA uniquely caused the counting problem.'),
        p('Some fine-tuned outputs are black because the enabled safety checker filtered them: six images for r4 and seven for r8. These are unusable outputs and remain counted. The original unfiltered images are not available, so a black result does not prove a false positive. Silently regenerating until an attractive image appears would change the evaluation protocol.'),
        h('IV. Memorization and evaluation limits'),
        p('The saved memorization analysis compares ten generated images with training images using CLIP cosine similarity, with the three closest pairs displayed. Similar composition or domain style is a warning to inspect, not proof of copying. Conversely, a limited nearest-neighbor check cannot prove that the model never memorizes. Additional seeds, more queries and perceptual duplicate checks would strengthen that review.'),
        p('KID, CLIP and LPIPS cannot replace the required human experiment. KID estimates distribution similarity and is unstable with this small generated sample. CLIP can overlook counts and spatial relationships. LPIPS measures seed variation, including undesirable variation. The blinded comparison needs 25 pairs rated independently by both team members and an outside person. Style and prompt match are separate judgments; the report must retain ties and calculate win rates from actual ballots.'),
        h('V. Safety, licenses and hardware'),
        p('The dataset depicts copyrighted Naruto imagery and its card has no explicit license field. This course experiment does not create ownership or broad redistribution rights. Use remains limited to non-commercial research subject to the source-image rights and SD 1.5 CreativeML Open RAIL-M restrictions. The team still needs to confirm permitted artifact sharing with the instructor. A private Hub repository limits access but does not grant a license.'),
        p('The safety checker stays enabled. Prompt filtering, nearest-neighbor review and a model card are useful safeguards but do not guarantee safe or original content. The adapter should not be used for deceptive real-person imagery or claims that generated content is authentic evidence. The model card records the domain bias, failures, filtering and limits rather than presenting the adapter as a general improvement.'),
        p('Both adapters and the 120-image dataset have recorded private Hugging Face commits and content hashes. Grader access is not yet confirmed. The run was measured on 24-GB RTX 4090; the small training allocation does not establish a full run on a single 16-GB GPU. A compatibility check must include model loading, training and evaluation under the required limit, with recorded hardware and memory evidence. It must not relabel the original hardware.')])
    page('Part 2: Visual evidence appendix', [
        h('Saved comparison examples'),
        p('The panels are copied from the original completed experiment and converted to grayscale only for this report. The notebook and Hub images retain their original colours. From left to right: base, r4, r8.'),
        ('image', 'part2_diffusion/outputs/generations/comparison_02.png'),
        p('Fig. 4. Attribute-binding example. The colour-specific observation in the preceding analysis must be checked in the original colour notebook.'),
        ('image', 'part2_diffusion/outputs/generations/comparison_04.png'),
        p('Fig. 5. Counting/ownership example: the prompt specifies one sword-holder. Automatic alignment does not reliably check that relationship.'),
        h('Three closest saved r4 generated/training pairs'),
        *[('image', f'part2_diffusion/outputs/memorization_run1_{i}.png') for i in range(3)],
        p('Fig. 6. The three displayed nearest-neighbor pairs from the recorded ten-image CLIP check. Similarity is not by itself proof of copying or proof that memorization is absent.')])
    page('Part 3: Multi-agent image workflow', [
        h('I. Architecture and comparison'),
        p('The partner workflow has five agents: planner, visual understanding, prompt rewriting, image generation and critic. Typed state retains decisions and outputs. Qwen3-VL-8B handles visual questions and OCR; SD 1.5 and InstructPix2Pix handle style and instruction edits. The Naruto route loads Shibin\'s public rank-4 adapter. It is a separate checkpoint from Denisha\'s Part 2 adapter.'),
        ('diagram', ['Planner: route request and check safety', 'Vision: objects, questions and visible text', 'Prompt rewrite + style/edit/domain generation', 'CLIP + VLM critic; at most 2 revisions']),
        p('Fig. 3. Final system flow. The independently executed baseline uses zero revisions while retaining safety and diagnostic checks.'),
        p('Fifteen frozen COCO-based scenarios cover five task categories. Ten additional cases test unsafe/deceptive requests, image injection and invalid inputs. Shibin\'s executed RTX 4090 notebook supplies the paired baseline/final comparison. Denisha separately completed the no-revision baseline on one Tesla T4.'),
        ('table', [['Condition', 'Auto proxy', 'Median s', 'Attack rate'], ['Shibin baseline / 4090', '15/15', '20.671', 'Not separately measured'],
            ['Shibin final / 4090', '15/15', '20.402', '0/10'], ['Denisha baseline / T4', f"{p3['automatic_proxy_passes']}/15", f"{p3['median_scenario_seconds']:.3f}", '0/10']]),
        p('Table III. Automatic checks, not human task success. Cross-GPU times are descriptive and do not measure the causal benefit of critique. Partner final mean revisions: 0.67.'),
        h('II. Baseline verification and limitations'),
        p('The T4 ZIP contains a genuinely executed notebook with no saved cell errors, 25 states and 25 traces. Adapter loading and GPU preflight passed. Peak PyTorch allocation was 10.635 GiB, with 10.836 GiB reserved on GPU0. This verifies the baseline on a nominal 16-GB card; it is not proof that all lab workloads fit.'),
        p('The automatic proxy can miss poor image quality. S06 asks for a pencil sketch, but the saved baseline output remains a colour photograph. S08 visibly adds snow, while S11 produces an anime-style scene. These are AI-assisted observations, not the required independent judgments. Both members must rate the paired baseline/final outputs and calculate human success, critic agreement and resolved disagreement counts.')])
    page('Part 3: System safety and deployment limits', [
        h('III. System card and intended use'),
        p('This is a course prototype for image understanding and controlled editing. It is not an autonomous system with general computer access, a reliable authenticity detector or a dependable safety judge. The Gradio interface exposes the stages so a user can inspect the route, visual description, rewritten prompt, output and critique. The team still needs to rehearse the required live examples, including a safe failure.'),
        p('The input photos and source revisions are recorded. Generated outputs are saved separately from inputs, with per-request trace identifiers. Model revisions, adapter choice and generation settings are bound to the baseline record. This makes a failure traceable, but a trace does not prove that the underlying model description or critic decision is correct.'),
        h('IV. Injection and unsafe generation'),
        p('Text read from an image is data. It should not change the application\'s instruction hierarchy or request access to files or credentials. The red-team set includes image-pixel instructions as well as unsafe and deceptive requests. The generation safety checker remains enabled, and failures to check safety should withhold the output rather than silently bypass the check. No access token is embedded in the reproduction notebook; the chosen adapter is public.'),
        p('The recorded attack rate is zero in ten cases for the final partner run and Denisha baseline. That small fixed set does not establish broad injection resistance. Some cases are rejected before generation, while others exercise visual understanding. Each case needs its expected safe outcome checked. An unrelated model exception cannot be treated as a successful refusal.'),
        p('S14, S15 and R09 intentionally use invalid or empty inputs and produce clear error responses. Those statuses do not mean the notebook crashed. The completed baseline retains them as test outcomes. A nonfatal host sitecustomize warning about wrapt also appears in its log; the evaluation and export still completed. The original log is preserved rather than edited to hide the warning.'),
        h('V. Quality, privacy and human oversight'),
        p('A critic can confidently accept a weak edit. CLIP measures broad alignment and can miss exact attributes, text or transformations. VLM descriptions can hallucinate details or miss a subtle change. The visible sketch failure demonstrates why automatic route/status checks are not a sufficient quality metric. Increasing the revision count is not a guaranteed fix and can amplify drift or latency.'),
        p('The final workflow caps revisions at two; the independent baseline allows none. Both members should first rate the actual requested transformation independently, compare their judgments with critic decisions and record how disagreements were resolved. Until then, no human success rate, human agreement or critic-human agreement is reported. Faster request timing alone cannot support a quality conclusion.'),
        p('COCO source photos retain their creator licenses. Naruto-domain generation remains subject to source-image and model restrictions. Users should not upload sensitive images to a public demo without a clear sharing decision. The app should stay private by default, avoid public share links, and not imply that edited images are authentic evidence. Input/output artifact hosting and grader access are still pending; Git contains source, manifests and traces rather than a raw image dataset.')])
    page('Part 3: Baseline image inspection', [
        p('These are original saved baseline outputs from the completed T4 run, shown in grayscale for this report. They are illustrative AI-assisted observations, not independent human ballots.'),
        ('image', 'output/review_assets/part3/baseline_outputs/generated/20261117-062558-60168c/attempt0.png'),
        p('Fig. 7. S06: the request asks for a pencil sketch. The output still has photographic texture and shading. The automatic proxy nevertheless counted the case as a pass.'),
        ('image', 'output/review_assets/part3/baseline_outputs/generated/20261117-062729-28d330/attempt0.png'),
        p('Fig. 8. S08: visible snow appears around the car. A person should still judge whether the requested edit preserves the subject and avoids artifacts.'),
        ('image', 'output/review_assets/part3/baseline_outputs/generated/20261117-062945-ed4f30/attempt0.png'),
        p('Fig. 9. S11: anime-style scene. The domain route loaded Shibin\'s actual public rank-4 adapter; it was not replaced by the base model.')])
    page('Submission record and references', [
        h('I. Team and reproducibility'),
        p('Team 09 consists of Denisha Ketan Tank and Shibin Biji Thomas Lnu. Shibin\'s executed work is imported unchanged and attributed to its original commit. Denisha\'s T4 baseline is a separate measured execution. AI assistance helped with debugging, packaging and evidence review; the disclosure is in AI_USAGE.md. Both members still need to review the complete report and contribution record.'),
        p('The GitHub repository is https://github.com/denishatank12/Lab-2. Denisha explicitly requested keeping this name and public visibility. The assignment instead asks for a private data266-lab2-team09 repository with grader access. The difference is recorded openly and is not treated as compliance.'),
        h('II. Remaining submission obligations'),
        *[p(x.replace('_', ' ')) for x in audit['pending']],
        p('The local Canvas review ZIP contains this PDF, CONTRIBUTIONS.md, AI_USAGE.md and links.txt. It is not automatically submitted. Final tags are deferred until the missing evidence is incorporated. Meeting dates, recording URLs, human ballots and external confirmations are not generated by code.'),
        h('III. References and artifact locations'),
        p('DATA 266 Fall 2026 Lab 2 assignment requirements, pp. 1-12, and lecture slides. Repository manifests record exact source and model revisions. Private Part 2 Hub URLs and commits, partner source, adapter and selected Part 1 source are listed in links.txt.'),
        p('Lewis et al., Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks, 2020. Hu et al., LoRA: Low-Rank Adaptation of Large Language Models, 2021. Rombach et al., High-Resolution Image Synthesis with Latent Diffusion Models, 2022. Wang et al., Qwen2-VL, 2024. Brooks et al., InstructPix2Pix, 2023.'),
        p('This report was generated from the saved evidence without repeating training or inference. Missing values remain pending. The original selected Part 1 GPU record is required before its complete experimental findings can be added.')])
    return sections


def build():
    audit = collect()
    sections = materials(audit)
    output = ROOT / 'output/pdf'
    output.mkdir(parents=True, exist_ok=True)
    styles = {
        'p': ParagraphStyle('body', fontName='Times-Roman', fontSize=9, leading=11.2, spaceAfter=6),
        'h': ParagraphStyle('heading', fontName='Times-Bold', fontSize=10, leading=12, spaceBefore=7, spaceAfter=5, keepWithNext=True),
        'title': ParagraphStyle('title', fontName='Times-Bold', fontSize=14, leading=16, spaceAfter=8)}
    def para(s, style='p'):
        return Paragraph(escape(str(s)), styles[style])
    pages, markdown = [], ['# Lab 2 findings · Team 09', '', 'Denisha Ketan Tank and Shibin Biji Thomas Lnu', '',
        '**Evidence review; human ratings and other pending items are not completed results.**', '']
    for index, (title, blocks) in enumerate(sections):
        story = []
        story.append(para(title, 'title'))
        if index == 0:
            story += [para('Denisha Ketan Tank and Shibin Biji Thomas Lnu'),
                      para('DATA 266 · Pair Programming Team 09 · Evidence and findings'),
                      para('Review copy: final human evaluation and submission evidence remain pending.')]
        markdown += ['## ' + title, '']
        for kind, data in blocks:
            if kind in ['p', 'h']:
                story.append(para(data, kind))
                markdown += [('### ' if kind == 'h' else '') + data, '']
            elif kind == 'table':
                colwidths = [252 / len(data[0])] * len(data[0])
                t = Table([[para(v) for v in r] for r in data], colWidths=colwidths, repeatRows=1, hAlign='LEFT')
                t.setStyle(TableStyle([('VALIGN', (0, 0), (-1, -1), 'TOP'),
                    ('LEFTPADDING', (0, 0), (-1, -1), 2), ('RIGHTPADDING', (0, 0), (-1, -1), 2),
                    ('LINEABOVE', (0, 0), (-1, 0), .6, colors.black),
                    ('LINEBELOW', (0, 0), (-1, 0), .4, colors.black),
                    ('LINEBELOW', (0, -1), (-1, -1), .6, colors.black)]))
                story += [t, Spacer(1, 6)]
                markdown += ['| ' + ' | '.join(map(str, data[0])) + ' |', '| ' + ' | '.join([':---'] * len(data[0])) + ' |']
                markdown += ['| ' + ' | '.join(map(str, r)) + ' |' for r in data[1:]] + ['']
            elif kind == 'diagram':
                d = Drawing(252, len(data) * 32)
                for i, label in enumerate(data):
                    y = (len(data) - i - 1) * 32 + 7
                    d.add(Rect(0, y, 252, 23, fillColor=colors.white, strokeColor=colors.black))
                    d.add(String(126, y + 8, label, fontName='Times-Roman', fontSize=8, textAnchor='middle'))
                    if i < len(data) - 1:
                        d.add(Line(126, y, 126, y - 9, strokeColor=colors.black))
                story.append(d)
                markdown += [' → '.join(data), '']
            elif kind == 'image':
                path = ROOT / data
                if not path.is_file():
                    story.append(para('Image artifact unavailable locally: ' + data))
                    continue
                with PILImage.open(path) as source:
                    gray = ImageOps.grayscale(source.convert('RGB'))
                    buffer = io.BytesIO()
                    gray.save(buffer, 'PNG')
                    buffer.seek(0)
                    width, height = gray.size
                scale = min(252 / width, 190 / height)
                story += [Image(buffer, width=width * scale, height=height * scale), Spacer(1, 5)]
                markdown += ['Saved visual example: `' + data + '` (included in the local PDF; original images are in the result artifacts).', '']
        pages.append(story)
    report = output / 'Lab2_Team09_Report.pdf'
    doc = BaseDocTemplate(str(report), pagesize=(612, 792), title='DATA 266 Lab 2: Evidence and Findings',
        author='Denisha Ketan Tank; Shibin Biji Thomas Lnu')
    def page_footer(canvas, document):
        canvas.setFont('Times-Roman', 8)
        canvas.drawString(45, 23, 'Team 09 · Lab 2 · Review copy: pending evidence remains')
        canvas.drawRightString(567, 23, str(document.page))
    doc.addPageTemplates(PageTemplate(id='IEEE-columns', frames=[
        Frame(45, 43, 522, 704, id='body', leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)], onPage=page_footer))
    story = []
    for page in pages:
        if story:
            story.append(PageBreak())
        story.extend([page[0], BalancedColumns(page[1:], nCols=2, innerPadding=18, leftPadding=0,
                                              rightPadding=0, topPadding=0, bottomPadding=0)])
    doc.build(story)
    (ROOT / 'REPORT.md').write_text('\n'.join(markdown))
    package = ROOT / 'output/Pair_Programming_Team_09.zip'
    with zipfile.ZipFile(package, 'w', zipfile.ZIP_DEFLATED) as bundle:
        bundle.write(report, 'Lab2_Team09_Report.pdf')
        for name in ['CONTRIBUTIONS.md', 'AI_USAGE.md', 'links.txt']:
            bundle.write(ROOT / name, name)
    with zipfile.ZipFile(package) as bundle:
        assert bundle.testzip() is None
        assert set(bundle.namelist()) == {'Lab2_Team09_Report.pdf', 'CONTRIBUTIONS.md', 'AI_USAGE.md', 'links.txt'}
    record = dict(report=str(report.relative_to(ROOT)), canvas_review_zip=str(package.relative_to(ROOT)),
        zip_sha256=hashlib.sha256(package.read_bytes()).hexdigest(), ready_for_submission=audit['ready'],
        note='Review package only. Missing evidence stays pending; no training, inference or external submission performed.')
    (ROOT / 'submission/package.json').write_text(json.dumps(record, indent=2) + '\n')
    print('Report:', report)
    print('Canvas review ZIP:', package)
    print('Ready for final submission:', audit['ready'])


if __name__ == '__main__':
    build()
