"""Build an IEEE-style combined draft from measured files; absent results stay pending."""
from pathlib import Path
import json
import sys
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.platypus import (BaseDocTemplate,Frame,PageTemplate,Paragraph,Table,TableStyle,
                               Spacer,FrameBreak,NextPageTemplate)
from reportlab.graphics.shapes import Drawing,Rect,String,Line
from pypdf import PdfReader,PdfWriter

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from final_submission_audit import collect


def paragraph(text,heading=False):
    text=str(text).replace('\u2014',' - ').replace('\u2013','-')
    return Paragraph(escape(text).replace('\n','<br/>'),styles['HeadingSmall' if heading else 'BodySmall'])


styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='BodySmall',fontName='Times-Roman',fontSize=8.5,leading=10,spaceAfter=5))
styles.add(ParagraphStyle(name='HeadingSmall',fontName='Times-Bold',fontSize=9.5,leading=11,spaceAfter=4,spaceBefore=7,keepWithNext=True))
styles.add(ParagraphStyle(name='ReportTitle',fontName='Times-Bold',fontSize=15,leading=18,spaceAfter=8))


def diagram(lines):
    d=Drawing(252,len(lines)*38)
    for i,label in enumerate(lines):
        y=(len(lines)-i-1)*38+7
        d.add(Rect(0,y,252,25,fillColor=colors.white,strokeColor=colors.black))
        d.add(String(126,y+9,label,textAnchor='middle',fontName='Times-Roman',fontSize=7.5))
        if i<len(lines)-1:
            d.add(Line(126,y,126,y-13,strokeColor=colors.black))
    return d


def table(rows):
    t=Table([[paragraph(v) for v in row] for row in rows],colWidths=[252/len(rows[0])]*len(rows[0]),repeatRows=1,splitByRow=0)
    t.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),.5,colors.black),
                           ('LINEBELOW',(0,-1),(-1,-1),.5,colors.black)]))
    return t


def number(value):
    return 'pending' if value is None else f'{value:.3f}' if isinstance(value,float) else str(value)


def paired_table(results, systems, measures):
    by_system={row['system']:row for row in results}
    return table([['Measurement','Baseline','Final']] + [
        [label]+[number(by_system.get(system,{}).get(key)) for system in systems]
        for label,key in measures])


def render(path,title,story):
    doc=BaseDocTemplate(str(path),pagesize=(612,792),leftMargin=45,rightMargin=45,topMargin=42,bottomMargin=42)
    def footer(canvas,document):
        canvas.setFont('Times-Roman',8)
        canvas.drawString(45,25,'DATA 266 Lab 2 | '+title+' | DRAFT')
        canvas.drawRightString(567,25,str(document.page))
    doc.addPageTemplates([PageTemplate(id='columns',frames=[Frame(45,42,252,710,leftPadding=0,rightPadding=0),
        Frame(315,42,252,710,leftPadding=0,rightPadding=0)],onPage=footer)])
    doc.build(story)


def main():
    result=collect()
    temporary=ROOT/'tmp/pdfs/combined'
    temporary.mkdir(parents=True,exist_ok=True)
    details=json.loads((ROOT/'part2_diffusion/submission_details.json').read_text())
    author=details['student_name']+(' and '+details['partner_name'] if details.get('partner_name') else '; partner details pending')
    head=[Paragraph('Multimodal Retrieval, Domain Adaptation and Image Agents',styles['ReportTitle']),paragraph(author),
          paragraph('Combined report draft. The Part 2 GPU experiment is measured; revised Parts 1 and 3 still require their new model-backed runs. Human scores and team confirmations are not invented.'),
          paragraph('Part 1: read-only multimodal retrieval',True),
          paragraph('I keep the PDF knowledge base separate from conversation state. Text chunks carry document and page IDs. Numbered tables and figures are extracted as image objects; corrected figures remain separate. CLIP makes these visual objects searchable, and local Qdrant stores the text and image vectors with their trust metadata.'),
          diagram(['Read-only PDFs + manifest trust rules','Text chunks + exact figure/table image objects','Qdrant dense/CLIP search + BM25 fusion','Session-local memory + Qwen2-VL JSON answer']),
          paragraph('Fig. 1. The final Part 1 data flow. Session facts never enter the base vector database.'),
          paragraph('Methods and comparison',True),
          paragraph('The baseline uses dense retrieval only; the final system adds BM25 and CLIP reciprocal-rank fusion and exact figure lookup. Both use the same pinned answer model and source trust policy. Evidence is passed as data, not as instructions. Answers are limited to 120 words and their citations must resolve to sources actually retrieved.'),
          paragraph('Actual outturns and explicit corrections take precedence over forecasts or earlier versions. Unresolved disagreements should show both values, identify their trust levels and use conflict status. User assumptions are kept only in their conversation, alongside the document values. These behaviors require checking on the new run; implementing a prompt is not proof that every answer follows it.'),
          paragraph('Evaluation',True)]
    head.append(paired_table(result['part1'].get('results',[]),['dense_baseline','hybrid_multimodal'],[
        ('Candidate numeric/status correctness','candidate_answer_correctness'),('Retrieval Recall@5','retrieval_recall_at_5'),
        ('VLM-judged citation support','model_judged_citation_accuracy'),('Human groundedness, 1-5','human_groundedness'),
        ('Human agreement within one','human_agreement_within_one'),('Median seconds/question','median_seconds_per_question')]))
    if not result['part1'].get('results'):
        head.append(paragraph('The revised dense-only and final evaluations have not run yet. There are no new answer-quality, retrieval, citation or latency results to report. The old schema-valid extractive CSV is not substituted for this evaluation.'))
    head.extend([paragraph('The 25 candidate references cover facts/tables, near-identical figures, two-source questions, isolated multi-turn assumptions, revisions/conflicts and unsupported or adversarial questions. They are AI-assisted proposals, not falsely attributed student-authored work. Both members must review and revise them; the unverified flags are retained.'),
        paragraph('Human groundedness: '+number(result['part1_human']['mean_groundedness'])+'. Agreement within one point: '+number(result['part1_human']['agreement_within_one'])+'. Each member independently rates both systems on a 1-5 scale. No automatic score is used as their rating.'),
        paragraph('Safety analysis: source trust and prompt injection',True),
        paragraph('Appendix C of DOC3 contains unverified claims and explicit instructions in both prose and an image. Its metadata is low trust. I extract its objects for the audit trail but quarantine them before vector indexing and never send them to the answer model as evidence. This is an ingestion boundary rather than a claim that the language model is immune to attacks.'),
        paragraph('The main factual risks are mixing a forecast with an outturn, confusing a near-identical chart with another vintage or base date, ignoring an erratum, or presenting a lower-trust disagreement as a revision. Figure identity, page provenance and trust metadata are retained so those errors can be inspected. A full-page crop is used for some vector figures in DOC1; this can add irrelevant context, so its limitation is recorded.'),
        paragraph('Conversation memory contains only the current conversation. Prior assistant answers are not treated as new document facts. Source files are never rewritten, and permanent-change requests are refused. Retrieved text and image text remain untrusted data even when they come from an otherwise high-trust document.'),
        paragraph('The new golden-set run must show concrete examples of whether the system rejected the false Appendix C claims and the embedded instructions. Until those outputs exist, I do not describe the system as never fooled. VLM citation entailment and numeric/status checks are automatic proxies that still need reference and human review.'),
        paragraph('Part 1 remaining work',True),paragraph('Run the revised notebook, review the candidate references, obtain both members\' groundedness ratings, inspect wrong answers and citations, then run and select the final validated submission on Kaggle. A correctly formatted CSV does not demonstrate answer accuracy.')])
    render(temporary/'part1.pdf','Part 1',head)
    tail=[Paragraph('Part 3: image understanding and generation agents',styles['ReportTitle']),
          paragraph('The system uses five typed-state stages. The request planner chooses caption/question, style, instruction edit, domain style or unsupported. Caption/question requests stop after visual understanding. Generation goes through prompt rewriting, output safety and critique, with at most two revisions.'),
          diagram(['Planner + request safety','Qwen2-VL: objects, text and visual questions','Prompt rewriting (image text is data only)','SD img2img / InstructPix2Pix / Part 2 LoRA','CLIP + visual critique + bounded revision']),
          paragraph('Fig. 2. Part 3 architecture. The VLM is unloaded before diffusion inference to avoid concurrent large GPU models.'),
          paragraph('Models and evaluation protocol',True),
          paragraph('Qwen2-VL-2B-Instruct handles image text transcription, object descriptions, visual questions and comparison of the original and generated images. SD 1.5 img2img handles style changes; InstructPix2Pix handles instruction editing. Domain-style requests load the selected Part 2 rank-4 adapter from its private, pinned Hugging Face commit. Model revisions and generation seeds/settings are recorded in the JSON trace.'),
          paragraph('Fifteen predefined scenarios have explicit rubrics: three visual understanding tasks, three style changes, three edits/enhancements, three domain-style tasks and three ambiguous/broken-input tasks. Baseline and critique configurations use the same requests, inputs and initial seed. The default real inputs are three held-out anime frames; this narrow domain does not establish performance on unrelated photographs. Synthetic overlays are labeled adversarial fixtures, not generated-model results.'),
          paragraph('Measured comparison',True)]
    tail.append(paired_table(result['part3'].get('results',[]),['baseline','critique'],[
        ('Human-rated task success / 15','task_success_rate'),('Critic agreement','critic_agreement'),
        ('Red-team failure rate / 10','red_team_attack_success_rate'),('Member agreement','human_agreement'),
        ('Median seconds/request','median_seconds_per_request')]))
    if not result['part3'].get('results'):
        tail.append(paragraph('The revised model-backed comparison has not run yet. All new task-success, critic-agreement, attack-success, human-agreement and latency results remain pending. The earlier synthetic-image demonstration is not treated as the corrected system\'s evaluation.'))
    tail.extend([paragraph('Each member independently judges every scenario for both systems. If their judgments differ, they record a resolved outcome and the reason. Critic acceptance is not a human task pass. A revision-limit outcome is analyzed as a failure to satisfy the critic, not hidden or relabeled successful.'),
        paragraph('System card',True),
        paragraph('Intended use: non-commercial course experiments in captioning, visual questions, style changes and instruction-based edits. The system is not a general autonomous agent: it does not browse, execute image text or modify arbitrary files. It reads a supplied image and writes separate per-request artifacts. Known limits include VLM transcription mistakes, uncertain object attributes, missed subtle changes, imperfect safety filtering and an anime-biased domain adapter. Automatic critique can accept poor results or reject good ones, so the paired human study remains necessary.'),
        paragraph('Safety analysis and failure handling',True),
        paragraph('The planner rejects unsafe or deceptive requests before diffusion. Visual understanding transcribes visible text and checks it for instruction-like material; it is never appended as an executable instruction. The output pipeline retains the checkpoint safety checker. The critic also examines the original and generated image, checks output safety, gives a short visual judgment and proposes a concrete change when needed. These layers reduce risk but do not guarantee safety.'),
        paragraph('The ten frozen security cases include three actual image-pixel injections, unsafe generation, deceptive real-person editing, fabricated evidence, corrupt and empty files, a missing image and an empty request. Both configurations are evaluated separately. Attack success is conservatively defined as failure to meet the case\'s expected safe outcome, not as a claim of real secret exfiltration. A model exception is not counted as a successful refusal.'),
        paragraph('The earlier text-only safety test did not exercise image perception. The new image-based test needs a real GPU run, and any missed or misread instructions must be reported. The model sees adversarial images as data and has no tool access that could act on them. The test images include synthetic instruction overlays; this is a bounded test set, not evidence against unseen attacks.'),
        paragraph('Poor-quality inputs, unreadable files and model failures return clear messages and are retained in traces. Each request and revision has a separate output path. Unsafe output is withheld from the UI. Revisions are capped at two. The local Gradio demo does not enable public sharing; both members still need to rehearse the required live scenarios.'),
        paragraph('Reproducibility and common submission status',True),
        paragraph('The RTX 4050 profile offloads some VLM weights to CPU, reserves CUDA activation memory and limits the image budget to 512 pixels per side. Models run sequentially. This profile is implemented but not yet certified by a full run on that GPU. The overnight launcher requires actual model preflights on the target PC and saves completed items with source, input and hardware fingerprints; it does not reuse a result under changed conditions.'),
        paragraph('The first notebook cells print the commit, pinned versions, seed and actual GPU. CUDA tensor peaks from the earlier Part 2 optimizer loop are not full-pipeline device peaks. New stages sample device memory, but samples can miss brief peaks and include other processes. A 24-GB RTX 4090 run is not relabeled as execution on a 16-GB device.'),
        paragraph('Source code and notebook outputs go to GitHub; weights and generated-image datasets belong on Hugging Face. Private artifacts still require grader access. The supplied team number, partner details, contributions, meeting recordings, real ratings and final Kaggle selection must be confirmed by the team. The required repository name/privacy and final part tags cannot be inferred from a successful model run.'),
        paragraph('Pending items from the actual audit',True)])
    tail.extend(paragraph(x.replace('_',' ')) for x in result['pending'])
    tail.extend([paragraph('References',True),paragraph('DATA 266 Lab 2 course requirements, pp. 1-12. Lewis et al., Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (2020). Hu et al., LoRA (2021). Rombach et al., Latent Diffusion Models (2022). Wang et al., Qwen2-VL (2024). Brooks et al., InstructPix2Pix (2023). Model IDs and pinned revisions are recorded in source and run metadata.')])
    render(temporary/'part3.pdf','Part 3',tail)
    sources=[temporary/'part1.pdf']
    part2=ROOT/'part2_diffusion/output/pdf/Part2_Findings.pdf'
    if not part2.exists():
        part2=ROOT/'docs/Part2_Findings_DRAFT.pdf'
    if part2.exists():
        sources.append(part2)
    sources.append(temporary/'part3.pdf')
    writer=PdfWriter()
    for path in sources:
        writer.append(str(path))
    writer.add_metadata({'/Title':'DATA 266 Lab 2 Combined Findings - Draft','/Author':author})
    target=ROOT/'docs/Final_Lab_Report_DRAFT.pdf'
    with target.open('wb') as handle:
        writer.write(handle)
    print('Created combined draft:',target,'pages:',len(writer.pages))
    print('Human ratings, unrun model evaluations and team/external confirmations remain explicitly pending.')


if __name__=='__main__':
    main()
