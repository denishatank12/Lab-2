# Lab 2 findings · Team 09

Denisha Ketan Tank and Shibin Biji Thomas Lnu

**Evidence review; human ratings and other pending items are not completed results.**

## Part 1: Multimodal retrieval

### I. Task and selected implementation

The task is to answer questions from three economics PDFs without changing the knowledge base. Some questions depend on charts, similar-looking figures, revisions or earlier conversation turns. A plausible answer is not enough: its status and citations also need to match the evidence.

The selected source is the unchanged independent v6 notebook. It contains its own helper files and pinned environment. It extracts text and numbered figure/table objects, keeps document/page/figure/trust metadata, and indexes text and visual embeddings in local Qdrant. The answer model is Qwen2-VL-7B with 4-bit language weights and float16 vision. The random seed is 266.

Read-only PDFs and trust manifest → Text chunks and numbered visual objects → Dense + BM25 + CLIP retrieval and reranking → Session memory + grounded JSON answer

Fig. 1. The final retrieval path. Conversation assumptions are separate from the document index.

### II. Evaluation design

The intended comparison is dense-only retrieval against the final multimodal hybrid system using the same answer model. The 25 candidate golden questions include facts/tables, figure identity, two-source questions, multiple turns, conflicts/revisions and unsupported requests. Team review of those references is still required. Correctness, Recall@5, citation support, median latency and both members' groundedness ratings must be reported from the selected run.

| Evidence | Current record |
| :--- | :--- |
| Denisha public Kaggle score | 0.510, screenshot and live submission details |
| Newer partner submission | 0.805, Shibin, notebook Version 2 |
| Selected source | v6, unchanged hash recorded |
| Original executed v6 / CSV / metrics | Not available in current files |
| Both members' groundedness | Not supplied |

### III. Findings and limits

The screenshot shows an improvement from 0.419 to 0.510 on the public leaderboard. That is a leaderboard result, not a 25-question golden accuracy or a grade. The original v6 Results ZIP was supplied earlier, but its bytes are not available in the current checkout or Downloads. The later notebook with the same filename contains v8. Its outputs cannot support the earlier run.

A read-only check of Kaggle on November 17 also shows a newer completed submission by Shibin with public score 0.805. That is a separate partner result, not the score of Denisha's preserved v6 notebook. Its source has not been imported. The original 0.510 CSV is still listed, but Chrome blocked its download. No final submission choice was changed.

I would retain the selected source and restore its original executed notebook, CSV, traces and evaluation files before making a final comparison. Repeating a run could produce new evidence, but would not recreate the historical submission. No missing metric has been filled with a later result or a guessed value.

## Part 1: Safety and evidence quality

### IV. Source trust and prompt injection

The main safety issue in this task is allowing retrieved material to change the assistant's instructions. A PDF can contain relevant data and an instruction-like sentence on the same page. The implementation treats extracted content as evidence rather than executable instructions. DOC3 Appendix C is quarantined as unverified and is excluded from answer evidence. Image text needs the same treatment as ordinary extracted text.

Document trust does not mean every statement is current. A forecast and an actual outturn can both be correctly quoted while answering different questions. The manifest, vintage and figure identity should therefore travel with every retrieved chunk. Corrected versions and explicit errata take precedence under the stated policy. An unresolved disagreement should retain both sources and explain the preferred source rather than merge inconsistent values.

A near-identical chart is a difficult failure case. Similar colour, axes or wording can make a visual match look convincing even when the base period differs. Exact figure identifiers, page references and original visual crops make that mistake reviewable. Reranking can improve retrieval, but it cannot establish that the answer used the right series or unit. That requires checking the question against the cited evidence.

### V. Conversation boundaries

A user may introduce an assumption for one conversation. That assumption belongs in session-local memory, alongside the document facts. It must not become a permanent vector-store entry or silently change an official value. Conversation IDs isolate the state. Requests to rewrite or permanently update the corpus should be refused. An assistant's earlier answer is not an independent source for a later answer.

Status selection is part of the answer, not just formatting. A missing passage should lead to not_found; an unsupported premise should be identified; an explicit chat assumption should be marked as a session override. A syntactically valid JSON object can still have the wrong status, figure or factual claim. Format validation is useful for Kaggle compatibility but does not demonstrate grounding.

### VI. Review still required

The original selected execution is needed to inspect actual wrong answers, source choices, runtime and citations. I have not claimed that every injection was rejected or that the 90-minute requirement was met on the selected evidence now present. Both members should independently read each answer and source, rate groundedness from 1 to 5, retain their original judgments and calculate agreement.

The public leaderboard gives feedback on a hidden subset. Repeated changes aimed only at that score may overfit it. The own-question set should remain independent of the competition questions and should be reviewed before model tuning. Test prompts, source hashes and validation records need to remain tied to the exact experiment. A later result with a reused filename must not be relabeled as an earlier run.

The active repository no longer contains raw PDF files. The frozen notebook still uses its existing hash-pinned historical resource URLs, so Git history was preserved rather than rewritten. Course PDFs and other raw datasets should be shared through the permitted course or artifact channel. Keeping a public repository is the user's explicit choice; it differs from the assignment's private-repository requirement.

## Part 2: LoRA domain adaptation

### I. Data and training

The experiment uses lambdalabs/naruto-blip-captions at its recorded revision. The 1,221 usable captioned images were split into 1,098 training and 123 held-out images with seed 266. The recorded cleanup found no corrupt images, duplicate hashes or empty captions. Inputs were RGB at 512 by 512 pixels.

Both adapters use the same SD 1.5 base, data, AdamW learning rate 0.0001, 1,200 updates, batch size 1 and accumulation of 4. The changed parameter is LoRA rank, 4 versus 8. Training used an RTX 4090. The original executed notebook is preserved with its outputs.

Captioned images: fixed train/test split → SD 1.5 frozen base + LoRA r4 or r8 → 20 fixed prompts x 2 seeds x 3 models → KID, CLIP, LPIPS and blinded human review

| Run | Params | Loss | Seconds | Peak MiB |
| :--- | :--- | :--- | :--- | :--- |
| run1 | 797,184 | 0.03940 | 764.54 | 1949.88 |
| run2 | 1,594,368 | 0.03926 | 772.98 | 1960.89 |

Table I. Optimizer-loop measurements. GPU-hours were 0.2124 for r4 and 0.2147 for r8. These memory peaks are PyTorch allocations, not total pipeline or driver memory.

### II. Matched generation and results

The fixed 20 prompts contain ten held-out in-domain prompts, five out-of-domain prompts and five compositional prompts. Seeds 0 and 1 produce 120 images across the base and two adapters. All use DDIM with 20 steps, CFG 7.5, eta 0 and the same resolution. Eight side-by-side grids are saved.

| Model | KID mean | CLIP | LPIPS | Filtered |
| :--- | :--- | :--- | :--- | :--- |
| base | 0.1274 | 31.238 | 0.653 | 0 |
| run1 | 0.0681 | 29.536 | 0.755 | 6 |
| run2 | 0.0941 | 29.488 | 0.646 | 7 |

Table II. KID uses 20 in-domain generations against 123 test images. Lower KID is preferable; CLIP and LPIPS measure different properties. Small-sample KID is noisy.

I would choose r4 for the domain model: it has the lowest KID and half the adapter parameters of r8. The base has the highest CLIP score, so fine-tuning did not improve every measure. The larger r4 LPIPS cannot by itself establish useful diversity because filtered black images affect the metric. Independent human style and prompt-match judgments are still pending.

## Part 2: Failures, safety and rights

### III. Failure analysis

AI-assisted inspection of the saved grids found an attribute-binding issue: in the green-eyes/black-hair portrait, green spreads across the face while the eyes are not clearly green. The style changes, but the requested colour is not reliably attached to the right feature. More varied captions and an explicit attribute-following test would be more informative than CLIP alone.

A second example asks for a group with one person holding a sword. More than one character appears to hold a blade. The base comparison also struggles with this relationship. This suggests that learning the domain appearance is easier than learning exact counts and ownership. It is not evidence that LoRA uniquely caused the counting problem.

Some fine-tuned outputs are black because the enabled safety checker filtered them: six images for r4 and seven for r8. These are unusable outputs and remain counted. The original unfiltered images are not available, so a black result does not prove a false positive. Silently regenerating until an attractive image appears would change the evaluation protocol.

### IV. Memorization and evaluation limits

The saved memorization analysis compares ten generated images with training images using CLIP cosine similarity, with the three closest pairs displayed. Similar composition or domain style is a warning to inspect, not proof of copying. Conversely, a limited nearest-neighbor check cannot prove that the model never memorizes. Additional seeds, more queries and perceptual duplicate checks would strengthen that review.

KID, CLIP and LPIPS cannot replace the required human experiment. KID estimates distribution similarity and is unstable with this small generated sample. CLIP can overlook counts and spatial relationships. LPIPS measures seed variation, including undesirable variation. The blinded comparison needs 25 pairs rated independently by both team members and an outside person. Style and prompt match are separate judgments; the report must retain ties and calculate win rates from actual ballots.

### V. Safety, licenses and hardware

The dataset depicts copyrighted Naruto imagery and its card has no explicit license field. This course experiment does not create ownership or broad redistribution rights. Use remains limited to non-commercial research subject to the source-image rights and SD 1.5 CreativeML Open RAIL-M restrictions. The team still needs to confirm permitted artifact sharing with the instructor. A private Hub repository limits access but does not grant a license.

The safety checker stays enabled. Prompt filtering, nearest-neighbor review and a model card are useful safeguards but do not guarantee safe or original content. The adapter should not be used for deceptive real-person imagery or claims that generated content is authentic evidence. The model card records the domain bias, failures, filtering and limits rather than presenting the adapter as a general improvement.

Both adapters and the 120-image dataset have recorded private Hugging Face commits and content hashes. Grader access is not yet confirmed. The run was measured on 24-GB RTX 4090; the small training allocation does not establish a full run on a single 16-GB GPU. A compatibility check must include model loading, training and evaluation under the required limit, with recorded hardware and memory evidence. It must not relabel the original hardware.

## Part 2: Visual evidence appendix

### Saved comparison examples

The panels are copied from the original completed experiment and converted to grayscale only for this report. The notebook and Hub images retain their original colours. From left to right: base, r4, r8.

Saved visual example: `part2_diffusion/outputs/generations/comparison_02.png` (included in the local PDF; original images are in the result artifacts).

Fig. 4. Attribute-binding example. The colour-specific observation in the preceding analysis must be checked in the original colour notebook.

Saved visual example: `part2_diffusion/outputs/generations/comparison_04.png` (included in the local PDF; original images are in the result artifacts).

Fig. 5. Counting/ownership example: the prompt specifies one sword-holder. Automatic alignment does not reliably check that relationship.

### Three closest saved r4 generated/training pairs

Saved visual example: `part2_diffusion/outputs/memorization_run1_0.png` (included in the local PDF; original images are in the result artifacts).

Saved visual example: `part2_diffusion/outputs/memorization_run1_1.png` (included in the local PDF; original images are in the result artifacts).

Saved visual example: `part2_diffusion/outputs/memorization_run1_2.png` (included in the local PDF; original images are in the result artifacts).

Fig. 6. The three displayed nearest-neighbor pairs from the recorded ten-image CLIP check. Similarity is not by itself proof of copying or proof that memorization is absent.

## Part 3: Multi-agent image workflow

### I. Architecture and comparison

The partner workflow has five agents: planner, visual understanding, prompt rewriting, image generation and critic. Typed state retains decisions and outputs. Qwen3-VL-8B handles visual questions and OCR; SD 1.5 and InstructPix2Pix handle style and instruction edits. The Naruto route loads Shibin's public rank-4 adapter. It is a separate checkpoint from Denisha's Part 2 adapter.

Planner: route request and check safety → Vision: objects, questions and visible text → Prompt rewrite + style/edit/domain generation → CLIP + VLM critic; at most 2 revisions

Fig. 3. Final system flow. The independently executed baseline uses zero revisions while retaining safety and diagnostic checks.

Fifteen frozen COCO-based scenarios cover five task categories. Ten additional cases test unsafe/deceptive requests, image injection and invalid inputs. Shibin's executed RTX 4090 notebook supplies the paired baseline/final comparison. Denisha separately completed the no-revision baseline on one Tesla T4.

| Condition | Auto proxy | Median s | Attack rate |
| :--- | :--- | :--- | :--- |
| Shibin baseline / 4090 | 15/15 | 20.671 | Not separately measured |
| Shibin final / 4090 | 15/15 | 20.402 | 0/10 |
| Denisha baseline / T4 | 15/15 | 39.665 | 0/10 |

Table III. Automatic checks, not human task success. Cross-GPU times are descriptive and do not measure the causal benefit of critique. Partner final mean revisions: 0.67.

### II. Baseline verification and limitations

The T4 ZIP contains a genuinely executed notebook with no saved cell errors, 25 states and 25 traces. Adapter loading and GPU preflight passed. Peak PyTorch allocation was 10.635 GiB, with 10.836 GiB reserved on GPU0. This verifies the baseline on a nominal 16-GB card; it is not proof that all lab workloads fit.

The automatic proxy can miss poor image quality. S06 asks for a pencil sketch, but the saved baseline output remains a colour photograph. S08 visibly adds snow, while S11 produces an anime-style scene. These are AI-assisted observations, not the required independent judgments. Both members must rate the paired baseline/final outputs and calculate human success, critic agreement and resolved disagreement counts.

## Part 3: System safety and deployment limits

### III. System card and intended use

This is a course prototype for image understanding and controlled editing. It is not an autonomous system with general computer access, a reliable authenticity detector or a dependable safety judge. The Gradio interface exposes the stages so a user can inspect the route, visual description, rewritten prompt, output and critique. The team still needs to rehearse the required live examples, including a safe failure.

The input photos and source revisions are recorded. Generated outputs are saved separately from inputs, with per-request trace identifiers. Model revisions, adapter choice and generation settings are bound to the baseline record. This makes a failure traceable, but a trace does not prove that the underlying model description or critic decision is correct.

### IV. Injection and unsafe generation

Text read from an image is data. It should not change the application's instruction hierarchy or request access to files or credentials. The red-team set includes image-pixel instructions as well as unsafe and deceptive requests. The generation safety checker remains enabled, and failures to check safety should withhold the output rather than silently bypass the check. No access token is embedded in the reproduction notebook; the chosen adapter is public.

The recorded attack rate is zero in ten cases for the final partner run and Denisha baseline. That small fixed set does not establish broad injection resistance. Some cases are rejected before generation, while others exercise visual understanding. Each case needs its expected safe outcome checked. An unrelated model exception cannot be treated as a successful refusal.

S14, S15 and R09 intentionally use invalid or empty inputs and produce clear error responses. Those statuses do not mean the notebook crashed. The completed baseline retains them as test outcomes. A nonfatal host sitecustomize warning about wrapt also appears in its log; the evaluation and export still completed. The original log is preserved rather than edited to hide the warning.

### V. Quality, privacy and human oversight

A critic can confidently accept a weak edit. CLIP measures broad alignment and can miss exact attributes, text or transformations. VLM descriptions can hallucinate details or miss a subtle change. The visible sketch failure demonstrates why automatic route/status checks are not a sufficient quality metric. Increasing the revision count is not a guaranteed fix and can amplify drift or latency.

The final workflow caps revisions at two; the independent baseline allows none. Both members should first rate the actual requested transformation independently, compare their judgments with critic decisions and record how disagreements were resolved. Until then, no human success rate, human agreement or critic-human agreement is reported. Faster request timing alone cannot support a quality conclusion.

COCO source photos retain their creator licenses. Naruto-domain generation remains subject to source-image and model restrictions. Users should not upload sensitive images to a public demo without a clear sharing decision. The app should stay private by default, avoid public share links, and not imply that edited images are authentic evidence. Input/output artifact hosting and grader access are still pending; Git contains source, manifests and traces rather than a raw image dataset.

## Part 3: Baseline image inspection

These are original saved baseline outputs from the completed T4 run, shown in grayscale for this report. They are illustrative AI-assisted observations, not independent human ballots.

Saved visual example: `output/review_assets/part3/baseline_outputs/generated/20261117-062558-60168c/attempt0.png` (included in the local PDF; original images are in the result artifacts).

Fig. 7. S06: the request asks for a pencil sketch. The output still has photographic texture and shading. The automatic proxy nevertheless counted the case as a pass.

Saved visual example: `output/review_assets/part3/baseline_outputs/generated/20261117-062729-28d330/attempt0.png` (included in the local PDF; original images are in the result artifacts).

Fig. 8. S08: visible snow appears around the car. A person should still judge whether the requested edit preserves the subject and avoids artifacts.

Saved visual example: `output/review_assets/part3/baseline_outputs/generated/20261117-062945-ed4f30/attempt0.png` (included in the local PDF; original images are in the result artifacts).

Fig. 9. S11: anime-style scene. The domain route loaded Shibin's actual public rank-4 adapter; it was not replaced by the base model.

## Submission record and references

### I. Team and reproducibility

Team 09 consists of Denisha Ketan Tank and Shibin Biji Thomas Lnu. Shibin's executed work is imported unchanged and attributed to its original commit. Denisha's T4 baseline is a separate measured execution. AI assistance helped with debugging, packaging and evidence review; the disclosure is in AI_USAGE.md. Both members still need to review the complete report and contribution record.

The GitHub repository is https://github.com/denishatank12/Lab-2. Denisha explicitly requested keeping this name and public visibility. The assignment instead asks for a private data266-lab2-team09 repository with grader access. The difference is recorded openly and is not treated as compliance.

### II. Remaining submission obligations

part1 original selected execution

part1 team reference review

part1 real groundedness ratings

part2 three real raters

part2 rights and grader access

part3 real human ratings

part3 hub artifacts and access

required repository name and privacy

recordings and contribution review

final kaggle selection and team

single 16gb all workloads

report reviewed by both members

three final tags

live demo rehearsed

The local Canvas review ZIP contains this PDF, CONTRIBUTIONS.md, AI_USAGE.md and links.txt. It is not automatically submitted. Final tags are deferred until the missing evidence is incorporated. Meeting dates, recording URLs, human ballots and external confirmations are not generated by code.

### III. References and artifact locations

DATA 266 Fall 2026 Lab 2 assignment requirements, pp. 1-12, and lecture slides. Repository manifests record exact source and model revisions. Private Part 2 Hub URLs and commits, partner source, adapter and selected Part 1 source are listed in links.txt.

Lewis et al., Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks, 2020. Hu et al., LoRA: Low-Rank Adaptation of Large Language Models, 2021. Rombach et al., High-Resolution Image Synthesis with Latent Diffusion Models, 2022. Wang et al., Qwen2-VL, 2024. Brooks et al., InstructPix2Pix, 2023.

This report was generated from the saved evidence without repeating training or inference. Missing values remain pending. The original selected Part 1 GPU record is required before its complete experimental findings can be added.
