# AI assistance used for the runtime repair

## November 6 cross-part revision

AI assistance mapped the full PDF rubric to code, implemented the missing
multimodal retrieval and local vision-language answering path, rewrote the agent
model runtime and evaluation scripts, and added draft reporting, evidence exports
and regression tests. The proposed 25 golden references and scenario rubrics are
AI-assisted drafts; the team must review, revise and understand them rather than
claiming independent authorship. Their verification flags remain unset.

The revised Part 1 and Part 3 notebooks were executed locally only in explicit
CPU smoke mode. Their pretrained-model GPU measurements remain pending.
The successful earlier Part 2 GPU results were preserved, not regenerated or
altered. New upload checks bind to the actual original artifact hashes.

At the user's request, AI prepared blank real-rating forms and a separate
synthetic format example marked sample-only and do-not-submit. The examples are
not human observations, are not installed as rating files, and do not populate
the report's human measurements. Required real raters, contributions, access
permissions and live-demo confirmations cannot be invented by the assistant.

## Completed-run review and artifact upload

AI assistance also inspected the later successful GPU notebook and the original
result ZIP, imported actual saved results with backups, checked adapter tensors,
reviewed original image comparisons and the three closest training pairs, and
updated the Part 2 findings PDF. The selected rank-4 model and failure analysis
are AI-assisted interpretations of real measurements and images, not independent
human-rating results. The notebook's original GPU cells and outputs are preserved;
the appended review was actually run separately on CPU and is labeled that way.

With the user's authorization, both original adapters and the 120 generated
images were uploaded to private repositories under the verified Hugging Face
account `denishatank`. No passwords, tokens, original training-image dataset or
rater identities were published. Team details, real human judgments, sharing
permissions and grader access remain unresolved. The older failed-run account
below is historical and is superseded for Part 2 by this successful result bundle.

AI assistance was used to inspect the notebooks and requirements, identify setup
and memory problems, rewrite the shared environment setup, repair the data split,
implement cached LoRA training and actual metric calculations, repair agent image
handling and errors, and write executable smoke tests.

The validation distinguishes real dense retrieval from tiny-model tests and
model-disabled agent tests. A randomly initialized miniature diffusion model was
used to test backward passes, checkpoint resume, adapter loading and generation.
It is not presented as trained Stable Diffusion or as evidence of RTX 4050 speed,
memory usage or image quality. No human ratings or full-training scores were made up.

The corrected code still needs its full CUDA run on the student's machine.
Both members must inspect, run and understand the code and complete the external
and human-evaluation requirements. This repair is not a declaration that all
parts of the graded lab are finished.

The November 5 follow-up inspected the user's two executed notebooks. AI helped
repair the failed shared-environment installation, add an installation lock and
import-integrity checks, produce Part 2 report/audit code, validate real-rating
input schemas, and prepare opt-in Hugging Face uploads. The supplied Part 1 run
completed its 40-row validator on an RTX 4090. The supplied Part 2 run failed
during installation and did not train. The generated PDF is labeled a draft,
not a measured training report. Human ratings, team details and account setup
were explicitly deferred by the user.
