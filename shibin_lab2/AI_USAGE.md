# AI coding assistant usage

| Tool | Version / model | Used for | Parts |
|---|---|---|---|
| Claude Code (Anthropic) | _fill in the version shown in your Claude app_ | Drafting code for the Part 1 RAG package (ingestion, retrieval, answer validation, evaluation), the Part 2 LoRA training/generation/evaluation pipeline, the Part 3 agents, orchestrator, Gradio app and evaluation harness; unit tests; notebook generators; a first draft of the Part 1 golden set (see below); README and report drafts; debugging library-version issues | 1, 2, 3 |
| _add any other assistant you used_ | | | |

How we used it and what we checked ourselves:
- Every generated file was read, run and tested by us; we can explain all of it in the demo.
- All reported numbers come from executed notebook outputs; no numbers were written by the assistant.
- Golden set (Part 1): the assistant drafted the 25 questions and gold answers from the three PDFs (reading the
  tables and the figure values) and checked that none copies or paraphrases `Questions.csv`. We verified every answer,
  page and figure against the PDFs ourselves and corrected it where needed before using it for evaluation.
- The human ratings (Parts 1, 2 and 3) and the pass/fail judgements were given by us, not by an AI tool.
- The assistant did not see the hidden test set; no answers or question-specific rules are in the code.
