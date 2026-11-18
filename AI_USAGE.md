# AI coding assistant usage

Team 09: Denisha Ketan Tank and Shibin Biji Thomas Lnu.

| Assistant | Used by | Used for | Parts |
| :--- | :--- | :--- | :--- |
| Claude Code (Anthropic) | Shibin | Drafting and debugging the Part 1 RAG package, the Part 2 LoRA training/generation/evaluation pipeline and the Part 3 agents, orchestrator, Gradio app and evaluation harness; unit tests; notebook generators; a first draft of the 25-question Part 1 golden set; checking rating sheets for consistency; README and report drafting; Git and submission help | 1, 2, 3 |
| Codex (OpenAI) | Denisha | Code generation and debugging for her Part 1 v6 notebook, Part 2 LoRA runs and Part 3 baseline; dependency isolation; notebook packaging; regression tests; artifact validation; first report draft | 1, 2, 3 |

How we used it and what we checked ourselves:

- We ran every notebook ourselves on our own GPUs (RTX 4090, Kaggle T4); all reported numbers come from executed notebook outputs. No number was typed in by an assistant.
- The Part 1 golden set was drafted with the assistant from the three PDFs and checked not to copy or paraphrase Questions.csv. Its evaluation was not completed before the deadline (see the report).
- All human ratings (Part 2 blinded A/B votes, Part 3 pass/fail and accept/revise judgements) were given by people, not by an AI tool. The assistant only checked the filled sheets for formatting and consistency.
- AI was used to help describe failure modes visible in saved images (Part 2 attribute binding and counting, Part 3 pencil-sketch failure); these are descriptions, not measurements.
- The assistants did not see the hidden test set; the code contains no question IDs, question text or question-specific rules.
- No access token is hard-coded in the notebooks; the Hugging Face models used by the notebooks are public.
