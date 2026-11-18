# Contributions and knowledge transfer

Team: Pair_Programming_Team_09 · Denisha Ketan Tank and Shibin Biji Thomas Lnu

Both members worked on all three parts. The graded notebooks are in `shibin_lab2/notebooks/`;
Denisha's independent implementations and runs are in `part1_rag/`, `part2_diffusion/` and `part3_agents/`.

| Part | Shibin Biji Thomas Lnu | Denisha Ketan Tank |
| :--- | :--- | :--- |
| 1 Multimodal RAG | Built the final RAG package (`shibin_lab2/part1/rag/`): ingestion with figure/table crops, VLM figure descriptions, ChromaDB + BM25 hybrid retrieval with exact figure lookup, fact store for revisions and conflicts, grounded JSON answers, per-conversation memory, Appendix C quarantine and injection filter. Ran the final Kaggle notebook (Version 2, public score 0.805) and selected it as the final submission. Committed the 25-question golden set. | Built and ran the team's earlier Part 1 notebook (v6: Qwen2-VL-7B, Qdrant, dense + BM25 + CLIP retrieval with reranking), improving the public score from 0.419 to 0.510. |
| 2 Diffusion LoRA | Data cleaning and split, the two final LoRA runs (rank 4 and rank 32), generation of 120 images, KID/CLIP/LPIPS evaluation, NSFW filtering, memorization check, Hugging Face models with model card and image dataset, and the blinded A/B evaluation. | Independent replication: two LoRA runs (rank 4 and rank 8) on the same dataset on an RTX 4090 with KID/CLIP/LPIPS, failure-mode inspection and model cards. |
| 3 Multi-agent system | Five agents, typed state, orchestration with JSON traces, failure handling, Gradio app, 15 scenarios and 10 red-team cases, and the final run with and without the critique loop. | Ran the complete workflow independently as a no-revision baseline on a Tesla T4 (25 states and traces, 0/10 red-team success, 10.6 GiB peak), confirming it fits one 16 GB GPU; built the standalone reproduction notebook. |
| Human evaluation | Part 2 rater; Part 3 rater (pass/fail for all 30 outputs) and critic accept/revise decisions. | Part 2 rater; Part 3 rater (pass/fail for all 30 outputs). The 13 Part 3 disagreements were resolved together. |
| Report and submission | Results, analysis and safety sections for all three parts; links.txt; tags. | Repository organisation, evidence packaging, first report draft and the Part 2 and Part 3 replication sections. |

Part 2 human evaluation: three raters, both members and one person outside the team.

## Meeting recordings

| Date | Topic | Participants | Recording link |
| :--- | :--- | :--- | :--- |
| 2026-11-17 | Lab 2 discussion | Denisha Ketan Tank and Shibin Biji Thomas Lnu | [discussion.mov](https://github.com/denishatank12/Lab-2/blob/p1-v1.0/discussion.mov) |
