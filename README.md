# DATA 266 Lab 2 · Pair_Programming_Team_09

Denisha Ketan Tank and Shibin Biji Thomas Lnu

## Graded submission (tags p1-v1.0, p2-v1.0, p3-v1.0)

The graded code and executed notebooks are in **[`shibin_lab2/`](shibin_lab2/)**. Its own
[README](shibin_lab2/README.md) explains how to run each part, and `shibin_lab2/requirements.txt` pins the versions.

| Part | Executed notebook | Main result |
| :--- | :--- | :--- |
| 1 Multimodal RAG | [`shibin_lab2/notebooks/p1_rag.ipynb`](shibin_lab2/notebooks/p1_rag.ipynb) (Kaggle `shibinbiji/notebookec4b2aff1d`, Version 2) | Kaggle public score 0.805, 40/40 answered in 38 min on a T4 |
| 2 Diffusion LoRA | [`shibin_lab2/notebooks/p2_finetune.ipynb`](shibin_lab2/notebooks/p2_finetune.ipynb) | Final model LoRA rank 4: KID 10.5 vs 132.5 (base); human win rate 0.40 style, 0.507 prompt |
| 3 Multi-agent system | [`shibin_lab2/notebooks/p3_agents.ipynb`](shibin_lab2/notebooks/p3_agents.ipynb) | Human task success 10/15 with critique vs 9/15 without; critic agreement 0.68; red team 0/10 |

Hugging Face (public): [run 1 LoRA r4](https://huggingface.co/Shibin22/sd15-naruto-run1-lora-r4) (final, with model card),
[run 2 LoRA r32](https://huggingface.co/Shibin22/sd15-naruto-run2-lora-r32),
[generated images](https://huggingface.co/datasets/Shibin22/lab2-p2-generated-images).

No hosted API is used; all models are open models run locally on one GPU.

## Submission files

* [REPORT.md](REPORT.md): source text of the report PDF (all three parts).
* [CONTRIBUTIONS.md](CONTRIBUTIONS.md), [AI_USAGE.md](AI_USAGE.md), [links.txt](links.txt).
* [discussion.mov](discussion.mov): Lab 2 discussion recording, November 17, 2026.

## Denisha's independent experiments

| Part | Notebook | What it shows |
| :--- | :--- | :--- |
| 1 | [`part1_rag/notebooks/Lab2_Part1_Kaggle_Final.ipynb`](part1_rag/notebooks/Lab2_Part1_Kaggle_Final.ipynb) | Earlier team system (v6), public score 0.510 |
| 2 | [`part2_diffusion/notebooks/Part2_FineTune_Run.ipynb`](part2_diffusion/notebooks/Part2_FineTune_Run.ipynb) | LoRA rank 4 vs rank 8 replication with KID/CLIP/LPIPS |
| 3 | [`part3_agents/notebooks/Part3_Baseline_Executed.ipynb`](part3_agents/notebooks/Part3_Baseline_Executed.ipynb) | The Part 3 workflow run without revisions on one Tesla T4 (16 GB) |

The repository is public and named `Lab-2` at Denisha's request; the assignment asks for a private
`data266-lab2-team09` repository.
