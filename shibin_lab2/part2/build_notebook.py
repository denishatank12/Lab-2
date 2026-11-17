"""Generate notebooks/p2_finetune.ipynb (self-contained: the p2 package is written with %%writefile).

    python part2/build_notebook.py
"""
import glob
import os
import subprocess

import nbformat as nbf

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "notebooks", "p2_finetune.ipynb")


def M(s):
    return nbf.v4.new_markdown_cell(s.strip())


def C(s):
    return nbf.v4.new_code_cell(s.strip())


def main():
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    except Exception:
        commit = ""
    if not commit:  # no git on this machine (e.g. the Windows lab PC): commit of the code in this copy
        f = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "LAB_COMMIT")
        commit = open(f, encoding="utf-8").read().strip() if os.path.isfile(f) else ""
    cells = [M("""
# DATA 266 Lab 2 - Part 2: Fine-tuning Stable Diffusion 1.5 for Naruto-style images

Data: `lambdalabs/naruto-blip-captions` (1,221 images with BLIP captions). Base model: `stable-diffusion-v1-5`.
Two runs with the **same data, steps, batch size and learning rate**: (1) LoRA rank 4, (2) LoRA rank 32.
Evaluation against the **base model**: KID (in-domain vs test images), CLIP score, LPIPS diversity between seeds,
blinded human comparison, GPU-hours and peak memory; safety: license, NSFW filter, memorization check.

Sections: 0 setup | 1 data (2.1) | 2 training (2.2) | 3 generation (2.3) | 4 evaluation (2.4) | 5 human evaluation |
6 failure modes | 7 safety (C4) | 8 Hugging Face upload | 9 results table.
""")]
    cells.append(C(f"""
# Cell 1 - run information (required)
import subprocess, sys, random, numpy as np, torch
SEED = 266; random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
try:
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
    text=True).stdout.strip()
except OSError:  # git not installed
    commit = ""
commit = commit or "{commit}"  # copy without .git: commit the notebook was generated from (tag p2-v1.0)
print({{"commit": commit, "python": sys.version.split()[0], "torch": torch.__version__,
       "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", "seed": SEED}})
"""))
    cells.append(M("## 0. Setup"))
    cells.append(C("""
import importlib.metadata as md, subprocess, sys
PINS = {'diffusers': '0.43.0', 'peft': '0.21.2', 'transformers': '5.22.2', 'accelerate': '1.15.0', 'datasets': '5.2.0',
        'torchmetrics': '1.9.0', 'torch-fidelity': '0.4.0', 'lpips': '0.1.4'}
def _ver(p):
    try:
        return md.version(p)
    except md.PackageNotFoundError:
        return None
todo = [f"{p}=={v}" for p, v in PINS.items() if _ver(p) != v]
if todo:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *todo], check=True)
print({p: _ver(p) for p in PINS})
"""))
    cells.append(C("import os; os.makedirs('p2', exist_ok=True); os.makedirs('eval_p2', exist_ok=True)"))
    for path in sorted(glob.glob(os.path.join(HERE, "p2", "*.py"))):
        with open(path, encoding="utf-8") as fh:
            cells.append(C(f"%%writefile p2/{os.path.basename(path)}\n{fh.read()}"))
    with open(os.path.join(HERE, "eval", "prompts.json"), encoding="utf-8") as fh:
        cells.append(C(f"%%writefile eval_p2/prompts.json\n{fh.read()}"))
    cells.append(C("""
import os, sys, json, time
sys.path.insert(0, os.getcwd())
for m in [m for m in list(sys.modules) if m == "p2" or m.startswith("p2.")]:
    del sys.modules[m]
import pandas as pd, matplotlib.pyplot as plt
from PIL import Image
from p2.config import P2Config
from p2 import data as D, train as T, generate as G, evaluate as E, hub as H
pd.set_option("display.max_colwidth", 90); pd.set_option("display.width", 220)
OUT = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
cfg = P2Config(work_dir=os.path.join(OUT, "artifacts_p2"))
PROMPTS = json.load(open("eval_p2/prompts.json", encoding="utf-8"))
print(cfg)
"""))
    cells.append(M("## 1. Data preparation (Task 2.1)\nCorrupted = cannot be decoded, smaller than 128 px, blank, or empty "
                   "caption. Duplicates = identical pixels or a 64-bit difference hash within 4 bits. Split 90/10 with seed 266 "
                   "on the cleaned images; every image is resized (short side) and centre-cropped to 512x512."))
    cells.append(C("""
t0 = time.time()
prep = D.prepare(cfg)
rep = prep["report"]
print("cleaning report:", rep.summary())
print("removed as corrupted / unusable:", rep.corrupted)
print("exact duplicates:", rep.exact_duplicates, "| near duplicates:", rep.near_duplicates)
TRAIN, TEST = D.load_split(cfg, "train"), D.load_split(cfg, "test")
print(f"train={len(TRAIN)} test={len(TEST)}  ({time.time() - t0:.0f}s)")
print("train captions:", D.describe(TRAIN))
print("test captions:", D.describe(TEST))
fig, axes = plt.subplots(2, 6, figsize=(18, 6.5))
for ax, r in zip(axes.flat, TRAIN[:12]):
    ax.imshow(Image.open(r["path"])); ax.set_title(r["text"][:38], fontsize=8); ax.axis("off")
plt.tight_layout(); plt.show()
"""))
    cells.append(M("""
**Dataset description.** Frames of Naruto characters from Narutopedia, each with one short BLIP caption
(statistics above). *Limitations / biases* (check them against the statistics and examples above and edit in your own words):
1. The captions are short and generic (BLIP, mostly "a man with ..."), so they describe hair colour and clothing but not
   the art style, names or actions; many captions repeat or contain BLIP artefacts ("a hoodie and a hoodie").
2. The images are skewed towards a few main (mostly male) characters and close-up portraits, so the model will be
   better at faces than at scenes, landscapes or objects; it also inherits the franchise's copyrighted designs.
"""))
    cells.append(M("## 2. Fine-tuning (Task 2.2): two runs, same data / steps / batch size / learning rate"))
    cells.append(C("""
STATS = {}
for name, rank in cfg.runs:
    f = os.path.join(cfg.work_dir, "lora", name, "train_stats.json")
    if os.path.isfile(f) and os.environ.get("P2_RETRAIN", "0") != "1":
        STATS[name] = json.load(open(f))  # a previous run of this notebook trained it (set P2_RETRAIN=1 to train again)
        print(f"[train {name}] loaded existing adapter and its measured statistics from {f}")
    else:
        STATS[name] = T.train_lora(cfg, name, rank, TRAIN)
cols = ["run", "method", "trainable_params", "learning_rate", "steps", "batch_size", "epochs", "resolution",
        "peak_gpu_memory_gb", "train_time_s", "gpu_hours", "final_loss_avg100", "device"]
train_table = pd.DataFrame([{c: s[c] for c in cols} for s in STATS.values()])
display(train_table)
plt.figure(figsize=(7, 3))
for n, s in STATS.items():
    plt.plot([25 * i for i in range(len(s["loss_curve"]))], s["loss_curve"], label=n)
plt.xlabel("step"); plt.ylabel("loss (25-step mean)"); plt.legend(); plt.title("Training loss"); plt.show()
"""))
    cells.append(M("## 3. Image generation (Task 2.3): 20 fixed prompts x seeds 0 and 1, base model and both runs"))
    cells.append(C("""
models = {"base": None, **{n: s["lora_dir"] for n, s in STATS.items()}}
t0 = time.time()
ROWS = G.generate_all(cfg, PROMPTS["prompts"], models, os.path.join(cfg.work_dir, "generated"))
print(len(ROWS), "images in", round(time.time() - t0), "s;", pd.DataFrame(ROWS).groupby("model").size().to_dict())
grid_path = E.grid(ROWS, PROMPTS["grid_prompts"], list(models), os.path.join(OUT, "p2_grid.png"), seed=0)
plt.figure(figsize=(14, 24)); plt.imshow(Image.open(grid_path)); plt.axis("off"); plt.title("seed 0: base | run 1 | run 2"); plt.show()
"""))
    cells.append(M("## 4. Evaluation (Task 2.4): KID, CLIP score, LPIPS diversity"))
    cells.append(C("""
clip = E.Clip()
for r, s in zip(ROWS, clip.score([r["path"] for r in ROWS], [r["prompt"] for r in ROWS])):
    r["clip_score"] = s
LP = E.lpips_pairs(ROWS)
TEST_PATHS = [r["path"] for r in TEST]
SUMMARY = E.summary_table(ROWS, TEST_PATHS, STATS, LP)
display(SUMMARY)
display(pd.DataFrame(ROWS).groupby(["model", "kind"]).clip_score.mean().unstack().round(2))
display(LP.groupby(["model", "kind"]).lpips.mean().unstack().round(3))
# suggested final run: lowest in-domain KID, ties broken by in-domain CLIP (the team makes the final choice)
fine = SUMMARY[SUMMARY.model != "base"].copy()
fine["kid"] = fine["KID in-domain (x1e3, lower=better)"].str.split(" ").str[0].astype(float)
FINAL = os.environ.get("P2_FINAL") or fine.sort_values(["kid", "CLIP in-domain"], ascending=[True, False]).iloc[0]["model"]
print("final run:", FINAL)
"""))
    cells.append(M("## 5. Human evaluation\n25 blinded pairs (final run vs base model, left/right randomised with seed 266). "
                   "Three raters (both members + one outside person) each write A or B for *domain style* and for *prompt "
                   "match* in `p2_human_eval.csv`, without opening the KEY file. Re-run this cell once the sheet is filled in."))
    cells.append(C("""
HUMAN_DIR = os.path.join(OUT, "p2_human_eval")
filled = next((p for p in ["../part2/eval/p2_human_eval.csv", "part2/eval/p2_human_eval.csv"] if os.path.isfile(p)), "")
sheet, key = E.human_eval_sheets(ROWS, FINAL, "base", HUMAN_DIR)
if filled:
    sheet = filled
HUMAN = E.human_eval_stats(sheet, key)
print("sheet:", sheet, "| key:", key)
print(HUMAN or "not rated yet: fill rater1/2/3 columns with A or B and re-run")
"""))
    cells.append(M("## 6. Failure modes\nCandidates found automatically: the final run's lowest CLIP scores (prompt ignored) and "
                   "its most similar seed pairs (repeated outputs). Describe at least two failure modes in your own words."))
    cells.append(C("""
df = pd.DataFrame(ROWS)
worst = df[(df.model == FINAL) & ~df.nsfw_filtered].sort_values("clip_score").head(4)
same = LP[LP.model == FINAL].sort_values("lpips").head(2)
show = list(worst.itertuples())
fig, axes = plt.subplots(1, len(show) + 2 * len(same), figsize=(3.2 * (len(show) + 2 * len(same)), 3.6))
k = 0
for r in show:
    axes[k].imshow(Image.open(r.path)); axes[k].set_title(f"{r.prompt_id} CLIP {r.clip_score:.1f}\\n{r.prompt[:30]}", fontsize=8); k += 1
for r in same.itertuples():
    for s in (0, 1):
        p = df[(df.model == FINAL) & (df.prompt_id == r.prompt_id) & (df.seed == s)].path.iloc[0]
        axes[k].imshow(Image.open(p)); axes[k].set_title(f"{r.prompt_id} seed {s}\\nLPIPS {r.lpips:.3f}", fontsize=8); k += 1
for ax in axes: ax.axis("off")
plt.tight_layout(); plt.show()
"""))
    cells.append(M("## 7. Safety (C4): license, NSFW filter, memorization check\n**License:** Naruto characters are "
                   "copyrighted; the dataset card states no license, so the model is for non-commercial research only. "
                   "SD 1.5 is CreativeML OpenRAIL-M. **NSFW filter:** every generated image is screened by an NSFW image "
                   "classifier (`Falconsai/nsfw_image_detection`); filtered images are blurred and excluded from all metrics. "
                   "The Stable Diffusion safety checker's flags are recorded for comparison (on our first run it blacked out "
                   "many harmless anime portraits); review its extra flags below. **Memorization:** for 10 in-domain images "
                   "of the final run, the most similar training image by CLIP image embedding; the 3 closest pairs are shown."))
    cells.append(C("""
print("NSFW classifier - filtered images per model:", df.groupby("model").nsfw_filtered.sum().to_dict())
print("SD safety checker - would flag per model:   ", df.groupby("model").sd_checker_flag.apply(lambda x: int(x.fillna(False).astype(bool).sum())).to_dict())
sd_only = df[df.sd_checker_flag.fillna(False).astype(bool) & ~df.nsfw_filtered]
print(len(sd_only), "images flagged by the SD checker but not by the classifier - review them below (false positives?)")
if len(sd_only):
    k = min(8, len(sd_only))
    fig, axes = plt.subplots(1, k, figsize=(2.6 * k, 3))
    for ax, r in zip(np.atleast_1d(axes), sd_only.head(k).itertuples()):
        ax.imshow(Image.open(r.path)); ax.set_title(f"{r.model}\\n{r.prompt_id} s{r.seed} p={r.nsfw_prob:.2f}", fontsize=7); ax.axis("off")
    plt.tight_layout(); plt.show()
ok = df[~df.nsfw_filtered]
mem_gen = list(ok[(ok.model == FINAL) & (ok.kind == "in_domain")].sort_values(["seed", "prompt_id"]).path)[:10]
TOP3, MEM = E.memorization(clip, mem_gen, [r["path"] for r in TRAIN], k=3)
display(pd.DataFrame(MEM)[["generated", "nearest_train", "cosine"]].assign(
    generated=lambda d: d.generated.map(os.path.basename), nearest_train=lambda d: d.nearest_train.map(os.path.basename)))
fig, axes = plt.subplots(2, 3, figsize=(10, 7))
for j, m in enumerate(TOP3):
    axes[0, j].imshow(Image.open(m["generated"])); axes[0, j].set_title(f"generated (cos {m['cosine']:.3f})", fontsize=9)
    axes[1, j].imshow(Image.open(m["nearest_train"])); axes[1, j].set_title("nearest training image", fontsize=9)
for ax in axes.flat: ax.axis("off")
plt.tight_layout(); plt.show()
"""))
    cells.append(M("## 8. Upload to Hugging Face (both adapters + model card for the final run; all generated images)\n"
                   "Needs `huggingface-cli login` (or the HF_TOKEN variable / Kaggle secret). Without it this cell only "
                   "prints what it would push; Part 3 can also load the adapter from the local folder."))
    cells.append(C("""
def results_md(name):
    r = SUMMARY.set_index("model")
    keep = ["KID in-domain (x1e3, lower=better)", "CLIP score all", "CLIP in-domain", "CLIP out-of-domain", "LPIPS diversity"]
    return r.loc[["base", name], keep].to_markdown()
data_info = {"train": len(TRAIN), "test": len(TEST), "removed": rep.summary()["total"] - rep.summary()["kept"]}
REPOS = {}
for name, s in STATS.items():
    REPOS[name] = H.push_lora(cfg, s, results_md(name), data_info)
IMAGES_REPO = H.push_images(cfg, ROWS)
print("models:", REPOS, "| images:", IMAGES_REPO)
print("local adapter for Part 3 (LORA_REPO):", os.path.abspath(STATS[FINAL]["lora_dir"]))
"""))
    cells.append(M("## 9. Results table (C2): base model (baseline) vs fine-tuned runs"))
    cells.append(C("""
table = SUMMARY.copy()
if HUMAN:
    table["human win rate vs base (style / prompt)"] = [
        "-" if m != FINAL else f"{HUMAN.get('win_rate_style', '-')} / {HUMAN.get('win_rate_prompt', '-')}" for m in table.model]
display(table)
table.to_csv(os.path.join(OUT, "p2_results_table.csv"), index=False)
train_table.to_csv(os.path.join(OUT, "p2_training_table.csv"), index=False)
pd.DataFrame(ROWS).to_csv(os.path.join(OUT, "p2_generated.csv"), index=False)
json.dump({"final_run": FINAL, "summary": SUMMARY.to_dict(orient="records"), "training": train_table.to_dict(orient="records"),
           "human": HUMAN, "memorization_top3": TOP3, "repos": REPOS, "images_repo": IMAGES_REPO,
           "local_final_adapter": os.path.abspath(STATS[FINAL]["lora_dir"])},
          open(os.path.join(OUT, "p2_results.json"), "w", encoding="utf-8"), indent=1, default=str)
print("saved p2_results_table.csv, p2_training_table.csv, p2_generated.csv, p2_results.json")
"""))
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                      "language_info": {"name": "python"}, "accelerator": "GPU"}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    nbf.write(nb, OUT)
    print("wrote", OUT, len(cells), "cells")


if __name__ == "__main__":
    main()
