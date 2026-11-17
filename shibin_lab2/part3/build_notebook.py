"""Generate notebooks/p3_agents.ipynb (self-contained: writes the agents package with %%writefile).

    python part3/build_notebook.py
"""
import glob
import os
import subprocess

import nbformat as nbf

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "notebooks", "p3_agents.ipynb")


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
# DATA 266 Lab 2 - Part 3: Multimodal multi-agent system for image understanding and generation

Five agents share one typed state (`AgentState`, pydantic):

**Planner** (input safety + intent/route) -> **Vision Understanding** (Qwen2.5-VL: caption, objects, visible text,
answers) -> **Prompt Engineering** (rewrites the request; never copies text found inside the image) -> **Image
Generation** (SD 1.5 img2img for style, SD 1.5 + **our Part 2 LoRA from Hugging Face** for domain style,
InstructPix2Pix for edits, PIL for enhancement; records model/prompt/seed/steps) -> **Critique** (CLIP score +
VLM judgement + output safety) -> accept, or revise back to Prompt Engineering (max 2 revisions).
Caption/question requests stop after Vision; unsafe requests are refused by the Planner.

Sections: 0 setup | 1 models | 2 test inputs | 3 one request per route | 4 scenarios with/without critique |
5 red team | 6 human evaluation | 7 results table | 8 Gradio demo.
""")]
    cells.append(C(f"""
# Cell 1 - run information (required)
import subprocess, sys, random, numpy as np, torch
SEED = 266; random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
try:
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
    text=True).stdout.strip()
except OSError:  # git not installed (e.g. some Windows lab machines)
    commit = ""
commit = commit or "{commit}"  # Kaggle/Colab copy has no .git: commit the notebook was generated from (tag p3-v1.0)
print({{"commit": commit, "python": sys.version.split()[0], "torch": torch.__version__,
       "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", "seed": SEED}})
"""))
    cells.append(M("## 0. Setup"))
    cells.append(C("""
import importlib.metadata as md, subprocess, sys
PINS = {'transformers': '5.22.2', 'accelerate': '1.15.0', 'bitsandbytes': '0.50.2', 'diffusers': '0.43.0', 'peft': '0.21.2', 'gradio': '6.31.0', 'pydantic': '2.14.3', 'qwen-vl-utils': '0.0.14', 'datasets': '5.2.0'}
def _ver(pkg):
    try:
        return md.version(pkg)
    except md.PackageNotFoundError:
        return None
todo = [f"{p}=={v}" for p, v in PINS.items() if _ver(p) != v]
if todo:  # install exactly the versions the code was tested with (torch: Kaggle's own build is kept)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *todo], check=True)
print({p: _ver(p) for p in PINS})
"""))
    cells.append(C("""
import os
# ---- configuration (edit here) ---------------------------------------------------------------
os.environ.setdefault("LORA_REPO", "")            # Part 2 adapter, e.g. "<hf-user>/sd15-naruto-lora-r16"
os.environ.setdefault("LORA_WEIGHT_NAME", "")     # e.g. "pytorch_lora_weights.safetensors" if the repo has several files
os.environ.setdefault("DOMAIN_NAME", "Naruto")    # the Part 2 dataset domain
os.environ.setdefault("P3_STEPS", "30")
print({k: os.environ[k] for k in ["LORA_REPO", "LORA_WEIGHT_NAME", "DOMAIN_NAME", "P3_STEPS"]})
if not os.environ["LORA_REPO"]:
    print("WARNING: LORA_REPO is empty - domain-style requests will use base SD 1.5 (set it to the Part 2 model).")
os.makedirs("agents", exist_ok=True); os.makedirs("eval", exist_ok=True)
"""))
    for path in sorted(glob.glob(os.path.join(HERE, "agents", "*.py"))):
        with open(path, encoding="utf-8") as fh:
            cells.append(C(f"%%writefile agents/{os.path.basename(path)}\n{fh.read()}"))
    for name in ["make_images.py", "inputs.json", "scenarios.json", "redteam.json"]:
        with open(os.path.join(HERE, "eval", name), encoding="utf-8") as fh:
            cells.append(C(f"%%writefile eval/{name}\n{fh.read()}"))
    with open(os.path.join(HERE, "app.py"), encoding="utf-8") as fh:
        cells.append(C(f"%%writefile app.py\n{fh.read()}"))
    cells.append(M("## 1. Models (loaded lazily; one 16 GB GPU)"))
    cells.append(C("""
import sys, json, time, importlib
sys.path.insert(0, os.getcwd())
for m in [m for m in list(sys.modules) if m == "agents" or m.startswith("agents.")]:
    del sys.modules[m]
import pandas as pd
from agents.backends import BackendConfig, MockBackends, RealBackends
from agents.graph import Orchestrator, MERMAID
from agents import evaluate as ev
pd.set_option("display.max_colwidth", 110); pd.set_option("display.width", 220)
OUT = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
cfg = BackendConfig()
# P3_MOCK=1 only checks the notebook's structure with stand-in models; graded runs use the real models
backends = MockBackends(cfg=cfg) if os.environ.get("P3_MOCK") == "1" else RealBackends(cfg)
print("backends:", type(backends).__name__)
orch = Orchestrator(backends, cfg, out_dir=os.path.join(OUT, "p3_outputs"), trace_dir=os.path.join(OUT, "p3_traces"))
print(cfg)
print(MERMAID)
"""))
    cells.append(M("## 2. Test inputs: 16 COCO val2014 photos (fetched from Hugging Face, listed in `eval/inputs.json`) "
                   "+ generated red-team images (instruction notes, a real photo with a planted note, broken/empty/tiny files)"))
    cells.append(C("""
import importlib.util
spec = importlib.util.spec_from_file_location("make_images", "eval/make_images.py"); mi = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mi)
IMAGES = "eval/images"
print(mi.make_all(IMAGES))
import matplotlib.pyplot as plt
from PIL import Image
names = sorted(f for f in os.listdir(IMAGES) if f.endswith((".jpg", ".png")) and os.path.getsize(os.path.join(IMAGES, f)) > 1000
               and f not in ("tiny.png",))
cols = 7
fig, axes = plt.subplots((len(names) + cols - 1) // cols, cols, figsize=(2.6 * cols, 2.4 * ((len(names) + cols - 1) // cols)))
for ax in axes.flat:
    ax.axis("off")
for ax, n in zip(axes.flat, names):
    ax.imshow(Image.open(os.path.join(IMAGES, n))); ax.set_title(n, fontsize=7)
plt.tight_layout(); plt.show()
"""))
    cells.append(M("## 3. One request per route - every agent's output"))
    cells.append(C("""
def show(s):
    print(f"status={s.status} intent={s.intent} revisions={s.revisions} total={s.total_seconds:.1f}s")
    print("message:", s.final_message)
    for t in s.trace:
        print(f"  [{t.agent:<8} {t.seconds:6.1f}s] {json.dumps(t.output, default=str)[:400]}")
    imgs = ([s.image_path] if s.image_path and os.path.isfile(s.image_path) else []) + [g.image_path for g in s.generations]
    if imgs:
        fig, axes = plt.subplots(1, len(imgs), figsize=(3.2 * len(imgs), 3.2))
        for i, (ax, p) in enumerate(zip(np.atleast_1d(axes), imgs)):
            ax.imshow(Image.open(p)); ax.axis("off")
            ax.set_title("input" if (i == 0 and s.image_path) else os.path.basename(p), fontsize=8)
        plt.tight_layout(); plt.show()

demo_requests = [("coco_stop_sign.jpg", "What is written on the sign?"),
                 ("coco_train_bridge.jpg", "Turn this photo into a watercolor painting"),
                 ("coco_hydrant.jpg", "Make the fire hydrant blue"),
                 ("coco_dark_living_room.jpg", "Enhance this photo: brighten it and sharpen it"),
                 ("coco_zebras.jpg", "Redraw this scene in Naruto anime style")]
for img, req in demo_requests:
    print("=" * 100, f"\\nREQUEST: {req!r} on {img}")
    show(orch.run(req, os.path.join(IMAGES, img)))
if torch.cuda.is_available():
    print("peak GPU memory (GB):", round(torch.cuda.max_memory_allocated() / 1e9, 2))
"""))
    cells.append(M("## 4. The 15 scenarios, with and without the critique loop (Task 3.3)\n"
                   "`scenarios.json` was committed before the system was run. `auto_pass` is a machine-checkable proxy; "
                   "the official pass/fail comes from the human judgement in section 6."))
    cells.append(C("""
scen = ev.load_cases("eval/scenarios.json", "scenarios")
with_c, states_with = ev.run_scenarios(orch, scen, IMAGES, use_critique=True)
without_c, states_without = ev.run_scenarios(orch, scen, IMAGES, use_critique=False)
cols = ["id", "category", "status", "intent", "mode", "revisions", "first_clip", "first_vlm", "first_decision", "auto_pass", "auto_why", "seconds"]
print("WITH critique"); display(with_c[cols])
print("WITHOUT critique"); display(without_c[cols])
"""))
    cells.append(C("""
# side by side: input | without critique | with critique (generation scenarios)
gen_ids = [r.id for r in with_c.itertuples() if r.final_image]
fig, axes = plt.subplots(len(gen_ids), 3, figsize=(9, 3 * len(gen_ids)))
for row, sid in zip(np.atleast_2d(axes), gen_ids):
    sc = next(x for x in scen if x["id"] == sid)
    a = without_c.set_index("id").loc[sid]; b = with_c.set_index("id").loc[sid]
    panels = [(os.path.join(IMAGES, sc["image"]) if sc["image"] else None, f"{sid} input"),
              (a.final_image or None, f"no critique"), (b.final_image or None, f"critique (rev={b.revisions})")]
    for ax, (p, t) in zip(row, panels):
        if p and os.path.isfile(p): ax.imshow(Image.open(p))
        ax.set_title(t + ("\\n" + sc["request"][:40] if "input" in t else ""), fontsize=8); ax.axis("off")
plt.tight_layout(); plt.show()
ev.copy_outputs(states_with + states_without, os.path.join(OUT, "p3_images"))
"""))
    cells.append(M("## 5. Red team (10 cases)"))
    cells.append(C("""
red = ev.load_cases("eval/redteam.json", "cases")
red_df, red_states = ev.run_redteam(orch, red, IMAGES)
display(red_df[["id", "category", "request", "status", "refusal_reason", "attack_success", "why", "seconds"]])
print("attack success rate:", f"{int(red_df.attack_success.sum())}/{len(red_df)}")
"""))
    cells.append(M("## 6. Human evaluation\nBoth members independently fill `rater_A` / `rater_B` (pass/fail) in "
                   "`p3_human_task_success.csv` using each scenario's rubric, resolve disagreements in `resolved` + "
                   "`resolution_note`, and fill `human_decision` (accept/revise) for every attempt in "
                   "`p3_human_critic_agreement.csv`. Attach the filled sheets and re-run this cell."))
    cells.append(C("""
import glob
filled1 = next((p for p in ["part3/eval/p3_human_task_success.csv", "../part3/eval/p3_human_task_success.csv", *glob.glob("/kaggle/input/**/p3_human_task_success.csv", recursive=True)] if os.path.isfile(p)), "")
filled2 = next((p for p in ["part3/eval/p3_human_critic_agreement.csv", "../part3/eval/p3_human_critic_agreement.csv", *glob.glob("/kaggle/input/**/p3_human_critic_agreement.csv", recursive=True)] if os.path.isfile(p)), "")
p1, p2 = (filled1, filled2) if (filled1 and filled2) else ev.human_sheets(with_c, without_c, states_with, OUT, scen)
hstats = ev.human_stats(p1, p2)
print("sheets:", p1, p2); print(hstats or "not rated yet")
"""))
    cells.append(M("## 7. Results table (C2)"))
    cells.append(C("""
table = ev.summary_table(with_c, without_c, red_df, hstats)
display(table)
RESULTS = {"table": table.to_dict(orient="records"), "human": hstats,
           "redteam": red_df[["id", "status", "attack_success", "why"]].to_dict(orient="records"),
           "median_s_with": float(with_c.seconds.median()), "median_s_without": float(without_c.seconds.median()),
           "lora_repo": cfg.lora_repo, "vlm": cfg.vlm_model, "clip": cfg.clip_model, "sd": cfg.sd_model,
           "pix2pix": cfg.pix2pix_model}
with_c.to_csv(os.path.join(OUT, "p3_scenarios_with_critique.csv"), index=False)
without_c.to_csv(os.path.join(OUT, "p3_scenarios_without_critique.csv"), index=False)
red_df.to_csv(os.path.join(OUT, "p3_redteam.csv"), index=False)
table.to_csv(os.path.join(OUT, "p3_results_table.csv"), index=False)
json.dump(RESULTS, open(os.path.join(OUT, "p3_results.json"), "w", encoding="utf-8"), indent=1, default=str)
print("traces saved:", len(os.listdir(orch.trace_dir)), "JSON files in", orch.trace_dir)
"""))
    cells.append(M("## 8. Gradio demo (used in the team demo)\nSet `RUN_GRADIO=1` to start it here; `share=True` "
                   "prints a public link on Kaggle/Colab."))
    cells.append(C("""
if os.environ.get("RUN_GRADIO", "0") == "1":
    from app import build
    demo = build(backends=backends, cfg=cfg, out_dir=os.path.join(OUT, "p3_outputs"), trace_dir=os.path.join(OUT, "p3_traces"))
    demo.queue().launch(share=True, debug=False)
else:
    print("RUN_GRADIO is not 1 - skipping the UI (python part3/app.py also starts it)")
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
