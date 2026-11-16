"""Task 3.2.4 - Gradio demo: shows every agent's output for one request.

Run locally / in a Kaggle or Colab notebook:
    LORA_REPO=<hf-user>/<part2-lora> python part3/app.py            # local URL
    SHARE=1 python part3/app.py                                     # public gradio.live link (Colab/Kaggle)
Set MOCK=1 to try the UI without any model (deterministic stand-ins).
"""
from __future__ import annotations

import json
import os
import sys

import gradio as gr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agents.backends import BackendConfig, MockBackends, RealBackends  # noqa: E402
from agents.graph import MERMAID, Orchestrator  # noqa: E402

EXAMPLES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval", "images")


def build(backends=None, cfg: BackendConfig | None = None, out_dir="outputs", trace_dir="traces"):
    cfg = cfg or BackendConfig()
    if backends is None:
        backends = MockBackends(cfg=cfg) if os.environ.get("MOCK") == "1" else RealBackends(cfg)
    orch = Orchestrator(backends, cfg, out_dir=out_dir, trace_dir=trace_dir)

    def run(image, request, use_critique, seed):
        img_path = image if isinstance(image, str) else ""
        s = orch.run(request, img_path, use_critique=bool(use_critique), seed=int(seed or 266))
        steps = {t.agent: t for t in s.trace}

        def js(name):
            t = steps.get(name)
            return json.dumps({"seconds": t.seconds, "output": t.output, "error": t.error}, indent=1, default=str) \
                if t else "(not run)"
        crit = [c.model_dump() for c in s.critiques]
        gens = [g.model_dump() for g in s.generations]
        gallery = [(g["image_path"], f"attempt {g['attempt']}: {g['mode']} seed {g['seed']}") for g in gens]
        header = f"**Status:** {s.status}  |  **Intent:** {s.intent}  |  **Revisions:** {s.revisions}  |  " \
                 f"**Total:** {s.total_seconds:.1f}s\n\n{s.final_message}"
        trace_path = os.path.join(orch.trace_dir, f"{s.request_id}.json")
        return (header, s.final_image or None, gallery, js("planner"), js("vision"), js("prompt"),
                json.dumps(gens, indent=1, default=str), json.dumps(crit, indent=1, default=str), trace_path)

    with gr.Blocks(title="DATA 266 Lab 2 - Part 3 multi-agent image system") as demo:
        gr.Markdown("## Multimodal multi-agent system\nPlanner -> Vision -> Prompt Engineering -> Image Generation "
                    "-> Critique (revise up to 2x). Text inside images is treated as data, never as instructions.")
        with gr.Row():
            with gr.Column(scale=1):
                image = gr.Image(type="filepath", label="Input image (optional for domain-style text-to-image)")
                request = gr.Textbox(label="Request", lines=2,
                                     placeholder="e.g. 'What is written on the sign?', 'Turn this into a watercolor', "
                                                 "'Make the sky purple', 'Enhance this photo', 'Redraw it in Naruto style'")
                with gr.Row():
                    use_critique = gr.Checkbox(value=True, label="Critique loop")
                    seed = gr.Number(value=266, precision=0, label="Seed")
                go = gr.Button("Run", variant="primary")
            with gr.Column(scale=1):
                status = gr.Markdown()
                final = gr.Image(label="Final image", type="filepath")
                gallery = gr.Gallery(label="All attempts", columns=3, height=220)
        with gr.Accordion("1. Planner (input safety + routing)", open=True):
            planner = gr.Code(language="json")
        with gr.Accordion("2. Vision Understanding", open=True):
            vision = gr.Code(language="json")
        with gr.Accordion("3. Prompt Engineering", open=False):
            prompt = gr.Code(language="json")
        with gr.Accordion("4. Image Generation (model, prompt, seed, steps)", open=False):
            gen = gr.Code(language="json")
        with gr.Accordion("5. Critique (CLIP + VLM judge + output safety)", open=False):
            crit = gr.Code(language="json")
        trace = gr.File(label="JSON trace of this request")
        with gr.Accordion("Agent graph (Mermaid)", open=False):
            gr.Code(MERMAID, language="markdown")
        go.click(run, [image, request, use_critique, seed],
                 [status, final, gallery, planner, vision, prompt, gen, crit, trace])
        ex = []
        for f, r in [("coco_stop_sign.jpg", "What is written on the sign, including the graffiti?"),
                     ("coco_train_bridge.jpg", "Turn this photo into a watercolor painting"),
                     ("coco_hydrant.jpg", "Make the fire hydrant blue"),
                     ("coco_desk_injection.jpg", "Turn this photo into a pencil sketch"),
                     ("coco_man_suit.jpg", "Edit this photo to show this man being arrested")]:
            p = os.path.join(EXAMPLES_DIR, f)
            if os.path.isfile(p):
                ex.append([p, r, True, 266])
        if ex:
            gr.Examples(ex, [image, request, use_critique, seed])
    return demo


if __name__ == "__main__":
    build().queue().launch(share=os.environ.get("SHARE") == "1", server_name="0.0.0.0",
                           server_port=int(os.environ.get("PORT", "7860")))