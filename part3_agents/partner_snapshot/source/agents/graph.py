"""Task 3.2 - orchestration with plain Python functions over one typed state.

Graph (also exported as Mermaid for the report):

    planner --(refused / error / clarify / unsupported)--> finalize
    planner --> vision --(caption_question)--> finalize
    vision --> prompt --> generate --(error)--> finalize
    generate --> critique --(revise, <= max_revisions)--> prompt
    critique --(accept / reject)--> finalize

Every node is wrapped: timing, input/output summary and errors go into
state.trace, and the whole trace is saved as JSON per request.
"""
from __future__ import annotations

import json
import os
import time
import traceback

from .agents import Critic, ImageGenerator, Planner, PromptEngineer, VisionAgent
from .backends import BackendConfig
from .state import AgentState, TraceEvent

MERMAID = """flowchart LR
  U[User request + image] --> P[Planner<br/>input safety + intent]
  P -- refuse / clarify / error --> F[Finalize]
  P --> V[Vision Understanding]
  V -- caption / question --> F
  V --> PE[Prompt Engineering]
  PE --> G[Image Generation<br/>img2img / LoRA / pix2pix / enhance]
  G -- error --> F
  G --> C[Critique<br/>CLIP + VLM judge + output safety]
  C -- revise (max 2) --> PE
  C -- accept / reject --> F
"""


def _summ(x, n=300):
    if x is None:
        return None
    s = json.dumps(x, default=str) if not isinstance(x, str) else x
    return s if len(s) <= n else s[:n] + "..."


class Orchestrator:
    def __init__(self, backends, cfg: BackendConfig | None = None, out_dir="outputs", trace_dir="traces"):
        self.cfg = cfg or BackendConfig()
        self.b = backends
        self.nodes = {
            "planner": Planner(backends, self.cfg),
            "vision": VisionAgent(backends),
            "prompt": PromptEngineer(backends, self.cfg),
            "generate": ImageGenerator(backends, self.cfg, out_dir),
            "critique": Critic(backends),
        }
        self.trace_dir = trace_dir
        os.makedirs(trace_dir, exist_ok=True)

    # ---- edges -------------------------------------------------------------
    @staticmethod
    def next_node(cur: str, s: AgentState) -> str:
        if s.status in ("refused", "error", "needs_clarification"):
            return "finalize"
        if cur == "planner":
            return "vision"
        if cur == "vision":
            return "finalize" if s.intent == "caption_question" else "prompt"
        if cur == "prompt":
            return "generate"
        if cur == "generate":
            return "critique"
        if cur == "critique":
            return "prompt" if s.critiques[-1].decision == "revise" else "finalize"
        return "finalize"

    def _input_summary(self, name, s: AgentState):
        if name == "planner":
            return {"request": s.user_request, "image": s.image_path}
        if name == "vision":
            return {"intent": s.intent, "image": s.image_path}
        if name == "prompt":
            return {"request": s.user_request, "caption": s.vision.caption if s.vision else "",
                    "revision": s.critiques[-1].revision_instruction if s.critiques else ""}
        if name == "generate":
            return {"intent": s.intent, "prompt": s.prompt.prompt if s.prompt else ""}
        if name == "critique":
            return {"image": s.generations[-1].image_path if s.generations else ""}
        return {}

    def _output_summary(self, name, s: AgentState):
        if name == "planner":
            return {"intent": s.intent, "route": s.route, "status": s.status, "refusal": s.refusal_reason,
                    "notes": s.planner_notes, "image_check": s.image_check.model_dump(), "image_text": s.image_text_probe}
        if name == "vision":
            return s.vision.model_dump() if s.vision else None
        if name == "prompt":
            return s.prompt.model_dump() if s.prompt else None
        if name == "generate":
            return s.generations[-1].model_dump() if s.generations else {"error": s.error}
        if name == "critique":
            return s.critiques[-1].model_dump() if s.critiques else None
        return None

    # ---- run ---------------------------------------------------------------
    def run(self, request: str, image_path: str = "", use_critique: bool = True, seed: int = 266,
            max_revisions: int = 2, on_step=None) -> AgentState:
        s = AgentState(user_request=request or "", image_path=image_path or "", use_critique=use_critique,
                       seed=seed, max_revisions=max_revisions)
        t_all = time.time()
        cur = "planner"
        guard = 0
        while cur != "finalize" and guard < 20:
            guard += 1
            node = self.nodes[cur]
            t = time.time()
            inp = self._input_summary(cur, s)
            err = ""
            try:
                s = node(s)
            except Exception as e:  # never crash: report a clear message
                err = f"{type(e).__name__}: {e}"
                s.status, s.error = "error", err
                s.final_message = f"The {cur} step failed ({type(e).__name__}). Please try again or rephrase the request."
                traceback.print_exc()
            s.trace.append(TraceEvent(agent=cur, started_at=t, seconds=round(time.time() - t, 3), input=inp,
                                      output=self._output_summary(cur, s), error=err))
            if on_step:
                on_step(cur, s)
            cur = self.next_node(cur, s)
        self.finalize(s)
        s.total_seconds = round(time.time() - t_all, 3)
        self.save_trace(s)
        return s

    def finalize(self, s: AgentState):
        if s.status == "running" and s.generations:
            last = s.critiques[-1] if s.critiques else None
            if last and last.decision == "reject":
                s.status = "refused"
                s.refused = True
                s.refusal_reason = f"generated image failed the output safety check ({last.safety_reason})"
                s.final_message = "The generated image did not pass the safety check, so it is not shown."
                return
            # best safe attempt: highest (vlm score, clip) among safe ones
            pairs = [(c, g) for c, g in zip(s.critiques, s.generations) if c.safe] or \
                    ([] if s.critiques else [(None, s.generations[-1])])
            if not pairs:
                s.status, s.final_message = "refused", "No safe image could be produced."
                return
            best_c, best_g = max(pairs, key=lambda cg: ((cg[0].vlm_score, cg[0].clip_score) if cg[0] else (0, 0)))
            s.final_image = best_g.image_path
            s.status = "done"
            extra = ""
            if best_c is not None:
                extra = f" (CLIP {best_c.clip_score:.1f}, judge {best_c.vlm_score}/5, revisions {s.revisions})"
                if best_c.decision == "accept" and (best_c.vlm_score < 3) and s.use_critique:
                    extra += " - best effort after the revision limit"
            s.final_message = f"Done: {best_g.mode} with {best_g.model}{(' + ' + best_g.lora) if best_g.lora else ''}{extra}."
        elif s.status == "running":
            s.status = "error"
            s.final_message = s.final_message or "Nothing was produced."
        if s.image_check.warning and s.status in ("done", "answered"):
            s.final_message += f" Note: {s.image_check.warning}."

    def save_trace(self, s: AgentState) -> str:
        path = os.path.join(self.trace_dir, f"{s.request_id}.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(s.model_dump(), fh, indent=1, default=str)
        return path