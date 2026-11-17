"""End-to-end Part 1 system: build once, then one Chat per conversation_id.

    system = RAGSystem(Config()).build()
    chat = system.new_conversation()        # empty memory
    chat.answer("What was GDP growth in 2012?")
    chat.answer("And in 2013?")             # follow-up uses this conversation only
"""
from __future__ import annotations

import json
import os
import re
import time

from .answer import (NO_ANSWER, SYSTEM, user_value_not_in_documents, VINTAGES, build_user_prompt, doc_lines, doc_vintage, finalize, parse_json,
                     policy_lines, render_evidence)
from .config import Config
from .describe import describe_figures
from .facts import FactStore
from .index import KnowledgeBase
from .ingest import Item, ingest
from .llm import free, load_describer, load_llm
from .manifest import load_manifest
from .memory import ConversationMemory
from .retrieve import Retriever


INGEST_VERSION = "2026-11-16b"  # bump when ingestion output changes, so cached items are rebuilt


def _cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


class RAGSystem:
    def __init__(self, cfg: Config | None = None, llm=None, log=print):
        self.cfg = cfg or Config()
        self.llm = llm
        self.log = log
        self.kb = None
        self.retriever = None
        self.timings = {}

    def build(self, items=None, embedder=None) -> "RAGSystem":
        cfg = self.cfg
        os.makedirs(cfg.work_dir, exist_ok=True)
        self.manifest = load_manifest(cfg.resolved_data_dir())
        cache = os.path.join(cfg.work_dir, "items_cache.json")
        key = self._cache_key()
        if items is None and os.path.isfile(cache):
            with open(cache, encoding="utf-8") as fh:
                c = json.load(fh)
            if c.get("key") == key:
                items = [Item(**{k: v for k, v in d.items() if k != "cite_key"}) for d in c["items"]]
                self.log(f"[build] reusing {len(items)} ingested/described items from {cache}")
        if items is None and self.llm is None and cfg.llm_backend == "two_stage" and _cuda():
            items = self._describe_in_subprocess()
            if not self.timings.get("describe_failed"):  # never cache a run without descriptions as if complete
                with open(cache, "w", encoding="utf-8") as fh:
                    json.dump({"key": key, "items": [it.to_dict() for it in items]}, fh)
        if items is None:
            t = time.time()
            items = ingest(cfg, log=self.log)
            self.timings["ingest_s"] = time.time() - t
            if cfg.describe_figures:
                t = time.time()
                describer = self.llm if self.llm is not None and (
                    cfg.llm_backend != "two_stage" or getattr(self.llm, "supports_images", False)) else None
                own = describer is None
                if own:
                    describer = load_describer(cfg) if cfg.llm_backend == "two_stage" else load_llm(cfg)
                describe_figures(items, describer, log=self.log)
                if own and cfg.llm_backend == "two_stage":
                    free(describer)  # the answering model needs the GPU memory
                    self.log(f"[build] unloaded {cfg.vlm_model}")
                elif own:
                    self.llm = describer
                self.timings["describe_s"] = time.time() - t
            with open(cache, "w", encoding="utf-8") as fh:
                json.dump({"key": key, "items": [it.to_dict() for it in items]}, fh)
        if self.llm is None:
            t = time.time()
            self.llm = load_llm(cfg)
            self.timings["load_llm_s"] = time.time() - t
            self.log(f"[build] answering model: {getattr(self.llm, 'name', '?')}")
        t = time.time()
        self.kb = KnowledgeBase(cfg, items, embedder=embedder, log=self.log)
        self.timings["index_s"] = time.time() - t
        self.kb.save_items()
        self.retriever = Retriever(self.kb, cfg)
        self.system_prompt = SYSTEM.format(doc_lines=doc_lines(self.manifest), policy=policy_lines(self.manifest))
        VINTAGES.clear()
        VINTAGES.update({d: doc_vintage(e) for d, e in self.manifest["docs"].items() if doc_vintage(e)})
        if os.environ.get("P1_T0"):
            self.deadline = float(os.environ["P1_T0"]) + 60 * float(os.environ.get("P1_TIME_LIMIT_MIN", "80"))
        self.facts = FactStore(items, dict(VINTAGES))  # table/text facts + precedence (final system only)
        self.log(f"[build] fact store: {len(self.facts.facts)} facts, {len(self.facts.revisions)} explicit revisions")
        return self

    def _describe_in_subprocess(self) -> list:
        """Run ingestion + VLM figure descriptions in a child process so the GPU is fully free afterwards."""
        import subprocess
        import sys
        cfg = self.cfg
        out = os.path.join(cfg.work_dir, "items_described.json")
        env = dict(os.environ, DATA_DIR=cfg.resolved_data_dir(), WORK_DIR=cfg.work_dir, VLM_MODEL=cfg.vlm_model,
                   LOAD_IN_4BIT="1" if cfg.load_in_4bit else "0", DESCRIBE_FIGURES="1" if cfg.describe_figures else "0",
                   PYTHONUTF8="1", PYTHONUNBUFFERED="1")
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # folder that contains the rag package
        t = time.time()
        self.log(f"[build] describing figures with {cfg.vlm_model} in a separate process "
                 "(model download + description: about 10-20 min on a T4) ...")
        tail = []
        p = subprocess.Popen([sys.executable, "-m", "rag.describe_cli", out], cwd=root, env=env,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in p.stdout:  # stream progress so a long step does not look stuck
            line = line.rstrip()
            tail = (tail + [line])[-40:]
            if line.startswith(("[ingest]", "[describe", "[describe_cli]")):
                self.log(line)
        p.wait()
        if p.returncode != 0 or not os.path.isfile(out):
            # never lose the submission because of the describer: fall back to captions + extracted text
            self.log("[build] WARNING: figure description failed; continuing without VLM descriptions. Last output:\n"
                     + "\n".join(tail[-15:]))
            items = ingest(cfg, log=self.log)
            self.timings.update(ingest_s=0.0, describe_s=0.0, describe_failed=True)
            return items
        with open(out, encoding="utf-8") as fh:
            d = json.load(fh)
        self.timings.update(ingest_s=d["ingest_s"], describe_s=d["describe_s"], describe_process_s=time.time() - t)
        return [Item(**{k: v for k, v in it.items() if k != "cite_key"}) for it in d["items"]]

    def _cache_key(self) -> str:
        import hashlib
        h = hashlib.sha1()
        d = self.cfg.resolved_data_dir()
        for f in sorted(os.listdir(d)):
            if f.endswith((".pdf", ".json")):
                with open(os.path.join(d, f), "rb") as fh:
                    h.update(f.encode() + hashlib.sha1(fh.read()).digest())
        describer = self.cfg.vlm_model if self.cfg.llm_backend in ("two_stage", "qwen_vl") else getattr(self.llm, "name", "")
        h.update(f"{INGEST_VERSION}|{describer}|{self.cfg.describe_figures}|{self.cfg.chunk_words}|"
                 f"{self.cfg.chunk_overlap}|{self.cfg.figure_zoom}".encode())
        return h.hexdigest()

    def with_mode(self, mode: str, use_reranker: bool = False) -> "RAGSystem":
        """A view of the same index/LLM with another retrieval mode (no re-ingestion)."""
        other = RAGSystem.__new__(RAGSystem)
        other.__dict__.update(self.__dict__)
        other.cfg = self.cfg.variant(mode=mode, use_reranker=use_reranker)
        other.retriever = Retriever(self.kb, other.cfg)
        if self.retriever is not None and getattr(self.retriever, "_reranker", None) is not None:
            other.retriever._reranker = self.retriever._reranker
        return other

    def new_conversation(self, conversation_id: str = "") -> "Chat":
        return Chat(self, ConversationMemory(conversation_id))


class Chat:
    def __init__(self, system: RAGSystem, memory: ConversationMemory):
        self.sys = system
        self.memory = memory
        self.last = {}

    PREMISE_RE = re.compile(r"^\s*(why|how come|what caused|what made|given that|since|because)\b", re.I)

    def premise_check(self, question: str, ev_text: str) -> str:
        """For questions that presuppose a fact ("Why did X contract?"), ask a focused yes/no check first."""
        if not self.PREMISE_RE.search(question) or self.memory.user_values:
            return ""
        try:
            out = self.sys.llm.chat(
                "You check whether a question's built-in assumption is true according to the evidence. "
                "Reply with TRUE, FALSE or UNKNOWN, then one short sentence citing the evidence value.",
                f"Evidence:\n{ev_text[:6000]}\n\nQuestion: {question}\nIs the situation the question assumes true "
                "according to the evidence?", max_new_tokens=60)
        except Exception:
            _free_cuda()
            return ""
        out = (out or "").strip()
        if out.upper().startswith("FALSE"):
            return ("A check of the evidence found that the question's assumption is FALSE (" + out[5:].strip(" :.-")[:200]
                    + "). Answer with status false_premise and give the documented value.")
        return ""

    def standalone(self, question: str) -> str:
        """Rewrite a follow-up ("And in 2013?") into a self-contained question using this conversation only."""
        hist = "\n".join(f"User: {t.question}" for t in self.memory.turns[-3:])
        try:
            out = self.sys.llm.chat(
                "You rewrite a follow-up question so it can be understood on its own. Reply with the rewritten "
                "question only - no answer, no explanation.",
                f"Earlier questions in this conversation:\n{hist}\nFollow-up: {question}\nRewritten question:",
                max_new_tokens=60)
        except Exception:
            _free_cuda()
            return ""
        out = (out or "").strip().strip('"').split("\n")[0].strip()
        if not out or out.startswith("{") or len(out.split()) > 45 or "?" not in out and len(out.split()) < 3:
            return ""
        return out

    def answer(self, question: str) -> dict:
        s = self.sys
        t0 = time.time()
        mem = self.memory
        ctx = mem.context_query(question)
        carry = mem.carry_labels(question)
        late = getattr(s, "deadline", None) is not None and time.time() > s.deadline
        if late and not getattr(s, "_late_warned", False):
            s._late_warned = True
            print("[answer] time limit reached: remaining questions use retrieval + fact store only", flush=True)
        standalone = self.standalone(question) if ctx and not late else ""
        retr = s.retriever.retrieve(standalone or question, context_query=ctx, carry_labels=carry)
        mem.note_user_value(question)
        resolved = []
        if s.cfg.mode == "final" and getattr(s, "facts", None) is not None:
            resolved = [] if retr.named_labels else s.facts.lookup(standalone or question)
            # the sources of every resolved value are evidence too, right after the named figures/tables
            srcs = [it for it in s.facts.items_for(resolved) if it not in retr.pinned]
            retr.evidence = (list(retr.pinned) + srcs
                             + [it for it in retr.evidence if it not in retr.pinned and it not in srcs])
        budget = s.cfg.evidence_chars
        ev_text, idmap = render_evidence(retr.evidence, budget=budget)
        premise_note = "" if late else self.premise_check(standalone or question, ev_text)
        user = build_user_prompt(question, retr, mem, ev_text, standalone=standalone, idmap=idmap,
                                 premise_note=premise_note, resolved=resolved)
        count = getattr(s.llm, "count_tokens", None)
        while count is not None and budget > 2500 and count(s.system_prompt, user) > s.cfg.max_prompt_tokens:
            budget = int(budget * 0.8)  # hard cap on the prompt length (attention memory grows with its square)
            ev_text, idmap = render_evidence(retr.evidence, budget=budget)
            user = build_user_prompt(question, retr, mem, ev_text, standalone=standalone, idmap=idmap,
                                     premise_note=premise_note, resolved=resolved)
        images = []
        if s.cfg.attach_named_figure and getattr(s.llm, "supports_images", False):
            pins = [p for p in retr.pinned if p.image_path and os.path.isfile(p.image_path)]
            corrected = {p.figure_id.replace(" (corrected)", "") for p in pins if p.figure_id.endswith("(corrected)")}
            pins = [p for p in pins if p.figure_id not in corrected]  # show the corrected chart, not the superseded one
            images = [p.image_path for p in pins[:2]]
        raw = ""
        if not late:
            try:
                raw = s.llm.chat(s.system_prompt, user, images=images, max_new_tokens=s.cfg.max_new_tokens)
            except Exception as e:  # out of GPU memory: free it and retry once with half the evidence
                if "out of memory" not in str(e).lower():
                    raise
                _free_cuda()
                print(f"[answer] out of GPU memory - retrying with half the evidence ({budget // 2} chars)", flush=True)
                ev_text, idmap = render_evidence(retr.evidence, budget=budget // 2)
                user = build_user_prompt(question, retr, mem, ev_text, standalone=standalone, idmap=idmap,
                                         premise_note=premise_note, resolved=resolved)
                try:
                    raw = s.llm.chat(s.system_prompt, user, images=[], max_new_tokens=s.cfg.max_new_tokens)
                except Exception as e2:
                    if "out of memory" not in str(e2).lower():
                        raise
                    _free_cuda()
                    print("[answer] still out of memory - answering from retrieval + fact store only", flush=True)
        obj = parse_json(raw)
        out = finalize(obj, idmap, retr, question, mem, resolved=resolved)
        fixed = user_value_not_in_documents(question, mem, getattr(s, "facts", None) if s.cfg.mode == "final" else None)
        if fixed:
            out = fixed
        mem.add(question, out, retr.named_labels)
        self.last = {
            "question": question, "context_query": ctx, "standalone": standalone, "raw": raw, "result": out,
            "retrieved": [it.cite_key for it in retr.ranked[:10]],
            "evidence": [it.cite_key for it in retr.evidence], "pinned": [p.cite_key for p in retr.pinned],
            "resolved": [r.line() for r in resolved],
            "blocked": [b.cite_key for b in retr.blocked],
            "excluded_in_evidence": any(it.excluded for it in retr.evidence), "seconds": time.time() - t0,
        }
        return out


def _free_cuda():
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def run_questions(system: RAGSystem, df, log_path: str | None = None) -> tuple[list[dict], list[dict]]:
    """Answer a Questions.csv-style DataFrame conversation by conversation."""
    rows, logs = [], []
    progress = os.path.join(system.cfg.work_dir, "progress.txt")
    t_all, done = time.time(), 0
    # Kaggle allows 90 minutes for the whole notebook (P1_T0 = notebook start, set in cell 1): after the limit the
    # remaining questions are answered without the language model (retrieval + fact store), so a valid file is written
    t0 = float(os.environ.get("P1_T0") or t_all)
    system.deadline = t0 + 60 * float(os.environ.get("P1_TIME_LIMIT_MIN", "80"))
    for conv_id, turns in df.groupby("conversation_id", sort=False):
        chat = system.new_conversation(conv_id)
        for _, r in turns.sort_values("turn", key=lambda t: t.astype(int)).iterrows():
            try:
                a = chat.answer(r["question"])
            except Exception as e:  # never crash the submission
                import traceback
                traceback.print_exc()
                a = {"answer": NO_ANSWER, "status": "not_found", "citations": [], "figure_references": []}
                chat.last = {"question": r["question"], "seconds": 0.0, "error": str(e), "retrieved": []}
            rows.append({"id": r["id"], "conversation_id": conv_id, "question_id": r["question_id"], **a})
            logs.append({"id": r["id"], **chat.last})
            done += 1
            print(f"[answer] {done}/{len(df)} {r['question_id']} {chat.last.get('seconds', 0):.0f}s status={a['status']} "
                  f"(elapsed {(time.time() - t0) / 60:.0f} min)", flush=True)
            try:  # watch it with: Get-Content <work_dir>\progress.txt -Wait
                with open(progress, "a", encoding="utf-8") as fh:
                    fh.write(f"{time.strftime('%H:%M:%S')} {done}/{len(df)} {r['question_id']} "
                             f"{chat.last.get('seconds', 0):.1f}s total {time.time() - t_all:.0f}s status={a['status']}\n")
            except OSError:
                pass
    if log_path:
        with open(log_path, "w", encoding="utf-8") as fh:
            json.dump(logs, fh, indent=1, default=str)
    return rows, logs
