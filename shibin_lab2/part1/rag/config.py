"""Configuration for the Part 1 multimodal RAG system.

Every knob lives here so the notebook, the tests and the Kaggle run share one
definition. Values can be overridden with environment variables (useful on
Kaggle, e.g. VLM_MODEL=Qwen/Qwen2.5-VL-7B-Instruct) or by passing a Config.
"""
from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass, field

SEED = 266

# File names given in the assignment (Canvas / Kaggle Data tab).
DOC_FILES = {
    "DOC1": "Document.pdf",
    "DOC2": "Document2_Supplement_A_Outturns_2011-2013.pdf",
    "DOC3": "Document3_Briefing_B_Labour_Prices_Policy.pdf",
}
MANIFEST_FILE = "kb_manifest.json"

# Fallback trust levels (from the assignment text) used only when
# kb_manifest.json does not state them.
DEFAULT_TRUST = {"DOC1": "high", "DOC2": "high", "DOC3": "medium"}
# Sections that must never be used as evidence (answering rule 3).
UNTRUSTED_SECTIONS = {("DOC3", "Appendix C")}


def _env(name: str, default):
    v = os.environ.get(name)
    if v is None:
        return default
    if isinstance(default, bool):
        return v.strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(default, int):
        return int(v)
    if isinstance(default, float):
        return float(v)
    return v


def find_data_dir(start_dirs=None) -> str:
    """Locate the folder that holds Document.pdf.

    Search order: $DATA_DIR, the competition mount on Kaggle (/kaggle/input/*),
    ./data, ../data and the current directory.
    """
    cands = []
    if os.environ.get("DATA_DIR"):
        cands.append(os.environ["DATA_DIR"])
    cands += start_dirs or []
    cands += sorted(glob.glob("/kaggle/input/*")) + sorted(glob.glob("/kaggle/input/*/*")) \
        + sorted(glob.glob("/kaggle/input/*/*/*"))
    here = os.path.dirname(os.path.abspath(__file__))
    cands += ["data", "../data", os.path.join(here, "..", "..", "data"), "."]
    for d in cands:
        if _manifest_ok(d) or os.path.isfile(os.path.join(d, DOC_FILES["DOC1"])):
            return os.path.abspath(d)
    # last resort: recursive search for the manifest (it names the PDFs), then for Document.pdf
    for root in ["/kaggle/input", "."]:
        for name in (MANIFEST_FILE, DOC_FILES["DOC1"]):
            hits = sorted(glob.glob(os.path.join(root, "**", name), recursive=True))
            if hits:
                return os.path.abspath(os.path.dirname(hits[0]))
    seen = sorted(glob.glob("/kaggle/input/**/*", recursive=True))[:40]
    raise FileNotFoundError(
        "Could not find the competition data (kb_manifest.json + PDFs). Set DATA_DIR or put the files in ./data. "
        f"Files under /kaggle/input: {seen}")


def _manifest_ok(d: str) -> bool:
    """A folder holding kb_manifest.json and every PDF it lists."""
    p = os.path.join(d, MANIFEST_FILE)
    if not os.path.isfile(p):
        return False
    try:
        with open(p, encoding="utf-8") as fh:
            files = [x.get("file") for x in json.load(fh).get("documents", []) if isinstance(x, dict)]
    except Exception:
        return True
    return all(os.path.isfile(os.path.join(d, f)) for f in files if f)


@dataclass
class Config:
    data_dir: str = ""
    work_dir: str = field(default_factory=lambda: _env("WORK_DIR", "artifacts/p1"))

    # --- models -----------------------------------------------------------
    embed_model: str = field(default_factory=lambda: _env("EMBED_MODEL", "BAAI/bge-small-en-v1.5"))
    reranker_model: str = field(default_factory=lambda: _env("RERANK_MODEL", "BAAI/bge-reranker-base"))
    # Vision-language model: describes figures at ingestion and writes answers.
    # 7B in 4-bit fits one 16 GB GPU with room to spare; VLM_MODEL=Qwen/Qwen2.5-VL-3B-Instruct LOAD_IN_4BIT=0 is the light option.
    vlm_model: str = field(default_factory=lambda: _env("VLM_MODEL", "Qwen/Qwen3-VL-8B-Instruct"))
    # "two_stage" (default): the VLM describes every figure, is unloaded, and a stronger text model answers.
    # "qwen_vl": one VLM for both. "hf_text": small text model (CPU tests). "none": ingestion only.
    llm_backend: str = field(default_factory=lambda: _env("LLM_BACKEND", "two_stage"))
    answer_model: str = field(default_factory=lambda: _env("ANSWER_MODEL", "Qwen/Qwen3-8B"))
    text_llm_model: str = field(default_factory=lambda: _env("TEXT_LLM_MODEL", "Qwen/Qwen2.5-0.5B-Instruct"))
    # 4-bit NF4 on GPU (~6 GB for the 7B VLM); ignored on CPU
    load_in_4bit: bool = field(default_factory=lambda: _env("LOAD_IN_4BIT", True))
    max_new_tokens: int = field(default_factory=lambda: _env("MAX_NEW_TOKENS", 380))
    describe_figures: bool = field(default_factory=lambda: _env("DESCRIBE_FIGURES", True))
    # Attach the image of an explicitly named figure when answering.
    attach_named_figure: bool = field(default_factory=lambda: _env("ATTACH_FIGURE", True))

    # --- chunking ---------------------------------------------------------
    chunk_words: int = 160
    chunk_overlap: int = 40
    figure_zoom: float = 2.0  # render scale for figure crops (2.0 = 144 dpi)

    # --- retrieval --------------------------------------------------------
    mode: str = "final"  # "baseline" = dense only; "final" = hybrid + figure lookup
    top_k: int = 10  # evidence items passed to the generator
    per_doc_min: int = 2  # best items of every document that are always included
    follow_refs: int = 3  # tables/figures cited by the retrieved text that are added to the evidence
    candidates: int = 30  # per-retriever candidate pool before fusion
    rrf_k: int = 60
    use_reranker: bool = field(default_factory=lambda: _env("USE_RERANKER", False))
    # total characters of evidence sent to the answer model (~2.5k tokens): keeps a 16 GB T4 well inside memory
    evidence_chars: int = field(default_factory=lambda: _env("EVIDENCE_CHARS", 9000))
    max_prompt_tokens: int = field(default_factory=lambda: _env("MAX_PROMPT_TOKENS", 3500))
    rerank_pool: int = 20
    history_turns: int = 3  # earlier turns of the same conversation given to the generator

    seed: int = SEED

    def resolved_data_dir(self) -> str:
        if not self.data_dir:
            self.data_dir = find_data_dir()
        return self.data_dir

    def variant(self, **kw) -> "Config":
        d = dict(self.__dict__)
        d.update(kw)
        return Config(**d)
