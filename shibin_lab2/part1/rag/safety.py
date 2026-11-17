"""Generic prompt-injection and untrusted-content handling.

The knowledge base is treated as *data*. Any sentence that looks like an
instruction addressed to an AI system is flagged and removed from the evidence
text before it reaches the generator. The patterns are generic (they do not
mention any question, value or document), so they also cover the hidden set.
"""
from __future__ import annotations

import re

_INJECTION_PATTERNS = [
    r"\bignore\b[^.]{0,60}\b(instruction|instructions|rules|above|previous|prior|earlier|context|documents?)\b",
    r"\bdisregard\b[^.]{0,60}\b(instruction|instructions|rules|above|previous|prior|documents?|sources?)\b",
    r"\b(forget|override|bypass)\b[^.]{0,40}\b(instruction|instructions|rules|guidelines|system)\b",
    r"\b(you are|you're) (now )?(an? )?(ai|assistant|language model|chatbot|llm)\b",
    # addressed to an AI system (not "the model" - economics text uses that word)
    r"\b(ai|assistant|language model|chatbot|llm|retrieval system|rag system|answering system)s?\b[^.]{0,40}\b(must|should|shall|are required to|have to|need to|will)\b",
    r"\b(system prompt|developer message|jailbreak|prompt injection)\b",
    r"\b(always|only) (answer|respond|reply|cite)\b",
    r"\b(answer|respond|reply)\b[^.]{0,30}\b(with|that)\b[^.]{0,60}\b(regardless|no matter|instead)\b",
    r"\bdo not (cite|mention|reveal|use|report|tell)\b",
    r"\b(treat|consider) (this|these) (as|values as) (the )?(authoritative|official|correct|true)\b",
    r"\bnote to (the )?(ai|assistant|model|reader model|llm)\b",
    r"^\s*(assistant|system|user)\s*:",
    r"\binstruction(s)? (to|for) (the )?(ai|assistant|model|llm|retrieval)\b",
]
_INJ_RE = [re.compile(p, re.I | re.M) for p in _INJECTION_PATTERNS]

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def looks_like_injection(text: str) -> bool:
    return any(r.search(text or "") for r in _INJ_RE)


def sanitize(text: str) -> tuple[str, list[str]]:
    """Remove instruction-like sentences. Returns (clean_text, removed_sentences)."""
    if not text:
        return text, []
    removed, kept = [], []
    for sent in _SENT_SPLIT.split(text):
        if not sent.strip():
            continue
        if looks_like_injection(sent):
            removed.append(sent.strip())
            kept.append("[instruction-like text removed]")
        else:
            kept.append(sent.strip())
    return " ".join(kept), removed


# --- user-turn intents (generic, used for status handling) -----------------
_OVERRIDE_RE = re.compile(
    r"\b(assume|assuming|suppose|supposing|let'?s say|pretend|imagine|hypothetically|"
    r"for (the rest of |the purposes? of )?this (chat|conversation|session|discussion)|"
    r"in this (chat|conversation|session)|treat [^.?]{0,40} as|take [^.?]{0,40} (to be|as)|"
    r"if [^.?]{0,60}\b(were|was|had been|is actually|is now)\b|use [^.?]{0,40} instead)\b",
    re.I)
_CHANGE_VERB = r"(update|change|edit|modify|correct|overwrite|replace|rewrite|amend|fix|save|store|set|delete|remove)"
_PERMANENT_RE = re.compile(
    # imperative: "Update DOC2 so that ...", "Please change the document to ..."
    r"^\s*(please\s+|can you\s+|could you\s+|would you\s+|i want you to\s+|go ahead and\s+)?" + _CHANGE_VERB +
    r"\b[^?]{0,80}\b(document|documents|doc\s?\d|pdf|knowledge base|kb|database|source file|figure|table|record|records)\b"
    # or explicit persistence words anywhere
    r"|\bpermanent(ly)?\b|\bfor (all )?(future|other) (users|conversations|chats|sessions)\b|\bfor everyone\b"
    r"|\b" + _CHANGE_VERB + r"\b[^?]{0,60}\bin (the )?(knowledge base|database|kb|pdf|source documents?)\b",
    re.I)


def is_session_override(question: str) -> bool:
    return bool(_OVERRIDE_RE.search(question or ""))


def is_permanent_change_request(question: str) -> bool:
    return bool(_PERMANENT_RE.search(question or ""))


_APPENDIX_RE = re.compile(r"\bappendix\s+([A-Z])\b", re.I)


def mentioned_appendices(question: str) -> set[str]:
    return {f"Appendix {m.upper()}" for m in _APPENDIX_RE.findall(question or "")}
