"""Task 1.3 - per-conversation memory.

One ConversationMemory per conversation_id. It holds the earlier turns and any
values the user supplied ("assume ...", "for this chat ..."). It lives only in
Python memory: nothing here is ever written to the vector database, and it is
discarded when the conversation ends (a new conversation starts empty).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .safety import is_permanent_change_request, is_session_override

_FOLLOWUP_RE = re.compile(
    r"^\s*(and|what about|how about|and what about|what of|in|for|then|same|also|now|compare|versus|vs\.?|"
    r"by how much|how much|why|is that|was that|does that|did that|so)\b|\b(it|that|this|those|these|they|them|its|same)\b",
    re.I)


@dataclass
class Turn:
    question: str
    answer: dict
    labels: list = field(default_factory=list)


@dataclass
class ConversationMemory:
    conversation_id: str = ""
    turns: list = field(default_factory=list)
    user_values: list = field(default_factory=list)  # verbatim user statements that set/change a value

    def is_followup(self, question: str) -> bool:
        return bool(self.turns) and (len(question.split()) <= 12 or bool(_FOLLOWUP_RE.search(question)))

    def context_query(self, question: str) -> str:
        """Earlier wording used as an extra retrieval query for follow-ups."""
        if not self.turns or not self.is_followup(question):
            return ""
        return " ".join(t.question for t in self.turns[-2:])

    def carry_labels(self, question: str) -> list:
        if not self.turns or not self.is_followup(question):
            return []
        for t in reversed(self.turns):
            if t.labels:
                return list(t.labels)
        return []

    def note_user_value(self, question: str) -> bool:
        if is_session_override(question) or is_permanent_change_request(question):
            self.user_values.append(question.strip())
            return True
        return False

    def add(self, question: str, answer: dict, labels: list):
        self.turns.append(Turn(question, answer, list(labels)))

    def history_text(self, n: int = 3) -> str:
        lines = []
        for t in self.turns[-n:]:
            lines.append(f"User: {t.question}")
            lines.append(f"Assistant ({t.answer.get('status', '')}): {t.answer.get('answer', '')}")
        return "\n".join(lines)

    def clear(self):
        self.turns.clear()
        self.user_values.clear()
