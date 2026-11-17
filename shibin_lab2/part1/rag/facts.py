"""Structured source facts and deterministic precedence (Task 1.2.3, answering rules 1-2).

The language model is good at wording an answer but unreliable at deciding *which* of several printed
values is current. This module makes that decision in code, from the knowledge base itself:

* every table row becomes facts (entity, measure, period, value, kind, document, trust, cite key);
  a value is a *forecast* when its column is marked (``2011*``, ``forecast``), when it is the bracketed
  forecast next to an outturn, or when its year is not before the document's data vintage;
* explicit revision notices in the text ("... for 2010 has been revised from 3.0% ... to 2.4%") become
  *revised* facts that supersede the earlier value;
* short level statements in the text ("X's output growth slowed to 7.8% in 2012") become *text* facts,
  so a briefing that disagrees with a table is noticed.

``resolve`` then applies the manifest's precedence: revision > outturn > forecast; two high/medium-trust
sources that still disagree are a ``conflict`` and the higher-trust value is preferred. Nothing here knows
any question: entities and measures are generic vocabulary matched against the documents' own labels.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .ingest import Item

ENTITIES = {
    "United States": r"\bu\.\s?s\.|\bunited states\b|\bamerican?\b|\bus\b|\busa\b",
    "Euro area": r"\beuro[- ]?(area|zone)\b|\beurozone\b",
    "China": r"\bchina\b|\bchinese\b",
    "World": r"\bworld\b|\bglobal\b",
    "Advanced economies": r"\badvanced\b",
    "Emerging and developing economies": r"\bemerging\b",
}
MEASURES = {
    "output growth": r"output growth|\bgdp growth|growth rate|\bgrowth\b|\bgrew\b|\bgrow\b|\boutput\b|\bgdp\b",
    "unemployment rate": r"unemploy",
    "inflation rate": r"inflation|deflator",
    "revenues": r"revenue",
    "spending": r"spending|expenditure|outlays",
    "budget balance": r"budget balance|deficit|surplus",
    "labour force participation rate": r"participation",
}
TRUST_RANK = {"high": 3, "medium": 2, "low": 0}
YEAR_RE = re.compile(r"\b(19[5-9]\d|20[0-4]\d)\b")
NUM = r"[-−–]?\d+(?:\.\d+)?"


def _f(s: str) -> float:
    return float(s.replace("−", "-").replace("–", "-"))


def match_entities(text: str) -> list[str]:
    t = text.lower()
    return [e for e, rx in ENTITIES.items() if re.search(rx, t)]


def match_measures(text: str) -> list[str]:
    t = text.lower()
    found = [m for m, rx in MEASURES.items() if re.search(rx, t)]
    # "unemployment growth" / "inflation rate growth" are not output growth
    if "output growth" in found and len(found) > 1 and not re.search(r"output|gdp|econom|grew|grow\b", t):
        found.remove("output growth")
    return found


def _last_match(text: str, vocab: dict) -> str | None:
    """The vocabulary entry whose last occurrence in `text` is closest to its end."""
    best, pos = None, -1
    for name, rx in vocab.items():
        for m in re.finditer(rx, text):
            if m.start() > pos:
                best, pos = name, m.start()
    return best


@dataclass
class Fact:
    entity: str
    measure: str
    period: str          # "2010" or "2000-2007"
    value: float
    kind: str            # actual | forecast | revised | text
    doc_id: str
    trust: str
    cite: str            # DOCx:Table A-2 or DOCx:pN
    page: int
    item_id: str
    note: str = ""

    def show(self) -> str:
        tag = {"forecast": " (forecast)", "revised": " (revised)", "text": " (text)"}.get(self.kind, "")
        return f"{self.value:g}{tag} [{self.cite}, trust {self.trust}]"


@dataclass
class Resolution:
    key: tuple
    status: str                          # answered | conflict
    value: float | None
    preferred: Fact | None
    facts: list = field(default_factory=list)
    reason: str = ""

    def line(self) -> str:
        ent, mea, per = self.key
        return f"{ent} | {mea} | {per}: {self.reason}"


def _vintage_year(v: str) -> int | None:
    m = YEAR_RE.search(v or "")
    return int(m.group(1)) if m else None


class FactStore:
    def __init__(self, items: list[Item], vintages: dict | None = None):
        self.vintages = {d: _vintage_year(v) for d, v in (vintages or {}).items()}
        self.items = {it.id: it for it in items}
        self.facts: list[Fact] = []
        self.revisions: list[Fact] = []
        for it in items:
            if it.excluded or TRUST_RANK.get(it.trust_level, 0) == 0:
                continue  # low-trust content is never evidence
            if it.kind == "table" and it.table_text:
                self.facts += self._table_facts(it)
            elif it.kind == "text":
                self.revisions += self._revisions(it)
                self.facts += self._text_facts(it)

    # ------------------------------------------------------------------ extraction
    def _forecast_year(self, doc: str, year: int) -> bool:
        v = self.vintages.get(doc)
        return v is not None and year >= v

    def _table_facts(self, it: Item) -> list[Fact]:
        out = []
        caption_ents = match_entities(it.caption or "")
        first_col = re.search(r"\(first column: ([^)]*)\)", it.table_text)
        header_ents = match_entities(first_col.group(1)) if first_col else []
        for line in it.table_text.splitlines():
            if ":" not in line or line.startswith("(first column"):
                continue
            label, cells = line.split(":", 1)
            row_ents, row_meas = match_entities(label), match_measures(label)
            for cell in cells.split(";"):
                if "=" not in cell:
                    continue
                col, val = (s.strip() for s in cell.split("=", 1))
                col_meas = match_measures(col)
                ents = row_ents or header_ents or caption_ents
                meas = row_meas or col_meas or match_measures(it.caption or "")
                years = YEAR_RE.findall(col) or YEAR_RE.findall(label)
                if len(ents) != 1 or len(meas) != 1 or not years:
                    continue
                if re.search(r"slope|intercept|needed", col + label, re.I):
                    continue  # regression estimates, not a time series
                period = "-".join(years[:2]) if re.search(r"(19|20)\d\d\s*-\s*(19|20)\d\d", col + " " + label) else years[0]
                m = re.fullmatch(rf"({NUM})\s*(?:\(({NUM})\))?", val)
                if not m:
                    continue
                marked = "*" in col or "forecast" in col.lower()
                single = "-" not in period
                kind = "forecast" if marked or (single and self._forecast_year(it.doc_id, int(period))) else "actual"
                base = dict(entity=ents[0], measure=meas[0], period=period, doc_id=it.doc_id, trust=it.trust_level,
                            cite=it.cite_key, page=it.page, item_id=it.id)
                out.append(Fact(value=_f(m.group(1)), kind=kind, **base))
                if m.group(2):  # "outturn (forecast)" cell
                    out.append(Fact(value=_f(m.group(2)), kind="forecast", note="bracketed forecast", **base))
        return out

    REV_RE = re.compile(
        rf"(?P<subj>[A-Z][\w.\s]{{1,40}}?)\s+(?P<meas>output growth|unemployment(?: rate)?|inflation(?: rate)?)\s+for\s+"
        rf"(?P<year>(?:19|20)\d\d)\s+has been revised from\s+(?P<old>{NUM})\s*%.{{0,80}}?\bto\s+(?P<new>{NUM})\s*%",
        re.S)

    def _revisions(self, it: Item) -> list[Fact]:
        out = []
        for m in self.REV_RE.finditer(it.text or ""):
            ents, meas = match_entities(m.group("subj")), match_measures(m.group("meas"))
            if len(ents) == 1 and len(meas) == 1:
                out.append(Fact(entity=ents[0], measure=meas[0], period=m.group("year"), value=_f(m.group("new")),
                                kind="revised", doc_id=it.doc_id, trust=it.trust_level, cite=it.cite_key, page=it.page,
                                item_id=it.id, note=f"revised from {_f(m.group('old')):g}"))
        return out

    # a level stated for an explicit year: "... slowed to 7.8% in 2012"
    LEVEL_RE = re.compile(rf"\b(?:was|were|stood at|slowed to|fell to|rose to|increased to|decreased to|reached)\s+"
                          rf"({NUM})\s*%\s+in\s+((?:19|20)\d\d)\b")

    def _text_facts(self, it: Item) -> list[Fact]:
        """Single-value level statements: one entity, one measure and one percentage for an explicit year."""
        out = []
        for sent in re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", it.text or "")):
            pct = re.findall(rf"{NUM}\s*%", sent)
            lv = self.LEVEL_RE.search(sent)
            if len(pct) != 1 or not lv or re.search(r"\bby\s+" + NUM, sent):
                continue
            if re.search(r"forecast|expected|needed|average|assum", sent, re.I):
                continue
            # the subject is the entity and measure written just before the verb (a merged caption or an
            # earlier clause may mention other measures)
            prefix = sent[:lv.start()][-60:].lower()
            ent = _last_match(prefix, ENTITIES)
            mea = _last_match(prefix, MEASURES)
            if ent and mea:
                out.append(Fact(entity=ent, measure=mea, period=lv.group(2), value=_f(lv.group(1)), kind="text",
                                doc_id=it.doc_id, trust=it.trust_level, cite=it.cite_key, page=it.page, item_id=it.id))
        return out

    # ------------------------------------------------------------------ resolution
    def resolve(self, entity: str, measure: str, period: str) -> Resolution | None:
        facts = [f for f in self.facts if (f.entity, f.measure, f.period) == (entity, measure, period)]
        revs = [f for f in self.revisions if (f.entity, f.measure, f.period) == (entity, measure, period)]
        if not facts and not revs:
            return None
        key = (entity, measure, period)
        if revs:
            r = revs[0]
            return Resolution(key, "answered", r.value, r, facts + revs,
                              f"use {r.value:g} - explicit revision in {r.cite} ({r.note}); it supersedes "
                              + ", ".join(f.show() for f in facts if abs(f.value - r.value) > 0.05) or f"use {r.value:g}")
        actual = [f for f in facts if f.kind in ("actual", "text")]
        forecasts = [f for f in facts if f.kind == "forecast"]
        if not actual:
            best = max(forecasts, key=lambda f: TRUST_RANK.get(f.trust, 0))
            return Resolution(key, "answered", best.value, best, facts,
                              f"only a forecast is reported: {best.show()} - say it is a forecast")
        values = {}
        for f in actual:
            values.setdefault(round(f.value, 2), []).append(f)
        if len(values) > 1 and len({f.doc_id for f in actual}) > 1:
            ranked = sorted(actual, key=lambda f: (-TRUST_RANK.get(f.trust, 0), f.kind == "text"))
            pref = ranked[0]
            return Resolution(key, "conflict", pref.value, pref, facts,
                              "CONFLICT (no revision note): " + " vs ".join(
                                  f.show() for f in sorted({round(f.value, 2): f for f in ranked}.values(),
                                                           key=lambda f: -TRUST_RANK.get(f.trust, 0)))
                              + f" - status conflict, give both values and prefer {pref.value:g} (higher trust)")
        pref = sorted(actual, key=lambda f: (-TRUST_RANK.get(f.trust, 0), f.kind == "text"))[0]
        old = [f for f in forecasts if abs(f.value - pref.value) > 0.05]
        reason = f"{pref.value:g} [{pref.cite}]"
        if old:
            reason += " - outturn; it supersedes the forecast " + ", ".join(f.show() for f in old)
        return Resolution(key, "answered", pref.value, pref, facts, reason)

    def lookup(self, question: str, max_keys: int = 8) -> list[Resolution]:
        """Resolutions for every (entity, measure, year) the question names that the documents report."""
        ents, meas = match_entities(question), match_measures(question)
        years = list(dict.fromkeys(YEAR_RE.findall(question)))
        rng = re.search(r"\b((?:19|20)\d\d)\s*(?:-|–|to|through|and)\s*((?:19|20)\d\d)\b", question)
        if rng and re.search(r"between|from|through|over|-|–", question):
            a, b = int(rng.group(1)), int(rng.group(2))
            if 0 < b - a <= 6:
                years = list(dict.fromkeys(years + [str(y) for y in range(a, b + 1)]))
        after = re.search(r"\b(after|since|following)\b[^.?]{0,40}?\b((?:19|20)\d\d)\b", question, re.I)
        if after:  # "after the 2008 crisis": the years that follow, as far as the documents go
            a = int(after.group(2))
            years = list(dict.fromkeys(years + [str(y) for y in range(a, a + 6)]))
        if not years and ents and meas:  # a series question ("yearly", "before and during"): every period reported
            years = sorted({f.period for f in self.facts if f.entity in ents and f.measure in meas})
        out = []
        for y in years:  # period-major: two entities in one question get the same periods
            for e in ents:
                for m in meas:
                    r = self.resolve(e, m, y)
                    if r:
                        out.append(r)
        return out[:max_keys]

    def items_for(self, res: list[Resolution]) -> list[Item]:
        seen, out = set(), []
        for r in res:
            for f in r.facts:
                if f.item_id not in seen and f.item_id in self.items:
                    seen.add(f.item_id)
                    out.append(self.items[f.item_id])
        return out
