"""Guardrails used by the Planner (input) and the Critique (output).

Layered: fast generic rules first (always on, no model needed), then a
vision-language-model check where a model is available. The rules are
category-based, not tied to any test case.
"""
from __future__ import annotations

import os
import re

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

# --- request text ----------------------------------------------------------
UNSAFE_PATTERNS = {
    "sexual": r"\b(nude|nudes|naked|nsfw|porn\w*|sexual\w*|sexy|explicit|undress\w*|strip(ped|ping)?\b|topless|lingerie|erotic\w*|hentai|onlyfans)\b",
    "minors_sexual": r"\b(child|kid|minor|teen|underage|schoolgirl|schoolboy)\b[^.]{0,40}\b(nude|naked|sexy|sexual|undress)",
    "violence_gore": r"\b(gore|gory|blood(y|ied)?|dismember\w*|decapitat\w*|mutilat\w*|corpse|behead\w*|torture\w*|massacre)\b",
    "weapons_harm": r"\b(bomb[- ]making|build (a )?(bomb|weapon|gun)|make (a )?(bomb|explosive)|shoot(ing)? (up|people)|kill (him|her|them|people|someone))\b",
    "hate": r"\b(nazi|swastika|kkk|racial slur|ethnic cleansing|white power)\b",
    "self_harm": r"\b(self[- ]harm|suicide|cutting (my|her|his) (wrist|arm)s?)\b",
    "illegal": r"\b(fake (id|passport|license|banknote|money)|counterfeit|forge(d|ry)? (a )?(signature|document|id))\b",
}
_UNSAFE_RE = {k: re.compile(v, re.I) for k, v in UNSAFE_PATTERNS.items()}

# deceptive edits of real people (misinformation / defamation / impersonation)
DECEPTIVE_EDIT_RE = re.compile(
    r"\b(deep ?fake|face ?swap|swap (his|her|their|the) face|put (his|her|their|the) face|make it look (real|like (he|she|they)))\b"
    r"|\b(make|show|edit|change|photoshop|depict|alter)\b[^.]{0,60}\b(him|her|them|he|she|they|(this|that|the) (man|woman|person|guy|girl|boy|people|politician|president|senator|minister|mayor|governor|ceo|celebrity|actor|actress|singer|athlete|boss|teacher|coworker|neighbou?r|journalist|officer))\b"
    r"[^.]{0,60}\b(arrest\w*|handcuff\w*|steal\w*|rob\w*|bribe\w*|drunk|high on|drugs?|crime|criminal|prison|jail|kissing|naked|nude|bleeding|injured|dead|attack\w*|shoot\w*|hitting|burn\w*|cheat\w*|racist|harass\w*|vandal\w*)",
    re.I)
REAL_PERSON_HINT_RE = re.compile(
    r"(?i:\b(president|prime minister|senator|politician|celebrity|ceo|actor|actress|singer|my (boss|coworker|colleague|neighbou?r|ex|teacher)|"
    r"real person|this person|this man|this woman|photo of (him|her|them|me))\b)")

# the user asks the system to obey text that is in the image
DEFER_TO_IMAGE_RE = re.compile(
    r"\b(follow|do|obey|execute|carry out|apply)\b[^.]{0,40}\b(instruction|instructions|what (it|the image|the text|the note|the sign) says|text|note|sign|written)\b"
    r"|\b(do what|whatever)\b[^.]{0,30}\b(says|is written|tells you)\b", re.I)

# text inside an image that reads like an instruction to an AI / generator
IMAGE_INSTRUCTION_RE = re.compile(
    r"\b(ignore|disregard|forget|override)\b[^.]{0,50}\b(instruction|instructions|user|request|rules|previous|above|prompt)\b"
    r"|\b(ai|assistant|model|system|bot|generator)s?\b[^.]{0,40}\b(must|should|shall|will|have to)\b"
    r"|\b(instead|always)\b[^.]{0,20}\b(draw|generate|create|output|make|render|say|answer|reply|respond)\b"
    r"|^\s*(draw|generate|create|render|output|make|replace|add|remove|say|answer|reply|respond|write|delete)\b[^.]{0,80}"
    r"|\b(instructions?|command|prompt|todo)\s*:",
    re.I | re.M)


def request_safety(text: str) -> tuple[bool, str]:
    """(unsafe?, category) from the request text alone."""
    for cat, rx in _UNSAFE_RE.items():
        if rx.search(text or ""):
            return True, cat
    return False, ""


def is_deceptive_person_edit(text: str) -> bool:
    return bool(DECEPTIVE_EDIT_RE.search(text or ""))


def mentions_real_person(text: str) -> bool:
    return bool(REAL_PERSON_HINT_RE.search(text or ""))


def defers_to_image(text: str) -> bool:
    return bool(DEFER_TO_IMAGE_RE.search(text or ""))


def instruction_like(lines: list[str]) -> list[str]:
    return [l for l in lines or [] if IMAGE_INSTRUCTION_RE.search(l)]


_GENERIC = set("""the a an and or of to in on at for with this that these those it its is are be was were you your user users
image images picture photo note notes text instead ignore disregard forget override instruction instructions must should shall
will have please draw generate create make render output reply respond answer say write add remove replace delete system ai
assistant model bot generator above previous prior request rules prompt only always never any all from into onto than then
there here what which who show shows showing""".split())


def injected_terms(image_text: list[str], user_request: str) -> list[str]:
    """Content words from instruction-like text inside the image that the user did not type themselves."""
    req = set(re.findall(r"[a-z]+", (user_request or "").lower()))
    terms = []
    sentences = []
    for t in image_text or []:
        joined = re.sub(r"\s*\n\s*", " ", str(t))  # a note's lines form sentences across line breaks
        sentences += [x for x in re.split(r"(?<=[.!?])\s+|\s+-\s+", joined) if x.strip()]
    for line in instruction_like(sentences):
        for w in re.findall(r"[a-z]+", line.lower()):
            if len(w) >= 3 and w not in _GENERIC and w not in req and w not in terms:
                terms.append(w)
    return terms


def remove_terms(text: str, terms: list[str]) -> tuple[str, list[str]]:
    removed = []
    out = text or ""
    for t in terms:
        if re.search(rf"\b{re.escape(t)}s?\b", out, re.I):
            out = re.sub(rf"\b{re.escape(t)}s?\b", "", out, flags=re.I)
            removed.append(t)
    out = re.sub(r"\b(an?|the)\s+(?=[,.;]|$)", "", out)
    out = re.sub(r"\s{2,}", " ", re.sub(r"\s+,", ",", out)).strip(" ,.;")
    return out, removed


def strip_image_text(prompt: str, image_text: list[str], user_request: str) -> tuple[str, list[str]]:
    """Remove any text that came from inside the image unless the user typed it themselves."""
    removed = []
    out = prompt or ""
    for t in image_text or []:
        t = t.strip().strip('"')
        if len(t) < 4 or t.lower() in (user_request or "").lower():
            continue
        if t.lower() in out.lower():
            out = re.sub(re.escape(t), "", out, flags=re.I)
            removed.append(t)
        # also remove long fragments of instruction-like image text
        for frag in re.split(r"[.;:!?\n]", t):
            frag = frag.strip()
            if len(frag.split()) >= 3 and frag.lower() in out.lower():
                out = re.sub(re.escape(frag), "", out, flags=re.I)
                removed.append(frag)
    out = re.sub(r"\s{2,}", " ", out).strip(" ,.;")
    return out, removed


# --- input image checks ----------------------------------------------------
SUPPORTED = {"PNG", "JPEG", "WEBP", "BMP", "GIF", "TIFF", "MPO"}


def check_image(path: str, min_side: int = 64):
    """Validate an uploaded image; never raises. Returns ImageCheck-like dict."""
    from .state import ImageCheck
    if not path:
        return ImageCheck(ok=False, issue="missing")
    if not os.path.isfile(path):
        return ImageCheck(ok=False, issue="missing")
    if os.path.getsize(path) == 0:
        return ImageCheck(ok=False, issue="empty")
    try:
        with Image.open(path) as im:
            fmt = im.format
            im.verify()
        with Image.open(path) as im:
            im.load()
            im = ImageOps.exif_transpose(im).convert("RGB")
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as e:
        return ImageCheck(ok=False, issue="broken", warning=str(e)[:200])
    if fmt not in SUPPORTED:
        return ImageCheck(ok=False, issue="unsupported_format", warning=str(fmt))
    w, h = im.size
    g = np.asarray(im.convert("L"), dtype=np.float32)
    lap = g[1:-1, 1:-1] * -4 + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:] if min(w, h) > 2 else np.zeros(1)
    blur = float(lap.var())
    bright = float(g.mean())
    if min(w, h) < min_side:
        return ImageCheck(ok=False, issue="too_small", width=w, height=h, blur=blur, brightness=bright)
    warn = []
    if blur < 15:
        warn.append("image is very blurry or flat")
    if bright < 20:
        warn.append("image is very dark")
    if bright > 245:
        warn.append("image is almost white")
    if float(g.std()) < 3:
        warn.append("image has almost no content")
    return ImageCheck(ok=True, issue="low_quality" if warn else "", width=w, height=h, blur=blur,
                      brightness=bright, warning="; ".join(warn))


def load_rgb(path: str, max_side: int = 768, multiple: int = 8) -> Image.Image:
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    w, h = im.size
    s = min(1.0, max_side / max(w, h))
    w2, h2 = max(multiple, int(w * s) // multiple * multiple), max(multiple, int(h * s) // multiple * multiple)
    return im.resize((w2, h2), Image.LANCZOS) if (w2, h2) != (w, h) else im
