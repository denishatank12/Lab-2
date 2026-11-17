"""Task 2.1 - data preparation: load, remove corrupted and duplicate images, split 90/10 (seed 266), resize.

Duplicates: exact duplicates (same pixels after decoding) and near-duplicates (64-bit difference hash within
`dup_hamming` bits). Of each duplicate group the first image is kept.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageOps

from .config import P2Config


@dataclass
class CleanReport:
    total: int = 0
    corrupted: list = field(default_factory=list)  # (index, reason)
    exact_duplicates: list = field(default_factory=list)  # (index, kept_index)
    near_duplicates: list = field(default_factory=list)  # (index, kept_index, hamming)
    kept: list = field(default_factory=list)

    def summary(self) -> dict:
        return {"total": self.total, "corrupted_removed": len(self.corrupted),
                "exact_duplicates_removed": len(self.exact_duplicates),
                "near_duplicates_removed": len(self.near_duplicates), "kept": len(self.kept)}


def dhash(img: Image.Image, size: int = 8) -> int:
    g = np.asarray(img.convert("L").resize((size + 1, size), Image.LANCZOS), dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def _decode(x) -> Image.Image:
    if isinstance(x, Image.Image):
        x.load()
        return x
    if isinstance(x, dict) and x.get("bytes") is not None:
        x = x["bytes"]
    if isinstance(x, (bytes, bytearray)):
        im = Image.open(io.BytesIO(x))
        im.verify()  # structural check
        im = Image.open(io.BytesIO(x))
        im.load()
        return im
    raise ValueError(f"unsupported image value {type(x)}")


def clean(images, captions, cfg: P2Config, log=print) -> CleanReport:
    """images: iterable of PIL images / raw bytes; returns which indices to keep and why others were removed."""
    rep = CleanReport()
    seen_md5, hashes = {}, []
    for i, (raw, cap) in enumerate(zip(images, captions)):
        rep.total += 1
        try:
            im = _decode(raw)
            im = ImageOps.exif_transpose(im).convert("RGB")
        except Exception as e:  # cannot be decoded
            rep.corrupted.append((i, f"undecodable: {type(e).__name__}"))
            continue
        if min(im.size) < cfg.min_side:
            rep.corrupted.append((i, f"too small {im.size}"))
            continue
        arr = np.asarray(im)
        if arr.std() < 2:
            rep.corrupted.append((i, "blank / single colour"))
            continue
        if not str(cap or "").strip():
            rep.corrupted.append((i, "empty caption"))
            continue
        md5 = hashlib.md5(arr.tobytes()).hexdigest()
        if md5 in seen_md5:
            rep.exact_duplicates.append((i, seen_md5[md5]))
            continue
        seen_md5[md5] = i
        h = dhash(im)
        near = None
        for j, hj in hashes:
            d = bin(h ^ hj).count("1")
            if d <= cfg.dup_hamming:
                near = (j, d)
                break
        if near:
            rep.near_duplicates.append((i, near[0], near[1]))
            continue
        hashes.append((i, h))
        rep.kept.append(i)
    log(f"[data] {rep.summary()}")
    return rep


def split(kept: list[int], cfg: P2Config) -> tuple[list[int], list[int]]:
    """90/10 split of the cleaned indices with seed 266 (test images are used only for evaluation)."""
    rng = np.random.default_rng(cfg.seed)
    order = rng.permutation(len(kept))
    n_test = int(round(cfg.test_fraction * len(kept)))
    test = sorted(kept[k] for k in order[:n_test])
    train = sorted(kept[k] for k in order[n_test:])
    return train, test


def to_square(im: Image.Image, size: int) -> Image.Image:
    """Resize the short side to `size` and centre-crop (the model's 512x512 resolution)."""
    im = im.convert("RGB")
    w, h = im.size
    s = size / min(w, h)
    im = im.resize((max(size, round(w * s)), max(size, round(h * s))), Image.BICUBIC)
    w, h = im.size
    l, t = (w - size) // 2, (h - size) // 2
    return im.crop((l, t, l + size, t + size))


def prepare(cfg: P2Config, log=print) -> dict:
    """Load the dataset, clean, split, resize and write <work>/data/{train,test}/*.png + captions.jsonl."""
    from datasets import load_dataset
    ds = load_dataset(cfg.dataset, split="train")
    ds_raw = ds.cast_column("image", ds.features["image"].__class__(decode=False))
    caps = ds["text"]
    rep = clean((r["image"] for r in ds_raw), caps, cfg, log)
    train, test = split(rep.kept, cfg)
    out = {}
    for name, idx in (("train", train), ("test", test)):
        d = os.path.join(cfg.work_dir, "data", name)
        os.makedirs(d, exist_ok=True)
        rows = []
        for i in idx:
            im = to_square(_decode(ds_raw[i]["image"]), cfg.resolution)
            fn = f"{i:05d}.png"
            im.save(os.path.join(d, fn))
            rows.append({"file": fn, "index": i, "text": caps[i]})
        with open(os.path.join(d, "captions.jsonl"), "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        out[name] = rows
    out["report"] = rep
    log(f"[data] train={len(train)} test={len(test)} written to {os.path.join(cfg.work_dir, 'data')}")
    return out


def describe(rows: list[dict]) -> dict:
    caps = [r["text"] for r in rows]
    words = [len(c.split()) for c in caps]
    first = {}
    for c in caps:
        k = " ".join(c.split()[:2])
        first[k] = first.get(k, 0) + 1
    return {"n": len(rows), "caption_words_mean": float(np.mean(words)), "caption_words_min": int(np.min(words)),
            "caption_words_max": int(np.max(words)),
            "most_common_openings": sorted(first.items(), key=lambda x: -x[1])[:8]}


def load_split(cfg: P2Config, name: str) -> list[dict]:
    d = os.path.join(cfg.work_dir, "data", name)
    with open(os.path.join(d, "captions.jsonl"), encoding="utf-8") as fh:
        rows = [json.loads(l) for l in fh]
    for r in rows:
        r["path"] = os.path.join(d, r["file"])
    return rows
