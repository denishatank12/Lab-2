"""Task 2.4 - evaluation of the base model and both fine-tuning runs.

  primary    KID (in-domain generations vs test images; noisy with 20 images) and CLIP score (prompt vs image)
  secondary  diversity: LPIPS distance between the two seeds of each prompt
  human      blinded pairs (final run vs base, left/right randomised), >= 3 raters -> win rate
  efficiency GPU-hours per training run and peak GPU memory (from train_stats.json)
  safety     NSFW flags from the SD safety checker; memorization check (CLIP image similarity to training images)
"""
from __future__ import annotations

import csv
import os
import random

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont

from .config import P2Config

CLIP_ID = "openai/clip-vit-large-patch14"


def _dev():
    return "cuda" if torch.cuda.is_available() else "cpu"


def _u8(paths, size=299):
    arr = [np.asarray(Image.open(p).convert("RGB").resize((size, size), Image.BICUBIC)) for p in paths]
    return torch.from_numpy(np.stack(arr)).permute(0, 3, 1, 2).contiguous()


def kid(gen_paths: list[str], real_paths: list[str]) -> tuple[float, float]:
    from torchmetrics.image.kid import KernelInceptionDistance
    sub = min(len(gen_paths), len(real_paths), 50)
    m = KernelInceptionDistance(subset_size=sub, subsets=50).to(_dev())
    for i in range(0, len(real_paths), 32):
        m.update(_u8(real_paths[i:i + 32]).to(_dev()), real=True)
    for i in range(0, len(gen_paths), 32):
        m.update(_u8(gen_paths[i:i + 32]).to(_dev()), real=False)
    mean, std = m.compute()
    return float(mean), float(std)


class Clip:
    def __init__(self, model_id=CLIP_ID):
        from transformers import CLIPModel, CLIPProcessor
        self.model = CLIPModel.from_pretrained(model_id).to(_dev()).eval()
        self.proc = CLIPProcessor.from_pretrained(model_id)

    @torch.inference_mode()
    def image_emb(self, paths, bs=32):
        out = []
        for i in range(0, len(paths), bs):
            ims = [Image.open(p).convert("RGB") for p in paths[i:i + bs]]
            x = self.proc(images=ims, return_tensors="pt").to(_dev())
            e = self.model.get_image_features(**x)
            e = e if isinstance(e, torch.Tensor) else e.pooler_output
            out.append(torch.nn.functional.normalize(e.float(), dim=-1).cpu())
        return torch.cat(out)

    @torch.inference_mode()
    def text_emb(self, texts):
        x = self.proc(text=texts, return_tensors="pt", padding=True, truncation=True, max_length=77).to(_dev())
        e = self.model.get_text_features(**x)
        e = e if isinstance(e, torch.Tensor) else e.pooler_output
        return torch.nn.functional.normalize(e.float(), dim=-1).cpu()

    def score(self, paths, texts) -> list[float]:
        """CLIP score = 100 * max(cos(image, text), 0) (as in torchmetrics' CLIPScore)."""
        ie, te = self.image_emb(paths), self.text_emb(texts)
        return [float(100 * max(0.0, (a * b).sum())) for a, b in zip(ie, te)]


def lpips_pairs(rows: list[dict]) -> pd.DataFrame:
    """LPIPS (AlexNet) between the seed-0 and seed-1 image of every prompt and model (higher = more diverse)."""
    import lpips
    # P2_LPIPS_RANDOM=1 only for offline CPU tests (the backbone weights come from download.pytorch.org)
    rand = os.environ.get("P2_LPIPS_RANDOM") == "1"
    net = lpips.LPIPS(net="alex", verbose=False, pnet_rand=rand).to(_dev()).eval()
    df = pd.DataFrame(rows)
    if "nsfw_filtered" in df:
        df = df[~df.nsfw_filtered]
    out = []
    for (m, pid), g in df.groupby(["model", "prompt_id"]):
        if len(g) < 2:
            continue
        a, b = [torch.from_numpy(np.asarray(Image.open(p).convert("RGB").resize((256, 256)), dtype=np.float32) / 127.5 - 1)
                .permute(2, 0, 1)[None].to(_dev()) for p in g.sort_values("seed")["path"][:2]]
        with torch.no_grad():
            d = float(net(a, b).item())
        out.append({"model": m, "prompt_id": pid, "kind": g["kind"].iloc[0], "lpips": d})
    return pd.DataFrame(out)


def grid(rows: list[dict], prompt_ids: list[str], models: list[str], path: str, seed: int = 0, size: int = 256):
    df = pd.DataFrame(rows)
    pad, head = 6, 28
    W = len(models) * (size + pad) + 260
    H = len(prompt_ids) * (size + pad) + head
    canvas = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(canvas)
    try:
        f = ImageFont.truetype("DejaVuSans.ttf", 13)
    except OSError:
        f = ImageFont.load_default()
    for j, m in enumerate(models):
        d.text((260 + j * (size + pad) + 4, 6), m, fill="black", font=f)
    for i, pid in enumerate(prompt_ids):
        y = head + i * (size + pad)
        sub = df[(df.prompt_id == pid) & (df.seed == seed)]
        prompt = sub["prompt"].iloc[0] if len(sub) else pid
        words, line, lines = prompt.split(), "", []
        for w in words:
            if len(line) + len(w) > 32:
                lines.append(line); line = ""
            line += w + " "
        lines.append(line)
        d.text((4, y + 4), f"{pid}\n" + "\n".join(lines), fill="black", font=f)
        for j, m in enumerate(models):
            r = sub[sub.model == m]
            if len(r):
                canvas.paste(Image.open(r["path"].iloc[0]).convert("RGB").resize((size, size)), (260 + j * (size + pad), y))
    canvas.save(path)
    return path


def memorization(clip: Clip, gen_paths: list[str], train_paths: list[str], k: int = 3) -> list[dict]:
    """For each generated image, the most similar training image (CLIP image cosine); returns the k closest pairs."""
    tr = clip.image_emb(train_paths)
    ge = clip.image_emb(gen_paths)
    sim = ge @ tr.T
    best = sim.max(dim=1)
    rows = [{"generated": g, "nearest_train": train_paths[int(j)], "cosine": float(v)}
            for g, v, j in zip(gen_paths, best.values, best.indices)]
    return sorted(rows, key=lambda r: -r["cosine"])[:k], rows


def pair_image(a: str, b: str, path: str, size=384):
    c = Image.new("RGB", (2 * size + 30, size + 40), "white")
    c.paste(Image.open(a).convert("RGB").resize((size, size)), (0, 40))
    c.paste(Image.open(b).convert("RGB").resize((size, size)), (size + 30, 40))
    d = ImageDraw.Draw(c)
    d.text((size // 2 - 5, 10), "A", fill="black")
    d.text((size + 30 + size // 2 - 5, 10), "B", fill="black")
    c.save(path)


def human_eval_sheets(rows: list[dict], final: str, base: str, out_dir: str, n: int = 25, seed: int = 266,
                      raters=("rater1", "rater2", "rater3")) -> tuple[str, str]:
    """Blinded pairs (final vs base, left/right randomised). The key is kept in a separate file."""
    os.makedirs(out_dir, exist_ok=True)
    sheet, key = os.path.join(out_dir, "p2_human_eval.csv"), os.path.join(out_dir, "p2_human_eval_KEY.csv")
    if os.path.isfile(sheet):  # keep a sheet that already has ratings; rebuild an empty one (images may have changed)
        old = pd.read_csv(sheet, dtype=str, keep_default_na=False)
        if any((old[c].str.strip() != "").any() for c in old.columns if c.startswith("rater")):
            return sheet, key
    df = pd.DataFrame(rows)
    if "nsfw_filtered" in df:
        ok = df[~df.nsfw_filtered].groupby(["prompt_id", "seed"]).model.nunique()
        keep = set(ok[ok == df.model.nunique()].index)  # pairs where neither image was filtered
        df = df[[(p, s_) in keep for p, s_ in zip(df.prompt_id, df.seed)]]
    combos = sorted(set(zip(df.prompt_id, df.seed)))
    rng = random.Random(seed)
    pick = rng.sample(combos, min(n, len(combos)))
    srows, krows = [], []
    for k, (pid, s) in enumerate(pick, 1):
        f = df[(df.model == final) & (df.prompt_id == pid) & (df.seed == s)].iloc[0]
        b = df[(df.model == base) & (df.prompt_id == pid) & (df.seed == s)].iloc[0]
        final_left = rng.random() < 0.5
        left, right = (f, b) if final_left else (b, f)
        img = os.path.join(out_dir, f"pair_{k:02d}.png")
        pair_image(left["path"], right["path"], img)
        r = {"pair": k, "image": os.path.basename(img), "prompt": f["prompt"]}
        for rt in raters:
            r[f"{rt}_style"] = ""   # A or B: which image is more in the Naruto domain style
            r[f"{rt}_prompt"] = ""  # A or B: which image matches the prompt better
        srows.append(r)
        krows.append({"pair": k, "final_is": "A" if final_left else "B", "prompt_id": pid, "seed": s})
    pd.DataFrame(srows).to_csv(sheet, index=False)
    pd.DataFrame(krows).to_csv(key, index=False)
    return sheet, key


def human_eval_stats(sheet: str, key: str) -> dict | None:
    if not (os.path.isfile(sheet) and os.path.isfile(key)):
        return None
    s = pd.read_csv(sheet, dtype=str, keep_default_na=False)
    k = pd.read_csv(key, dtype=str, keep_default_na=False).set_index("pair")
    out = {}
    for crit in ("style", "prompt"):
        cols = [c for c in s.columns if c.endswith(f"_{crit}")]
        wins = total = 0
        per_rater = {}
        for c in cols:
            v = s[c].str.strip().str.upper()
            ok = v.isin(["A", "B"])
            fin = s.loc[ok, "pair"].map(lambda p: k.loc[str(p), "final_is"])
            w = int((v[ok] == fin).sum())
            per_rater[c.rsplit("_", 1)[0]] = f"{w}/{int(ok.sum())}"
            wins += w
            total += int(ok.sum())
        if total:
            out[f"win_rate_{crit}"] = round(wins / total, 3)
            out[f"votes_{crit}"] = total
            out[f"per_rater_{crit}"] = per_rater
    return out or None


def summary_table(gen_rows, test_paths, train_stats: dict, lp: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame(gen_rows)
    rows = []
    for m in df.model.unique():
        d_all = df[df.model == m]
        d = d_all[~d_all.nsfw_filtered]  # metrics only on images that passed the NSFW filter
        lp_m = lp[(lp.model == m) & lp.prompt_id.isin(d.prompt_id)]
        ind = d[d.kind == "in_domain"]
        k_mean, k_std = kid(list(ind.path), test_paths)
        ts = train_stats.get(m, {})
        rows.append({
            "model": m,
            "KID in-domain (x1e3, lower=better)": f"{1e3 * k_mean:.2f} +- {1e3 * k_std:.2f}",
            "CLIP score all": round(float(d.clip_score.mean()), 2),
            "CLIP in-domain": round(float(ind.clip_score.mean()), 2),
            "CLIP out-of-domain": round(float(d[d.kind == "out_of_domain"].clip_score.mean()), 2),
            "CLIP compositional": round(float(d[d.kind == "compositional"].clip_score.mean()), 2),
            "LPIPS diversity": round(float(lp_m.lpips.mean()), 3),
            "images used / 40": f"{len(d)}/{len(d_all)}",
            "NSFW filtered (classifier)": int(d_all.nsfw_filtered.sum()),
            "SD checker would flag": int(d_all.sd_checker_flag.fillna(False).astype(bool).sum()),
            "GPU-hours (training)": ts.get("gpu_hours", "-"),
            "peak GPU memory GB (training)": ts.get("peak_gpu_memory_gb", "-"),
        })
    return pd.DataFrame(rows)
