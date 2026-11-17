"""CPU tests of Part 2 (tiny Stable Diffusion from hf-internal-testing; no GPU, seconds per step)."""
import os
import sys

import numpy as np
import pandas as pd
import pytest
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from p2.config import P2Config  # noqa: E402
from p2.data import clean, dhash, split, to_square  # noqa: E402

TINY = "hf-internal-testing/tiny-stable-diffusion-torch"


def _img(seed, size=(300, 200)):
    return Image.fromarray(np.random.default_rng(seed).integers(0, 255, (size[1], size[0], 3), dtype=np.uint8))


def test_clean_removes_corrupt_small_blank_and_duplicates():
    cfg = P2Config()
    a = _img(1)
    near = a.copy()
    near.putpixel((0, 0), (0, 0, 0))  # tiny change -> near duplicate
    imgs = [a, a.copy(), near, b"\x89PNG not an image", _img(2, (60, 60)), Image.new("RGB", (300, 300), (9, 9, 9)), _img(3)]
    caps = ["x"] * len(imgs)
    rep = clean(imgs, caps, cfg, log=lambda *a: None)
    s = rep.summary()
    assert s["exact_duplicates_removed"] == 1 and s["near_duplicates_removed"] == 1
    assert s["corrupted_removed"] == 3 and rep.kept == [0, 6]


def test_split_is_deterministic_90_10():
    cfg = P2Config()
    kept = list(range(100))
    a, b = split(kept, cfg), split(kept, cfg)
    assert a == b and len(a[1]) == 10 and not set(a[0]) & set(a[1])


def test_square_and_hash():
    im = to_square(_img(4, (640, 480)), 512)
    assert im.size == (512, 512)
    assert dhash(im) == dhash(im.copy())


@pytest.mark.slow
def test_tiny_train_generate_eval(tmp_path):
    os.environ["P2_LPIPS_RANDOM"] = "1"
    from p2 import evaluate as ev, generate, train
    rows = []
    for i in range(6):
        p = tmp_path / f"{i}.png"
        _img(i, (64, 64)).save(p)
        rows.append({"path": str(p), "text": f"a ninja {i}"})
    cfg = P2Config(base_model=TINY, work_dir=str(tmp_path), resolution=64, max_steps=2, batch_size=2, warmup_steps=1, steps=2)
    st = train.train_lora(cfg, "run1_lora_r4", 4, rows, log=lambda *a: None)
    assert st["trainable_params"] > 0 and os.path.isfile(os.path.join(st["lora_dir"], "pytorch_lora_weights.safetensors"))
    prompts = [{"id": "in01", "kind": "in_domain", "prompt": "a ninja"}]
    gen = generate.generate_all(cfg, prompts, {"base": None, "run1_lora_r4": st["lora_dir"]}, str(tmp_path / "g"),
                                log=lambda *a: None)
    assert len(gen) == 4 and all(os.path.isfile(r["path"]) for r in gen)
    lp = ev.lpips_pairs(gen)
    assert set(lp.model) == {"base", "run1_lora_r4"}
    sheet, key = ev.human_eval_sheets(gen, "run1_lora_r4", "base", str(tmp_path / "h"), n=2)
    s = pd.read_csv(sheet)
    k = pd.read_csv(key)
    for c in ("rater1_style", "rater2_style", "rater3_style"):
        s[c] = k["final_is"].values  # everyone picks the fine-tuned image
    s.to_csv(sheet, index=False)
    assert ev.human_eval_stats(sheet, key)["win_rate_style"] == 1.0
