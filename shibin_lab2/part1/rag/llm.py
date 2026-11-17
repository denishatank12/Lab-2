"""Generator backends.

* QwenVL   - Qwen2.5-VL (default Qwen/Qwen2.5-VL-3B-Instruct) via transformers.
             Used to describe figures at ingestion and to write answers; when a
             question names a figure, the cropped figure image is attached.
* HFText   - any text-only chat model (used for CPU tests).
* NoLLM    - returns "" (ingestion-only runs; descriptions fall back to text).

All decoding is greedy (do_sample=False) so runs are reproducible.
"""
from __future__ import annotations

import gc

import torch

from .config import Config


def dtype_kw(dt) -> dict:
    """transformers>=4.56 renamed torch_dtype -> dtype; support both."""
    import transformers
    major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
    return {"dtype": dt} if (major, minor) >= (4, 56) else {"torch_dtype": dt}


def _device_dtype():
    if torch.cuda.is_available():
        major = torch.cuda.get_device_capability(0)[0]
        # T4 / P100 have no bf16 -> fp16
        return "cuda", (torch.bfloat16 if major >= 8 else torch.float16)
    # bf16 halves CPU memory (a 3B VLM needs ~7.5 GB instead of ~15 GB)
    return "cpu", torch.bfloat16


class NoLLM:
    name = "none"
    supports_images = False

    def chat(self, system, user, images=None, max_new_tokens=256):
        return ""


def quant_kw(dt) -> dict:
    """4-bit NF4 weights (bitsandbytes) on GPU: a 14B model needs ~9.5 GB, an 8B VLM ~6 GB."""
    from transformers import BitsAndBytesConfig
    return {"quantization_config": BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                      bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dt),
            "device_map": {"": 0}}


_THINK_RE = None


def strip_think(text: str) -> str:
    import re
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()


class HFText:
    """Text-only chat model (answers from the evidence text, table facts and figure descriptions)."""
    supports_images = False

    def __init__(self, model_id: str, max_new_tokens=320, load_in_4bit=False):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        dev, dt = _device_dtype()
        self.name = model_id
        self.tok = AutoTokenizer.from_pretrained(model_id)
        kw = dtype_kw(dt)
        if load_in_4bit and dev == "cuda":
            kw.update(quant_kw(dt))
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kw)
        if "device_map" not in kw:
            self.model = self.model.to(dev)
        self.model.eval()
        self.dev = self.model.device
        self.max_new_tokens = max_new_tokens

    def count_tokens(self, system: str, user: str) -> int:
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        return len(self.tok(self._template(msgs))["input_ids"])

    def _template(self, msgs):
        try:  # Qwen3: answer directly, without a long "thinking" section (keeps the Kaggle run within time)
            return self.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
        except TypeError:
            return self.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)

    @torch.inference_mode()
    def chat(self, system, user, images=None, max_new_tokens=None):
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        enc = self.tok(self._template(msgs), return_tensors="pt").to(self.dev)
        out = self.model.generate(**enc, max_new_tokens=max_new_tokens or self.max_new_tokens, do_sample=False,
                                  pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id)
        return strip_think(self.tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True))


class QwenVL:
    supports_images = True

    def __init__(self, model_id: str, max_new_tokens=320, load_in_4bit=False, max_pixels=1024 * 28 * 28):
        from transformers import AutoProcessor
        try:  # works for Qwen2.5-VL and Qwen3-VL
            from transformers import AutoModelForImageTextToText as VLCls
        except ImportError:
            from transformers import Qwen2_5_VLForConditionalGeneration as VLCls
        dev, dt = _device_dtype()
        kw = dtype_kw(dt)
        if load_in_4bit and dev == "cuda":
            kw.update(quant_kw(dt))
        self.model = VLCls.from_pretrained(model_id, **kw)
        if "device_map" not in kw:
            self.model = self.model.to(dev)
        self.model.eval()
        try:
            self.proc = AutoProcessor.from_pretrained(model_id, min_pixels=128 * 28 * 28, max_pixels=max_pixels)
        except Exception:
            self.proc = AutoProcessor.from_pretrained(model_id)
        self.name = model_id
        self.dev = dev
        self.max_new_tokens = max_new_tokens

    @torch.inference_mode()
    def chat(self, system, user, images=None, max_new_tokens=None):
        from PIL import Image
        content = []
        pil = []
        for im in images or []:
            img = Image.open(im).convert("RGB") if isinstance(im, str) else im.convert("RGB")
            pil.append(img)
            content.append({"type": "image"})
        content.append({"type": "text", "text": user})
        msgs = [{"role": "system", "content": [{"type": "text", "text": system}]},
                {"role": "user", "content": content}]
        text = self.proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = self.proc(text=[text], images=pil or None, return_tensors="pt").to(self.model.device)
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens or self.max_new_tokens, do_sample=False)
        return strip_think(self.proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0])


def load_llm(cfg: Config):
    """The model that writes answers."""
    if cfg.llm_backend == "none":
        return NoLLM()
    if cfg.llm_backend == "hf_text":  # small text model (CPU tests)
        return HFText(cfg.text_llm_model, cfg.max_new_tokens)
    if cfg.llm_backend == "two_stage":  # strong text model answers from text + table facts + figure descriptions
        return HFText(cfg.answer_model, cfg.max_new_tokens, cfg.load_in_4bit)
    return QwenVL(cfg.vlm_model, cfg.max_new_tokens, cfg.load_in_4bit)  # one VLM for everything


def load_describer(cfg: Config):
    """The vision-language model that describes the figures at ingestion."""
    return QwenVL(cfg.vlm_model, cfg.max_new_tokens, cfg.load_in_4bit)


def free(obj):
    try:
        if hasattr(obj, "model"):
            del obj.model
    except Exception:
        pass
    del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
