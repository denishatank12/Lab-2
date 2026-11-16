"""Model backends for Part 3 (all open models, one 16 GB GPU).

  VLM        Qwen/Qwen2.5-VL-3B-Instruct      planner probe, vision, prompt rewriting, critique judgement
  img2img    stable-diffusion-v1-5 + LoRA      style change; domain style uses the Part 2 LoRA (LORA_REPO)
  edit       timbrooks/instruct-pix2pix        instruction edits
  enhance    PIL (autocontrast, denoise, unsharp mask, optional 2x Lanczos)
  CLIP       openai/clip-vit-large-patch14     prompt/image similarity
  safety     CompVis safety checker (from the SD 1.5 repo) + VLM yes/no check

Models load lazily and can be moved off the GPU after use. `MockBackends`
gives deterministic stand-ins so the orchestration logic is unit-testable on CPU.
"""
from __future__ import annotations

import gc
import json
import os
import re
import time
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


def _dev():
    import torch
    if torch.cuda.is_available():
        return "cuda", (torch.bfloat16 if torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16)
    return "cpu", torch.bfloat16  # halves CPU RAM for the VLM


def dtype_kw(dt) -> dict:
    import transformers
    major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
    return {"dtype": dt} if (major, minor) >= (4, 56) else {"torch_dtype": dt}


def _free():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def parse_json_obj(raw: str) -> dict:
    raw = re.sub(r"^```(json)?|```$", "", (raw or "").strip(), flags=re.M).strip()
    s, e = raw.find("{"), raw.rfind("}")
    if s >= 0 and e > s:
        for cand in (raw[s:e + 1], re.sub(r",\s*([}\]])", r"\1", raw[s:e + 1])):
            try:
                v = json.loads(cand)
                if isinstance(v, dict):
                    return v
            except Exception:
                pass
    return {}


@dataclass
class BackendConfig:
    vlm_model: str = os.environ.get("P3_VLM", "Qwen/Qwen3-VL-8B-Instruct")
    vlm_4bit: bool = os.environ.get("P3_VLM_4BIT", "1") == "1"
    sd_model: str = os.environ.get("P3_SD", "stable-diffusion-v1-5/stable-diffusion-v1-5")
    pix2pix_model: str = os.environ.get("P3_PIX2PIX", "timbrooks/instruct-pix2pix")
    clip_model: str = os.environ.get("P3_CLIP", "openai/clip-vit-large-patch14")
    lora_repo: str = os.environ.get("LORA_REPO", "")  # Part 2 adapter on Hugging Face, e.g. "<hf-user>/sd15-naruto-lora"
    lora_weight_name: str = os.environ.get("LORA_WEIGHT_NAME", "")  # e.g. pytorch_lora_weights.safetensors
    lora_scale: float = float(os.environ.get("LORA_SCALE", "1.0"))
    domain_name: str = os.environ.get("DOMAIN_NAME", "Naruto")  # the Part 2 dataset domain
    domain_trigger: str = os.environ.get("DOMAIN_TRIGGER", "")  # extra words the LoRA was trained with
    steps: int = int(os.environ.get("P3_STEPS", "30"))
    size: int = int(os.environ.get("P3_SIZE", "512"))
    cpu_offload: bool = os.environ.get("P3_OFFLOAD", "0") == "1"
    max_pixels: int = 768 * 28 * 28


class RealBackends:
    def __init__(self, cfg: BackendConfig | None = None, log=print):
        self.cfg = cfg or BackendConfig()
        self.log = log
        self._vlm = self._proc = None
        self._img2img = self._txt2img = self._pix2pix = None
        self._clip = self._clip_proc = None
        self._safety = self._safety_fe = None
        self._lora_loaded = False

    # ---------------- VLM ----------------
    def _load_vlm(self):
        if self._vlm is None:
            from transformers import AutoProcessor
            try:  # Qwen3-VL and Qwen2.5-VL
                from transformers import AutoModelForImageTextToText as VL
            except ImportError:
                from transformers import Qwen2_5_VLForConditionalGeneration as VL
            dev, dt = _dev()
            t = time.time()
            kw = dtype_kw(dt)
            if self.cfg.vlm_4bit and dev == "cuda":  # ~6 GB for the 8B VLM, leaves room for SD + pix2pix + CLIP
                from transformers import BitsAndBytesConfig
                kw.update(quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                                 bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dt),
                          device_map={"": 0})
            self._vlm = VL.from_pretrained(self.cfg.vlm_model, **kw)
            if "device_map" not in kw:
                self._vlm = self._vlm.to(dev)
            self._vlm.eval()
            try:
                self._proc = AutoProcessor.from_pretrained(self.cfg.vlm_model, min_pixels=128 * 28 * 28,
                                                           max_pixels=self.cfg.max_pixels)
            except Exception:
                self._proc = AutoProcessor.from_pretrained(self.cfg.vlm_model)
            self.log(f"[backends] loaded {self.cfg.vlm_model} on {dev}{' (4-bit)' if 'quantization_config' in kw else ''} "
                     f"in {time.time() - t:.0f}s")
        return self._vlm, self._proc

    def vlm(self, prompt: str, images: list | None = None, system: str = "", max_new_tokens: int = 300) -> str:
        import torch
        model, proc = self._load_vlm()
        content, pil = [], []
        for im in images or []:
            img = Image.open(im).convert("RGB") if isinstance(im, str) else im.convert("RGB")
            pil.append(img)
            content.append({"type": "image"})
        content.append({"type": "text", "text": prompt})
        msgs = ([{"role": "system", "content": [{"type": "text", "text": system}]}] if system else []) + \
               [{"role": "user", "content": content}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = proc(text=[text], images=pil or None, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        txt = proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        return re.sub(r"<think>.*?</think>", "", txt, flags=re.S).strip()

    def vlm_json(self, prompt, images=None, system="", max_new_tokens=300) -> dict:
        raw = self.vlm(prompt, images, system, max_new_tokens)
        d = parse_json_obj(raw)
        d.setdefault("_raw", raw)
        return d

    # ---------------- diffusion ----------------
    def _sd_kwargs(self):
        dev, dt = _dev()
        import torch
        # fp16 on GPU; bf16 on CPU (CPU tests only - halves RAM). The fp16 weight variant is downloaded either way.
        kw = {"torch_dtype": torch.float16 if dev == "cuda" else torch.bfloat16, "variant": "fp16"}
        return dev, kw

    def _place(self, pipe):
        dev, _ = _dev()
        if dev == "cuda" and self.cfg.cpu_offload:
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(dev)
        pipe.set_progress_bar_config(disable=True)
        return pipe

    def img2img_pipe(self, with_lora: bool):
        from diffusers import StableDiffusionImg2ImgPipeline
        if self._img2img is None:
            dev, kw = self._sd_kwargs()
            pipe = StableDiffusionImg2ImgPipeline.from_pretrained(self.cfg.sd_model, safety_checker=None,
                                                                  requires_safety_checker=False, **kw)
            self._img2img = self._place(pipe)
        self._set_lora(self._img2img, with_lora)
        return self._img2img

    def txt2img_pipe(self, with_lora: bool):
        from diffusers import StableDiffusionPipeline
        if self._txt2img is None:
            base = self.img2img_pipe(with_lora=False)
            self._txt2img = StableDiffusionPipeline(**{k: v for k, v in base.components.items()
                                                       if k in StableDiffusionPipeline._get_signature_keys(StableDiffusionPipeline)[0]
                                                       or k in ("safety_checker", "feature_extractor", "image_encoder")},
                                                    requires_safety_checker=False)
            self._txt2img.set_progress_bar_config(disable=True)
        self._set_lora(self._txt2img, with_lora)
        return self._txt2img

    def _set_lora(self, pipe, on: bool):
        if not self.cfg.lora_repo:
            return
        if on and not self._lora_loaded:
            kw = {"weight_name": self.cfg.lora_weight_name} if self.cfg.lora_weight_name else {}
            self.img2img_pipe_raw().load_lora_weights(self.cfg.lora_repo, adapter_name="domain", **kw)
            self._lora_loaded = True
        if self._lora_loaded:
            if on:
                pipe.set_adapters(["domain"], adapter_weights=[self.cfg.lora_scale])
                pipe.enable_lora()
            else:
                pipe.disable_lora()

    def img2img_pipe_raw(self):
        return self._img2img

    def pix2pix_pipe(self):
        from diffusers import EulerAncestralDiscreteScheduler, StableDiffusionInstructPix2PixPipeline
        if self._pix2pix is None:
            dev, kw = self._sd_kwargs()
            pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(self.cfg.pix2pix_model, safety_checker=None,
                                                                          requires_safety_checker=False, **kw)
            pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
            self._pix2pix = self._place(pipe)
        return self._pix2pix

    def generate(self, mode: str, prompt: str, negative: str, image: Image.Image | None, seed: int, steps: int,
                 strength: float = 0.6, guidance: float = 7.5, image_guidance: float = 1.5,
                 level: int = 1) -> tuple[Image.Image, dict]:
        import torch
        dev, _ = _dev()
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        if mode == "enhance":
            return enhance(image, level), {"model": f"PIL enhance level {level} (median, autocontrast, gamma, unsharp, 2x Lanczos)"}
        if mode == "pix2pix_edit":
            pipe = self.pix2pix_pipe()
            out = pipe(prompt=prompt, negative_prompt=negative or None, image=image, num_inference_steps=steps,
                       guidance_scale=guidance, image_guidance_scale=image_guidance, generator=g)
            return out.images[0], {"model": self.cfg.pix2pix_model}
        lora = mode.endswith("domain_lora") and bool(self.cfg.lora_repo)
        if mode.startswith("txt2img"):
            pipe = self.txt2img_pipe(with_lora=lora)
            out = pipe(prompt=prompt, negative_prompt=negative or None, num_inference_steps=steps,
                       guidance_scale=guidance, height=self.cfg.size, width=self.cfg.size, generator=g)
        else:
            pipe = self.img2img_pipe(with_lora=lora)
            out = pipe(prompt=prompt, negative_prompt=negative or None, image=image, strength=strength,
                       num_inference_steps=steps, guidance_scale=guidance, generator=g)
        return out.images[0], {"model": self.cfg.sd_model, "lora": self.cfg.lora_repo if lora else ""}

    # ---------------- CLIP ----------------
    def clip_score(self, image: Image.Image, text: str) -> float:
        import torch
        from transformers import CLIPModel, CLIPProcessor
        if self._clip is None:
            dev, _ = _dev()
            self._clip = CLIPModel.from_pretrained(self.cfg.clip_model).to(dev).eval()
            self._clip_proc = CLIPProcessor.from_pretrained(self.cfg.clip_model)
        inp = self._clip_proc(text=[text[:300]], images=image, return_tensors="pt", padding=True, truncation=True,
                              max_length=77).to(self._clip.device)
        with torch.inference_mode():
            out = self._clip(**inp)
        ie = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
        te = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
        return float(100.0 * (ie * te).sum().clamp(min=0))

    # ---------------- safety ----------------
    def nsfw_check(self, image: Image.Image) -> tuple[bool, str]:
        """(unsafe?, reason) from the Stable Diffusion safety checker."""
        import torch
        from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
        from transformers import CLIPImageProcessor
        if self._safety is None:
            dev, _ = _dev()
            self._safety = StableDiffusionSafetyChecker.from_pretrained(
                self.cfg.sd_model, subfolder="safety_checker", variant="fp16", torch_dtype=torch.float32).to(dev).eval()
            self._safety_fe = CLIPImageProcessor.from_pretrained(self.cfg.sd_model, subfolder="feature_extractor")
        x = self._safety_fe(images=[image], return_tensors="pt").pixel_values.to(self._safety.device)
        arr = np.asarray(image, dtype=np.float32)[None] / 255.0
        with torch.inference_mode():
            _, flags = self._safety(images=arr, clip_input=x)
        return bool(flags[0]), ("safety checker flagged NSFW concept" if flags[0] else "")

    def unload_diffusion(self):
        self._img2img = self._txt2img = self._pix2pix = None
        self._lora_loaded = False
        _free()


def enhance(image: Image.Image, level: int = 1, upscale: bool = True) -> Image.Image:
    """Deterministic enhancement; `level` (1-3) is raised by the critique when the result is not good enough."""
    im = image.convert("RGB")
    im = im.filter(ImageFilter.MedianFilter(3 if level <= 1 else 5))  # denoise first
    im = ImageOps.autocontrast(im, cutoff=1 + level)
    mean = float(np.asarray(im.convert("L"), dtype=np.float32).mean())
    if mean < 110:  # lift dark images towards mid-grey with a gamma curve
        target = 110 + 10 * (level - 1)
        gamma = np.log(target / 255.0) / np.log(max(mean, 1.0) / 255.0)
        lut = [min(255, int(255 * (i / 255.0) ** gamma)) for i in range(256)]
        im = im.point(lut * 3)
    im = im.filter(ImageFilter.UnsharpMask(radius=2, percent=100 + 60 * level, threshold=3))
    im = ImageEnhance.Color(im).enhance(1.0 + 0.1 * level)
    if upscale and max(im.size) <= 512:
        im = im.resize((im.width * 2, im.height * 2), Image.LANCZOS)
    return im


def quality_stats(image: Image.Image) -> dict:
    g = np.asarray(image.convert("L").resize((256, 256)), dtype=np.float32)
    lap = g[1:-1, 1:-1] * -4 + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]
    return {"brightness": round(float(g.mean()), 1), "contrast": round(float(g.std()), 1),
            "sharpness": round(float(lap.var()), 1)}


# ---------------------------------------------------------------------------
@dataclass
class MockBackends:
    """Deterministic stand-ins (no model downloads) for unit tests of the control flow."""
    cfg: BackendConfig = field(default_factory=BackendConfig)
    clip_values: list = field(default_factory=lambda: [31.0])  # successive CLIP scores returned
    vlm_scores: list = field(default_factory=lambda: [4])
    image_text: list = field(default_factory=list)
    has_person: bool = False
    unsafe_output: bool = False
    preserved: bool = True  # answer of the per-item preservation questions
    fail_generation: bool = False
    rewrite_prompt: str = ""  # what the (mock) prompt engineer VLM returns; "" -> template fallback
    caption: str = "a red house under a blue sky"
    calls: list = field(default_factory=list)

    def vlm(self, prompt, images=None, system="", max_new_tokens=300):
        if prompt.lower().startswith("answer yes or no only. does this image still show"):
            return "yes" if self.preserved else "no"
        if prompt.lower().startswith("answer yes or no"):
            return "yes" if self.unsafe_output else "no"
        return json.dumps(self.vlm_json(prompt, images, system))

    def vlm_json(self, prompt, images=None, system="", max_new_tokens=300) -> dict:
        self.calls.append(("vlm", prompt[:40]))
        p = prompt.lower()
        if "probe" in p:
            return {"visible_text": self.image_text, "has_person": self.has_person, "is_photo": self.has_person}
        if "classify" in p:
            return {"intent": "ambiguous"}
        if '"caption"' in p or "describe the image" in p:
            return {"caption": self.caption, "objects": ["house", "sky", "tree"],
                    "visible_text": self.image_text, "answers": [{"question": "q", "answer": "a"}],
                    "has_person": self.has_person, "is_photo": True, "style": "photo"}
        if "rewrite" in p:
            return {"prompt": self.rewrite_prompt, "target_description": self.rewrite_prompt, "keep": []}
        if "all_present" in p:
            return {"all_present": True, "missing": []}
        if "judge" in p or '"score"' in p:
            s = self.vlm_scores[min(len([c for c in self.calls if c[0] == 'judge']), len(self.vlm_scores) - 1)]
            self.calls.append(("judge", s))
            return {"score": s, "matches_request": s >= 3, "preserved": True, "issues": "" if s >= 3 else "style too weak",
                    "revision_instruction": "" if s >= 3 else "make the style stronger"}
        if '"unsafe"' in p or "unsafe" in p:
            return {"unsafe": self.unsafe_output, "reason": "mock"}
        return {}

    def generate(self, mode, prompt, negative, image, seed, steps, strength=0.6, guidance=7.5, image_guidance=1.5, level=1):
        self.calls.append(("gen", mode))
        if mode == "enhance" and image is not None:
            return enhance(image, level), {"model": f"PIL enhance level {level}"}
        if self.fail_generation:
            raise RuntimeError("CUDA out of memory (mock)")
        base = image if image is not None else Image.new("RGB", (512, 512), (120, 120, 200))
        return ImageOps.posterize(base.convert("RGB"), 3), {"model": f"mock-{mode}"}

    def clip_score(self, image, text):
        n = len([c for c in self.calls if c[0] == "clip"])
        self.calls.append(("clip", n))
        return float(self.clip_values[min(n, len(self.clip_values) - 1)])

    def nsfw_check(self, image):
        return self.unsafe_output, ("mock nsfw" if self.unsafe_output else "")

    def unload_diffusion(self):
        pass