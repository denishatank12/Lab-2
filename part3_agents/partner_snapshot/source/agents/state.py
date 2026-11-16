"""Shared, typed state passed between the five agents (Task 3.2.1)."""
from __future__ import annotations

import time
import uuid
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

Intent = Literal["caption_question", "style_change", "edit", "enhancement", "domain_style",
                 "unsupported", "ambiguous", "unsafe"]
Status = Literal["running", "answered", "done", "refused", "needs_clarification", "error"]


class ImageCheck(BaseModel):
    ok: bool = False
    issue: str = ""  # "" | missing | empty | broken | too_small | low_quality | unsupported_format
    width: int = 0
    height: int = 0
    blur: float = 0.0  # variance of Laplacian (low = blurry)
    brightness: float = 0.0
    warning: str = ""


class VisionOutput(BaseModel):
    caption: str = ""
    objects: list[str] = Field(default_factory=list)
    visible_text: list[str] = Field(default_factory=list)
    qa: list[dict] = Field(default_factory=list)  # [{"question":..., "answer":...}]
    has_person: bool = False
    is_photo: bool = False
    style: str = ""


class PromptSpec(BaseModel):
    prompt: str = ""  # what the generator receives (description for img2img, instruction for pix2pix)
    negative_prompt: str = ""
    target_description: str = ""  # what the result should show (used by CLIP in the critique)
    keep: list[str] = Field(default_factory=list)  # things the user wants unchanged
    removed_image_text: list[str] = Field(default_factory=list)


class GenRecord(BaseModel):
    attempt: int = 0
    mode: str = ""  # img2img_style | img2img_domain_lora | txt2img_domain_lora | pix2pix_edit | enhance
    model: str = ""
    lora: str = ""
    prompt: str = ""
    negative_prompt: str = ""
    seed: int = 0
    steps: int = 0
    strength: Optional[float] = None
    guidance_scale: Optional[float] = None
    image_guidance_scale: Optional[float] = None
    width: int = 0
    height: int = 0
    image_path: str = ""
    seconds: float = 0.0
    nsfw_flag_from_pipeline: Optional[bool] = None


class Critique(BaseModel):
    attempt: int = 0
    clip_score: float = 0.0  # 100 * cosine(image, target_description)
    vlm_score: int = 0  # 1-5
    vlm_judgement: str = ""
    preserved_ok: Optional[bool] = None
    safe: bool = True
    safety_reason: str = ""
    decision: Literal["accept", "revise", "reject"] = "accept"
    revision_instruction: str = ""
    param_hint: dict = Field(default_factory=dict)


class TraceEvent(BaseModel):
    agent: str
    started_at: float
    seconds: float
    input: Any = None
    output: Any = None
    error: str = ""


class AgentState(BaseModel):
    request_id: str = Field(default_factory=lambda: time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6])
    user_request: str = ""
    image_path: str = ""
    seed: int = 266
    use_critique: bool = True
    max_revisions: int = 2

    image_check: ImageCheck = Field(default_factory=ImageCheck)
    intent: Optional[Intent] = None
    route: list[str] = Field(default_factory=list)
    planner_notes: list[str] = Field(default_factory=list)
    image_text_probe: list[str] = Field(default_factory=list)  # text seen in the image by the planner's probe
    refused: bool = False
    refusal_reason: str = ""

    vision: Optional[VisionOutput] = None
    prompt: Optional[PromptSpec] = None
    generations: list[GenRecord] = Field(default_factory=list)
    critiques: list[Critique] = Field(default_factory=list)
    revisions: int = 0

    final_image: str = ""
    final_message: str = ""
    status: Status = "running"
    error: str = ""
    trace: list[TraceEvent] = Field(default_factory=list)
    total_seconds: float = 0.0