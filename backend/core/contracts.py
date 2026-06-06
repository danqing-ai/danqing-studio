"""
v3 API contracts: request/response DTOs and execution context.
1:1 aligned with REST and engine method signatures.
"""

from __future__ import annotations

import asyncio
import secrets
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Optional

from pydantic import BaseModel, Field, model_validator

from backend.core.asset_interfaces import IAssetStore


# ----- Task kinds (consistent with routes / scheduler) -----

TaskKind = Literal[
    "image.generation",
    "image.edit",
    "image.upscale",
    "video.generation",
    "video.edit",
    "audio.generation",
    "audio.edit",
]


# ----- Image -----


class AdapterRef(BaseModel):
    """LoRA / adapter — ``id`` is the registry LoRA model id (optional ``:version``, e.g. ``bbw-style:fp16``)."""

    id: str
    weight: float = Field(1.0, ge=0.0, le=2.0)


class StructuralGuide(BaseModel):
    asset_id: str
    type: Literal["canny", "depth", "pose", "redux"] = "canny"
    weight: float = 1.0


class StyleGuide(BaseModel):
    asset_id: str
    weight: float = 1.0


def work_title_metadata(title: str) -> dict[str, str]:
    """Optional user-facing work title stored in asset metadata JSON."""
    t = (title or "").strip()
    return {"title": t} if t else {}


class ImageGenerationRequest(BaseModel):
    model: str  # "z-image-turbo:fp16" or "z-image-turbo" (no version uses registry default)
    title: str = ""
    prompt: str
    negative_prompt: str = ""
    size: str = "1024x1024"
    n: int = Field(1, ge=1, le=8)
    steps: Optional[int] = None
    guidance: Optional[float] = None
    seed: Optional[int] = None
    scheduler: Optional[str] = None
    adapters: list[AdapterRef] = Field(default_factory=list)
    structural_guide: Optional[StructuralGuide] = None
    style_guide: Optional[StyleGuide] = None
    priority: Literal["normal", "high"] = "normal"
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExtendSpec(BaseModel):
    directions: list[Literal["top", "bottom", "left", "right"]]
    pixels: int = Field(256, ge=64, le=2048)


class ImageEditRequest(BaseModel):
    model: str
    operation: Literal["rewrite", "retouch", "extend"]
    source_asset_id: str
    title: str = ""
    prompt: str
    source_fidelity: float = Field(0.6, ge=0.0, le=1.0)
    mask_asset_id: Optional[str] = None
    extend: Optional[ExtendSpec] = None
    negative_prompt: str = ""
    n: int = Field(1, ge=1, le=8)
    steps: Optional[int] = None
    seed: Optional[int] = None
    # None → 使用注册表 ``parameters.guidance.default``（与 mflux CLI 对齐时应对 rewrite 显式传 0）
    guidance: Optional[float] = None
    scheduler: Optional[str] = None
    adapters: list[AdapterRef] = Field(default_factory=list)
    priority: Literal["normal", "high"] = "normal"
    metadata: dict[str, Any] = Field(default_factory=dict)
    # operation=rewrite: reference=full-image img2img; instruct=instruction-based edit (currently only flux1-kontext / text_editing). None=use legacy auto rules.
    rewrite_mode: Optional[Literal["reference", "instruct"]] = None

    @model_validator(mode="after")
    def _rewrite_mode_consistency(self) -> "ImageEditRequest":
        if self.rewrite_mode is not None and self.operation != "rewrite":
            raise ValueError("rewrite_mode is only valid when operation is rewrite")
        return self


class ImageUpscaleRequest(BaseModel):
    model: str
    source_asset_id: str
    scale: Literal[2, 4] = 2
    denoise: float = Field(0.0, ge=0.0, le=1.0)
    tile_size: int = Field(1024, ge=256, le=4096)
    priority: Literal["normal", "high"] = "normal"
    metadata: dict[str, Any] = Field(default_factory=dict)


# ----- Video -----


class VideoGenerationRequest(BaseModel):
    model: str
    title: str = ""
    prompt: str
    negative_prompt: str = ""
    size: str = "832x480"
    num_frames: int = 81
    fps: int = 16
    steps: Optional[int] = None
    guidance: Optional[float] = None
    shift: Optional[float] = None
    seed: Optional[int] = None
    adapters: list[AdapterRef] = Field(default_factory=list)
    priority: Literal["normal", "high"] = "normal"
    metadata: dict[str, Any] = Field(default_factory=dict)


class VideoEditRequest(BaseModel):
    model: str
    operation: Literal["animate"] = "animate"
    source_asset_id: str
    tail_asset_id: Optional[str] = None
    title: str = ""
    prompt: str
    negative_prompt: str = ""
    size: str = "832x480"
    num_frames: int = 81
    fps: int = 16
    steps: Optional[int] = None
    guidance: Optional[float] = None
    shift: Optional[float] = None
    seed: Optional[int] = None
    adapters: list[AdapterRef] = Field(default_factory=list)
    priority: Literal["normal", "high"] = "normal"
    metadata: dict[str, Any] = Field(default_factory=dict)


class VideoUpscaleRequest(BaseModel):
    model: str
    source_asset_id: str
    scale: Literal[2, 4] = 2
    denoise: float = Field(0.3, ge=0.0, le=1.0)
    tile_size: int = Field(1024, ge=256, le=4096)
    temporal_window: int = Field(5, ge=1, le=16)
    fps: int = 16
    priority: Literal["normal", "high"] = "normal"
    metadata: dict[str, Any] = Field(default_factory=dict)


# ----- Audio -----


class AudioGenerationRequest(BaseModel):
    model: str  # e.g. "audio-stub" or "audio-stub:stub"
    title: str = ""
    prompt: str
    negative_prompt: str = ""
    duration: Optional[int] = None  # seconds (10-600)
    instrumental: bool = False
    lyrics: str = ""
    vocal_language: str = ""
    vocal_type: str = ""  # male | female | chorus | duet | auto (caption template, not a DiT knob)
    bpm: Optional[int] = None  # auto-detect if None
    key_scale: str = ""  # e.g. "C Major", empty=auto-detect
    time_signature: str = ""  # "2","3","4","6", empty=auto-detect
    steps: Optional[int] = None
    guidance: Optional[float] = None
    seed: Optional[int] = None
    n: int = Field(2, ge=1, le=8)
    simple_mode: bool = False  # advanced: force 5Hz LM inspiration (create_sample)
    lm_expansion: Optional[str] = None  # auto | inspiration | format | off
    audio_format: str = "mp3"
    priority: Literal["normal", "high"] = "normal"
    metadata: dict[str, Any] = Field(default_factory=dict)


class AudioEditRequest(BaseModel):
    model: str
    operation: Literal["cover"] = "cover"
    source_asset_id: str
    prompt: str = ""
    source_fidelity: float = Field(1.0, ge=0.0, le=1.0)
    steps: Optional[int] = None
    guidance: Optional[float] = None
    seed: Optional[int] = None
    n: int = Field(1, ge=1, le=8)
    audio_format: str = "mp3"
    priority: Literal["normal", "high"] = "normal"
    metadata: dict[str, Any] = Field(default_factory=dict)


# U+23 U+24 U+25 (video) --- de-later


class CancelToken:
    __slots__ = ("_cancelled",)

    def __init__(self) -> None:
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def is_cancelled(self) -> bool:
        return self._cancelled

    def raise_if_cancelled(self) -> None:
        if self._cancelled:
            raise asyncio.CancelledError()


@dataclass
class ProgressEvent:
    progress: float
    step: Optional[int] = None
    total: Optional[int] = None
    eta_seconds: Optional[float] = None
    message: Optional[str] = None
    # UX phases: encoding | loading_model | denoising | decoding | saving
    phase: Optional[str] = None


@dataclass
class LogEvent:
    level: Literal["info", "warning", "error", "success"]
    message: str


@dataclass
class ExecutionContext:
    task_id: str
    cancel_token: CancelToken
    on_progress: Callable[[ProgressEvent], None]
    on_log: Callable[[LogEvent], None]
    work_dir: Path
    asset_store: IAssetStore


@dataclass
class EngineResult:
    primary_asset_id: str
    asset_ids: list[str] = field(default_factory=list)
    output_paths: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def new_asset_id() -> str:
    return "ast_" + secrets.token_hex(12)


def new_task_id() -> str:
    return "tsk_" + secrets.token_hex(12)


def parse_model_version(model_field: str) -> tuple[str, str]:
    """'flux1-dev:fp16' -> ('flux1-dev', 'fp16'); no colon means version is empty"""
    if ":" in model_field:
        a, b = model_field.split(":", 1)
        return a.strip(), b.strip()
    return model_field.strip(), ""


def parse_size(size: str) -> tuple[int, int]:
    parts = size.lower().replace("×", "x").split("x")
    if len(parts) != 2:
        return 1024, 1024
    return int(parts[0]), int(parts[1])
