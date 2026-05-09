"""
v3 API 契约：请求/响应 DTO 与执行上下文。
与 REST 及引擎方法签名 1:1 对齐。
"""

from __future__ import annotations

import asyncio
import secrets
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Optional

from pydantic import BaseModel, Field, model_validator

from backend.core.asset_interfaces import IAssetStore


# ----- 任务种类（与路由 / 调度器一致）-----

TaskKind = Literal[
    "image.generation",
    "image.edit",
    "image.upscale",
    "video.generation",
    "video.edit",
]


# ----- 图像 -----


class AdapterRef(BaseModel):
    id: str
    weight: float = Field(1.0, ge=0.0, le=2.0)


class StructuralGuide(BaseModel):
    asset_id: str
    type: Literal["canny", "depth", "pose", "redux"] = "canny"
    weight: float = 1.0


class StyleGuide(BaseModel):
    asset_id: str
    weight: float = 1.0


class ImageGenerationRequest(BaseModel):
    model: str  # "z-image-turbo:fp16" 或 "z-image-turbo"（无版本则用注册表默认）
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
    prompt: str
    source_fidelity: float = Field(0.6, ge=0.0, le=1.0)
    mask_asset_id: Optional[str] = None
    extend: Optional[ExtendSpec] = None
    negative_prompt: str = ""
    n: int = Field(1, ge=1, le=8)
    steps: Optional[int] = None
    seed: Optional[int] = None
    adapters: list[AdapterRef] = Field(default_factory=list)
    priority: Literal["normal", "high"] = "normal"
    metadata: dict[str, Any] = Field(default_factory=dict)
    # operation=rewrite：reference=整图 img2img；instruct=指令编辑（当前仅 flux1-kontext / text_editing）。None=沿用旧版自动规则。
    rewrite_mode: Optional[Literal["reference", "instruct"]] = None

    @model_validator(mode="after")
    def _rewrite_mode_consistency(self) -> "ImageEditRequest":
        if self.rewrite_mode is not None and self.operation != "rewrite":
            raise ValueError("rewrite_mode is only valid when operation is rewrite")
        if self.operation == "rewrite" and self.rewrite_mode == "instruct":
            base = self.model.split(":", 1)[0].strip()
            if base != "flux1-kontext":
                raise ValueError(
                    "rewrite_mode instruct requires model flux1-kontext; use rewrite_mode reference or omit rewrite_mode"
                )
        return self


class ImageUpscaleRequest(BaseModel):
    model: str
    source_asset_id: str
    scale: Literal[2, 4] = 2
    denoise: float = Field(0.3, ge=0.0, le=1.0)
    tile_size: int = Field(1024, ge=256, le=4096)
    priority: Literal["normal", "high"] = "normal"
    metadata: dict[str, Any] = Field(default_factory=dict)


# ----- 视频 -----


class VideoGenerationRequest(BaseModel):
    model: str
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


# ----- 执行上下文 -----


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
    """'flux1-dev:fp16' -> ('flux1-dev', 'fp16')；无冒号则 version 为空"""
    if ":" in model_field:
        a, b = model_field.split(":", 1)
        return a.strip(), b.strip()
    return model_field.strip(), ""


def parse_size(size: str) -> tuple[int, int]:
    parts = size.lower().replace("×", "x").split("x")
    if len(parts) != 2:
        return 1024, 1024
    return int(parts[0]), int(parts[1])
