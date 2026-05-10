"""Minimal generation result container for SeedVR2 upscale (no flux / concept deps)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
from PIL import Image

from backend.engine.seedvr2.runtime.common.config import ModelConfig


@dataclass
class GeneratedImage:
    image: Image.Image
    model_config: ModelConfig
    seed: int
    prompt: str
    steps: int
    guidance: float | None
    precision: mx.Dtype
    quantization: int
    generation_time: float
    lora_paths: list[str] | None = None
    lora_scales: list[float] | None = None
    height: int | None = None
    width: int | None = None
    controlnet_image_path: str | Path | None = None
    controlnet_strength: float | None = None
    image_path: str | Path | None = None
    image_paths: list[str] | list[Path] | None = None
    image_strength: float | None = None
    masked_image_path: str | Path | None = None
    depth_image_path: str | Path | None = None
    redux_image_paths: list[str] | list[Path] | None = None
    redux_image_strengths: list[float] | None = None
    concept_heatmap: Any = None
    negative_prompt: str | None = None
    init_metadata: dict | None = None
