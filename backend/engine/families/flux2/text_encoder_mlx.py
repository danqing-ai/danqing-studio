"""Deprecated import path — use ``text_encoder`` or ``common.text_encoders.qwen3_mlx``."""
from __future__ import annotations

from backend.engine.common.text_encoders.qwen3_mlx import Flux2TextEncoder

__all__ = ["Flux2TextEncoder"]
