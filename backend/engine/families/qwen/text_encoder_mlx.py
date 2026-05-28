"""Deprecated path — use ``common.text_encoders.qwen_image_mlx``."""
from __future__ import annotations

from backend.engine.common.text_encoders.qwen_image_mlx import (
    QwenImageTextEncoder,
    load_qwen25vl_mlx_encoder,
)

__all__ = ["QwenImageTextEncoder", "load_qwen25vl_mlx_encoder"]
