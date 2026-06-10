"""Qwen-Image 文本编码 — 对外入口（MLX 实现见 ``text_encoder_mlx``）。"""
from __future__ import annotations

from typing import Any

from backend.engine.families.qwen.text_encoder_mlx import QwenImageTextEncoder as _QwenImageTextEncoderMlx

__all__ = ["QwenImageTextEncoder"]


class QwenImageTextEncoder(_QwenImageTextEncoderMlx):
    """Registry entry — CUDA backend uses ``text_encoder_cuda`` without importing it from ``*_mlx.py``."""

    def __new__(
        cls,
        ctx: Any,
        model_path: str | Any,
        tokenizer_path: str = "",
        **kw: Any,
    ):
        if getattr(ctx, "backend", None) == "cuda":
            from backend.engine.families.qwen.text_encoder_cuda import QwenImageTextEncoderCuda

            return QwenImageTextEncoderCuda(ctx, model_path, **kw)
        return super().__new__(cls)
