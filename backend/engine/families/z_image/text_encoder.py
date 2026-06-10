"""Z-Image Text Encoder — MLX 自研栈；CUDA 前向见 ``text_encoder_cuda``。"""
from __future__ import annotations

from typing import Any

from .text_encoder_mlx import ZImageTextEncoder as _ZImageTextEncoderMlx

__all__ = ["ZImageTextEncoder"]


class ZImageTextEncoder(_ZImageTextEncoderMlx):
    """Registry entry — CUDA backend uses ``text_encoder_cuda`` without importing it from ``*_mlx.py``."""

    def __new__(
        cls,
        ctx: Any,
        model_path: str,
        max_seq_len: int = 512,
        tokenizer_path: str = "",
        **kw: Any,
    ):
        if getattr(ctx, "backend", None) == "cuda":
            from backend.engine.families.z_image.text_encoder_cuda import ZImageTextEncoderCuda

            return ZImageTextEncoderCuda(
                ctx,
                model_path,
                max_seq_len=max_seq_len,
                tokenizer_path=tokenizer_path,
                **kw,
            )
        return super().__new__(cls)
