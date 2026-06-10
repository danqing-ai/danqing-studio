"""Wan 文本编码 — 对外入口（backend dispatch）。"""
from __future__ import annotations

from backend.engine.runtime._base import RuntimeContext

from .text_encoder_mlx import WanUMT5EncoderMLX, resolve_wan_umt5_pth
from .text_encoder_cuda import WanUMT5EncoderCUDA


def WanUMT5Encoder(ctx: RuntimeContext, checkpoint_path: str, tokenizer_path: str, *, text_len: int = 512):
    """Return MLX or CUDA UMT5 encoder based on runtime backend."""
    if getattr(ctx, "backend", "mlx") == "cuda":
        return WanUMT5EncoderCUDA(ctx, checkpoint_path, tokenizer_path, text_len=text_len)
    return WanUMT5EncoderMLX(ctx, checkpoint_path, tokenizer_path, text_len=text_len)


__all__ = ["WanUMT5Encoder", "WanUMT5EncoderMLX", "WanUMT5EncoderCUDA", "resolve_wan_umt5_pth"]
