"""Z-Image Text Encoder — 对外入口（MLX 自研栈 + CUDA HF Qwen3，见 ``text_encoder_mlx`` / ``text_encoder_cuda``）。"""
from __future__ import annotations

from .text_encoder_mlx import ZImageTextEncoder

__all__ = ["ZImageTextEncoder"]
