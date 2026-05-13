"""Qwen-Image 文本编码 — 对外入口；MLX 实现见 ``text_encoder_mlx``。"""
from __future__ import annotations

from .text_encoder_mlx import QwenImageTextEncoder

__all__ = ["QwenImageTextEncoder"]
