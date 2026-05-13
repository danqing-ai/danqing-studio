"""Qwen-Image DiT — 对外入口；MLX 实现见 ``transformer_mlx``。"""
from __future__ import annotations

from .transformer_mlx import QwenImageTransformer

__all__ = ["QwenImageTransformer"]
