"""Flux2 Klein text encoder — 对外入口；MLX 实现见 ``text_encoder_mlx``。"""
from __future__ import annotations

from .text_encoder_mlx import Flux2TextEncoder

__all__ = ["Flux2TextEncoder"]
