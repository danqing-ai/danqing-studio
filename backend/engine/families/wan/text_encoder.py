"""Wan 文本编码 — 对外入口（实现见 ``common.text_encoders.wan_umt5_mlx``）。"""
from __future__ import annotations

from backend.engine.common.text_encoders.wan_umt5_mlx import WanUMT5EncoderMLX, resolve_wan_umt5_pth

__all__ = ["WanUMT5EncoderMLX", "resolve_wan_umt5_pth"]
