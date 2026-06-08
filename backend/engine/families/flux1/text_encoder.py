"""Flux.1 双编码器 — 对外入口（T5 + CLIP 见族内 ``flux1_*_mlx``）。"""
from __future__ import annotations

from backend.engine.families.flux1.flux1_dual_mlx import Flux1TextEncoder

__all__ = ["Flux1TextEncoder"]
