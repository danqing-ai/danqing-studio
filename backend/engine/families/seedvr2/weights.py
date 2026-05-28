"""SeedVR2 bundle 元数据与权重加载 — 对外入口（MLX 实现见 ``weights_mlx``）。"""
from __future__ import annotations

from .weights_mlx import ModelConfig, load_flat_bundle

__all__ = ["ModelConfig", "load_flat_bundle"]
