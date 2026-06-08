"""Z-Image Transformer — 对外入口（MLX / CUDA dispatch）。"""
from __future__ import annotations

from typing import Any

from backend.engine.common.model.dit_stem import DelegatingDiTStem


class ZImageTransformer(DelegatingDiTStem):
    """Z-Image DiT — MLX on both backends today (CUDA ctx runs MLX graph)."""

    def __init__(self, config: Any, ctx: Any):
        from .transformer_mlx import ZImageDiTMLX as _MLX

        super().__init__(
            config,
            ctx,
            mlx_cls=_MLX,
            cuda_cls=_MLX,
        )
