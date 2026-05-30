"""LongCat Transformer — 对外入口（MLX dispatch）。"""
from __future__ import annotations

from typing import Any

from backend.engine.common.dit_stem import DelegatingDiTStem


class LongCatTransformer(DelegatingDiTStem):
    """LongCat-Image MM-DiT — MLX only today."""

    def __init__(self, config: Any, ctx: Any):
        from .transformer_mlx import LongCatTransformer as _MLX

        super().__init__(
            config,
            ctx,
            mlx_cls=_MLX,
            unavailable_product="LongCat-Image",
        )
