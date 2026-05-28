"""LTX Video Transformer — 对外入口（MLX / CUDA dispatch）。"""
from __future__ import annotations

from typing import Any

from backend.engine.common.dit_stem import DelegatingDiTStem


class LTXTransformer(DelegatingDiTStem):
    """LTX Video DiT — selects MLX or CUDA implementation from ``RuntimeContext``."""

    def __init__(self, config: Any, ctx: Any, num_frames: int = 33):
        from .transformer_mlx import LTXTransformer as _MLX

        super().__init__(
            config,
            ctx,
            mlx_cls=_MLX,
            unavailable_product="LTX Video",
            num_frames=num_frames,
        )
