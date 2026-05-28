"""CogVideoX Transformer3D — 对外入口（MLX / CUDA dispatch）。"""
from __future__ import annotations

from typing import Any

from backend.engine.common.dit_stem import DelegatingDiTStem


class CogVideoXTransformer(DelegatingDiTStem):
    """CogVideoX DiT — selects MLX or CUDA implementation from ``RuntimeContext``."""

    def __init__(self, config: Any, ctx: Any, num_frames: int = 13):
        from .transformer_mlx import CogVideoXTransformer3D as _MLX

        super().__init__(
            config,
            ctx,
            mlx_cls=_MLX,
            unavailable_product="CogVideoX",
            num_frames=num_frames,
        )


CogVideoXTransformer3D = CogVideoXTransformer

__all__ = ["CogVideoXTransformer", "CogVideoXTransformer3D"]
