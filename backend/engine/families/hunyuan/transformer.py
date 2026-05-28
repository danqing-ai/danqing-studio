"""HunyuanVideo-1.5 transformer — 对外入口（MLX / CUDA dispatch）。"""
from __future__ import annotations

from typing import Any

from backend.engine.common.dit_stem import DelegatingDiTStem


class HunyuanVideoTransformer(DelegatingDiTStem):
    """Hunyuan Video DiT — selects MLX or CUDA implementation from ``RuntimeContext``."""

    def __init__(self, config: Any, ctx: Any):
        from .transformer_mlx import HunyuanVideoTransformer as _MLX

        super().__init__(
            config,
            ctx,
            mlx_cls=_MLX,
            unavailable_product="Hunyuan Video",
        )


__all__ = ["HunyuanVideoTransformer"]
