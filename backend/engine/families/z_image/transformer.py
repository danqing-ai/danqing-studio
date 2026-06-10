"""Z-Image Transformer — 对外入口（MLX / CUDA dispatch）。"""
from __future__ import annotations

from typing import Any

from backend.engine.common.model.dit_stem import DelegatingDiTStem


class ZImageTransformer(DelegatingDiTStem):
    """Z-Image DiT — native PyTorch on CUDA, MLX on Apple Silicon."""

    def __init__(self, config: Any, ctx: Any):
        from .transformer_mlx import ZImageDiTMLX as _MLX

        cuda_cls = None
        if getattr(ctx, "backend", "mlx") == "cuda":
            from .transformer_cuda import ZImageDiTCuda
            cuda_cls = ZImageDiTCuda

        super().__init__(
            config,
            ctx,
            mlx_cls=_MLX,
            cuda_cls=cuda_cls,
        )
