"""Qwen-Image DiT — 对外入口（MLX / CUDA dispatch）。"""
from __future__ import annotations

from typing import Any

from backend.engine.common.dit_stem import DelegatingDiTStem


class QwenImageTransformer(DelegatingDiTStem):
    """Qwen-Image DiT — selects MLX or CUDA implementation from ``RuntimeContext``."""

    def __init__(self, config: Any, ctx: Any):
        from .transformer_mlx import QwenImageTransformer as _MLX

        cuda_cls = None
        if getattr(ctx, "backend", "mlx") == "cuda":
            from .transformer_cuda import QwenImageTransformerCuda

            cuda_cls = QwenImageTransformerCuda

        super().__init__(
            config,
            ctx,
            mlx_cls=_MLX,
            cuda_cls=cuda_cls,
        )

    @property
    def dit(self):
        return getattr(self._inner, "dit", None)
