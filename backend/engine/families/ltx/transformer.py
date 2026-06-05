"""LTX 2.3 Video Transformer — public entry (MLX only; CUDA fails loud)."""
from __future__ import annotations

from typing import Any

from backend.engine.common.dit_stem import DelegatingDiTStem


class LTXTransformer(DelegatingDiTStem):
    """LTX 2.3 joint A/V DiT — MLX ``LTX23Transformer`` only."""

    def __init__(self, config: Any, ctx: Any, num_frames: int = 33):
        from .transformer_mlx import LTX23Transformer

        super().__init__(
            config,
            ctx,
            mlx_cls=LTX23Transformer,
            cuda_cls=None,
            unavailable_product="LTX 2.3 Video",
            num_frames=num_frames,
        )
