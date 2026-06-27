"""LongCat-Video Transformer stem — MLX-only family_generator (no standard DiT load)."""
from __future__ import annotations

from typing import Any

from backend.engine.common.model.base import TransformerBase


class LongCatTransformer(TransformerBase):
    """Placeholder stem; LongCat-Video uses ``family_generator`` (Shape C)."""

    def __init__(self, config: Any, ctx: Any, num_frames: int = 33):
        self.config = config
        self.ctx = ctx
        self.num_frames = num_frames

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(
            "LongCat-Video does not use the standard video denoise loop; "
            "video_pipeline_shape=family_generator"
        )
