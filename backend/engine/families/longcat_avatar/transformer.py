"""LongCat-Avatar Transformer stem — MLX-only family_avatar."""
from __future__ import annotations

from typing import Any

from backend.engine.common.model.base import TransformerBase


class LongCatAvatarTransformer(TransformerBase):
    def __init__(self, config: Any, ctx: Any, num_frames: int = 93):
        self.config = config
        self.ctx = ctx
        self.num_frames = num_frames

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(
            "LongCat-Avatar does not use the standard video denoise loop; "
            "video_pipeline_shape=family_avatar"
        )
