"""HiDream-O1 stem — Shape C ``image_pipeline_shape=family_generator`` (no standard DiT load)."""

from __future__ import annotations

from typing import Any

from backend.engine.common.model.base import TransformerBase


class HiDreamO1Transformer(TransformerBase):
    """Placeholder stem; HiDream-O1 uses ``family_generator`` (Shape C)."""

    def __init__(self, config: Any, ctx: Any) -> None:
        self.config = config
        self.ctx = ctx

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(
            "HiDream-O1-Image does not use the standard image denoise loop; "
            "image_pipeline_shape=family_generator"
        )
