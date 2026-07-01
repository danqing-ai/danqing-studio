"""Boogu-Image DiT stem — Shape C ``image_pipeline_shape=family_generator``."""

from __future__ import annotations

from typing import Any

from backend.engine.common.model.base import TransformerBase


class BooguImageTransformer(TransformerBase):
    """Placeholder stem; Boogu-Image uses ``family_generator`` (Shape C)."""

    def __init__(self, config: Any, ctx: Any):
        super().__init__(config, ctx)
        self.config = config
        self.ctx = ctx

    def forward(self, *args: Any, **kwargs: Any):
        raise RuntimeError(
            "Boogu-Image inference must run through family_generator "
            "(image_pipeline_shape=family_generator)"
        )
