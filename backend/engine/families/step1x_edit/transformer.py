"""Step1X-Edit stem — Shape C ``image_pipeline_shape=family_generator``."""

from __future__ import annotations

from typing import Any

from backend.engine.common.model.base import TransformerBase


class Step1XEditTransformer(TransformerBase):
    """Placeholder stem; Step1X-Edit uses ``family_generator`` (Shape C)."""

    def __init__(self, config: Any, ctx: Any) -> None:
        self.config = config
        self.ctx = ctx

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(
            "Step1X-Edit does not use the standard image denoise loop; "
            "image_pipeline_shape=family_generator"
        )
