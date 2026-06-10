"""CogView4 GLM-4 text encoder facade — MLX native; CUDA via ``text_encoder_cuda``."""
from __future__ import annotations

from typing import Any

from backend.engine.families.cogview4.text_encoder_mlx import CogView4TextEncoder as _CogView4TextEncoderMlx

__all__ = ["CogView4TextEncoder"]


class CogView4TextEncoder(_CogView4TextEncoderMlx):
    """Registry entry — CUDA encode stays outside ``*_mlx.py``."""

    def encode(self, texts: list[str]) -> Any:
        if getattr(self.ctx, "backend", None) == "mlx":
            return super().encode(texts)
        from backend.engine.families.cogview4.text_encoder_cuda import cogview4_encode_cuda

        return cogview4_encode_cuda(self, texts)
