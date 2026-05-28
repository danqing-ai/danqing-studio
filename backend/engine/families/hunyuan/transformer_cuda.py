"""Hunyuan Video DiT — CUDA 占位（尚未实现）。"""
from __future__ import annotations

from typing import Any

from backend.engine.common._base import TransformerBase

_CUDA_DEFERRED = (
    "Hunyuan Video CUDA DiT 尚未实现；请在 models_registry 的 backends 中使用 mlx。"
)


class HunyuanVideoTransformerCuda(TransformerBase):
    def __init__(self, config: Any, ctx: Any, num_frames: int = 129):
        super().__init__()
        del config, ctx, num_frames
        raise RuntimeError(_CUDA_DEFERRED)
