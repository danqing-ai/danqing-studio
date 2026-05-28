"""Flux.1 DiT — CUDA 占位（尚未实现）。"""
from __future__ import annotations

from typing import Any

from backend.engine.common._base import TransformerBase

_CUDA_DEFERRED = (
    "Flux.1 CUDA DiT 尚未实现；请在 models_registry 的 backends 中使用 mlx，"
    "或在 Apple Silicon 上使用 MLX 路径。"
)


class Flux1TransformerCuda(TransformerBase):
    def __init__(self, config: Any, ctx: Any):
        super().__init__()
        del config, ctx
        raise RuntimeError(_CUDA_DEFERRED)
