"""
归一化层 — 薄封装 RuntimeContext 的原生归一化模块。

所有模型经由 RuntimeContext 创建归一化层，保证 MLX/CUDA 互操作。
"""
from __future__ import annotations

from typing import Any


def RMSNorm(dims: int, eps: float = 1e-6, ctx: Any = None) -> Any:
    """RMS 归一化。"""
    return ctx.RMSNorm(dims, eps=eps)


def LayerNorm(dims: int, eps: float = 1e-5, ctx: Any = None) -> Any:
    """Layer 归一化。"""
    return ctx.LayerNorm(dims, eps=eps)


def GroupNorm(num_groups: int, num_channels: int, eps: float = 1e-5, ctx: Any = None) -> Any:
    """Group 归一化。"""
    return ctx.GroupNorm(num_groups, num_channels, eps=eps)


class AdaLayerNorm:
    """自适应 LayerNorm (AdaLN)：condition → scale + shift。"""

    def __init__(self, dim: int, ctx: Any, eps: float = 1e-6):
        self.ctx = ctx
        nn = ctx
        self.norm = nn.LayerNorm(dim, eps=eps, affine=False)
        self.scale = nn.Linear(dim, dim)
        self.shift = nn.Linear(dim, dim)

    def forward(self, x, condition):
        x = self.norm(x)
        scale = self.scale(condition)
        shift = self.shift(condition)
        return x * (1 + scale[:, None, :]) + shift[:, None, :]


class AdaLayerNormContinuous:
    """连续自适应 LayerNorm — 与 mflux AdaLayerNormContinuous 一致。

    使用单个 linear 层输出 scale + shift。
    """

    def __init__(self, embedding_dim: int, conditioning_embedding_dim: int, ctx: Any):
        self.ctx = ctx
        nn = ctx
        self.embedding_dim = embedding_dim
        self.linear = nn.Linear(conditioning_embedding_dim, embedding_dim * 2, bias=False)
        self.norm = nn.LayerNorm(embedding_dim, eps=1e-6, affine=False)

    def forward(self, x: Any, text_embeddings: Any) -> Any:
        import mlx.core as mx
        import mlx.nn as nn
        text_embeddings = self.linear(nn.silu(text_embeddings))
        chunk_size = self.embedding_dim
        scale = text_embeddings[:, 0 * chunk_size: 1 * chunk_size]
        shift = text_embeddings[:, 1 * chunk_size: 2 * chunk_size]
        x = self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]
        return x
