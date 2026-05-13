"""
归一化层 — 薄封装 RuntimeContext 的原生归一化模块。

所有模型经由 RuntimeContext 创建归一化层，保证 MLX/CUDA 互操作。
"""
from __future__ import annotations

import importlib
from typing import Any


def _rms_norm_apply(x: Any, weight: Any, eps: float) -> Any:
    """RMS norm over last dimension — matches ``mlx.nn.RMSNorm`` / ``_CudaRMSNorm``."""
    eps = float(eps)
    try:
        torch = importlib.import_module("torch")
        if isinstance(x, torch.Tensor):
            dtype = x.dtype
            xf = x.float()
            norm = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
            return (weight.float() * norm).to(dtype)
    except ImportError:
        pass

    mx = importlib.import_module("mlx.core")
    return mx.fast.rms_norm(x, weight, eps)


def RMSNorm(dims: int, eps: float = 1e-6, ctx: Any = None) -> Any:
    """RMS 归一化。"""
    return ctx.RMSNorm(dims, eps=eps)


RMSNorm._apply_norm = _rms_norm_apply  # type: ignore[attr-defined]


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
    """Continuous adaptive LayerNorm — matches reference AdaLayerNormContinuous.

    Uses a single linear layer to output scale + shift.
    """

    def __init__(self, embedding_dim: int, conditioning_embedding_dim: int, ctx: Any):
        self.ctx = ctx
        nn = ctx
        self.embedding_dim = embedding_dim
        self.linear = nn.Linear(conditioning_embedding_dim, embedding_dim * 2, bias=False)
        self.norm = nn.LayerNorm(embedding_dim, eps=1e-6, affine=False)

    def forward(self, x: Any, text_embeddings: Any) -> Any:
        ctx = self.ctx
        text_embeddings = self.linear(ctx.silu(text_embeddings))
        chunk_size = self.embedding_dim
        scale = text_embeddings[:, :chunk_size]
        shift = text_embeddings[:, chunk_size:2 * chunk_size]
        x = self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]
        return x
