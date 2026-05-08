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
