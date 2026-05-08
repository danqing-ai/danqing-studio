"""
激活函数 — 模型通用激活。

薄封装 RuntimeContext 原生函数。
"""
from __future__ import annotations

from typing import Any


def silu(x: Any, ctx: Any = None) -> Any:
    """SiLU / Swish 激活。"""
    return ctx.silu(x)


def gelu(x: Any, approximate: str = "none", ctx: Any = None) -> Any:
    """GELU 激活。

    approximate 选项：
    - "none": 精确 GELU
    - "tanh": tanh 近似（MLX 默认）
    """
    return ctx.gelu(x, approximate=approximate)
