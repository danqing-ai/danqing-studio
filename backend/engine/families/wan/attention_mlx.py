"""Wan video attention — MLX scaled dot-product (ported from Wan2.2-mlx reference)."""
from __future__ import annotations

from typing import Any

from backend.engine.common.attention import (
    attention_blhd,
    resolve_blhd_attention_mask,
)
from backend.engine.runtime._base import RuntimeContext

__all__ = ["wan_attention"]


def wan_attention(
    ctx: RuntimeContext,
    q: Any,
    k: Any,
    v: Any,
    *,
    q_lens: Any | None = None,
    k_lens: Any | None = None,
    mask: Any | None = None,
    softmax_scale: float | None = None,
    q_scale: float | None = None,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
    dtype: Any | None = None,
) -> Any:
    """Attention on ``[B, L, num_heads, head_dim]`` tensors (Wan layout)."""
    if window_size != (-1, -1):
        raise RuntimeError(
            f"Wan MLX attention does not support sliding window_size={window_size!r}"
        )

    if q_scale is not None:
        q = q * q_scale

    target_dtype = dtype if dtype is not None else getattr(q, "dtype", ctx.float32())

    scale = softmax_scale
    if scale is None:
        scale = float(q.shape[-1]) ** -0.5

    mask = resolve_blhd_attention_mask(
        ctx,
        q,
        mask=mask,
        causal=causal,
        q_lens=q_lens,
        k_lens=k_lens,
        dtype=target_dtype,
        neg_value=-1e9,
    )

    return attention_blhd(ctx, q, k, v, scale=scale, mask=mask, dtype=target_dtype)
