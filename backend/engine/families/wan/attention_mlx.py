"""Wan video attention — MLX scaled dot-product (ported from Wan2.2-mlx reference)."""
from __future__ import annotations

from typing import Any

import mlx.core as mx

from backend.engine.runtime._base import RuntimeContext

__all__ = ["wan_attention"]


def _seq_lens_from_grid_sizes(grid_sizes: mx.array) -> mx.array:
    g = grid_sizes.astype(mx.int64)
    return (g[:, 0] * g[:, 1] * g[:, 2]).astype(mx.int32)


def _build_key_padding_mask(
    ctx: RuntimeContext,
    k_lens: mx.array,
    seq_len: int,
    dtype: Any,
) -> mx.array:
    """Mask keys ``j >= k_lens[b]`` (Wan ``flash_attention`` ``k_lens`` semantics)."""
    del ctx  # reserved for symmetry with other families
    b = int(k_lens.shape[0])
    positions = mx.arange(seq_len, dtype=mx.int32)
    valid_k = positions.reshape(1, 1, 1, seq_len) < k_lens.reshape(b, 1, 1, 1)
    shape = (b, 1, seq_len, seq_len)
    neg = mx.full(shape, -1e9, dtype=dtype)
    return mx.where(valid_k, mx.zeros(shape, dtype=dtype), neg)


def wan_attention(
    ctx: RuntimeContext,
    q: Any,
    k: Any,
    v: Any,
    *,
    q_lens: Any | None = None,
    k_lens: Any | None = None,
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
    q = q.astype(target_dtype)
    k = k.astype(target_dtype)
    v = v.astype(target_dtype)

    # [B, L, H, D] -> [B, H, L, D] for ctx.attention
    q = ctx.permute(q, (0, 2, 1, 3))
    k = ctx.permute(k, (0, 2, 1, 3))
    v = ctx.permute(v, (0, 2, 1, 3))

    scale = softmax_scale
    if scale is None:
        scale = float(q.shape[-1]) ** -0.5

    seq_len = int(q.shape[2])
    mask = None
    if causal:
        import mlx.nn as nn

        mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
    elif k_lens is not None:
        mask = _build_key_padding_mask(ctx, k_lens.astype(mx.int32), seq_len, target_dtype)
    elif q_lens is not None:
        mask = _build_key_padding_mask(ctx, q_lens.astype(mx.int32), seq_len, target_dtype)

    out = ctx.attention(q, k, v, scale=scale, mask=mask)
    return ctx.permute(out, (0, 2, 1, 3))
