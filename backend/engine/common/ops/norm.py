"""
归一化层 — 薄封装 RuntimeContext 的原生归一化模块。

所有模型经由 RuntimeContext 创建归一化层，保证 MLX/CUDA 互操作。
"""
from __future__ import annotations

import importlib
from typing import Any


def split_last_dim_chunks(x: Any, num_chunks: int) -> tuple[Any, ...]:
    """Split tensor on last dim into equal chunks."""
    if int(num_chunks) <= 0:
        raise RuntimeError(f"num_chunks must be positive, got {num_chunks}")
    last = int(x.shape[-1])
    if last % int(num_chunks) != 0:
        raise RuntimeError(
            f"Cannot split last dim={last} into {num_chunks} equal chunk(s)."
        )
    c = last // int(num_chunks)
    return tuple(x[..., i * c:(i + 1) * c] for i in range(int(num_chunks)))


def unpack_modulation_6way(flat: Any) -> tuple[Any, Any, Any, Any, Any, Any]:
    """Unpack [*, 6*D] into (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)."""
    parts = split_last_dim_chunks(flat, 6)
    return parts[0], parts[1], parts[2], parts[3], parts[4], parts[5]


def unpack_modulation_4way(flat: Any) -> tuple[Any, Any, Any, Any]:
    """Unpack [*, 4*D] into (scale_msa, gate_msa, scale_mlp, gate_mlp)."""
    parts = split_last_dim_chunks(flat, 4)
    return parts[0], parts[1], parts[2], parts[3]


def unpack_modulation_2way(flat: Any) -> tuple[Any, Any]:
    """Unpack [*, 2*D] into two [*, D] tensors."""
    parts = split_last_dim_chunks(flat, 2)
    return parts[0], parts[1]


def unpack_modulation_3way(flat: Any) -> tuple[Any, Any, Any]:
    """Unpack [*, 3*D] into three [*, D] tensors."""
    parts = split_last_dim_chunks(flat, 3)
    return parts[0], parts[1], parts[2]


def unpack_modulation_table(table: Any, num_parts: int) -> tuple[Any, ...]:
    """Unpack ``[..., N, D]`` modulation table into ``N`` tensors of shape ``[..., D]``."""
    n = int(num_parts)
    if n <= 0:
        raise RuntimeError(f"num_parts must be positive, got {num_parts}")
    if len(table.shape) < 2 or int(table.shape[-2]) != n:
        raise RuntimeError(
            f"Expected modulation table shape [..., {n}, D], got {tuple(table.shape)}"
        )
    return tuple(table[..., i, :] for i in range(n))


def unpack_modulation_6table(table: Any) -> tuple[Any, Any, Any, Any, Any, Any]:
    """Unpack ``[..., 6, D]`` modulation table into six ``[..., D]`` tensors."""
    parts = unpack_modulation_table(table, 6)
    return parts[0], parts[1], parts[2], parts[3], parts[4], parts[5]


def unpack_modulation_2table(table: Any) -> tuple[Any, Any]:
    """Unpack ``[..., 2, D]`` modulation table into two ``[..., D]`` tensors."""
    parts = unpack_modulation_table(table, 2)
    return parts[0], parts[1]


def apply_scale_shift(x: Any, scale: Any, shift: Any, *, add_one: bool = True) -> Any:
    """Apply affine modulation ``x * (1 + scale) + shift`` (or ``x * scale + shift``)."""
    s = (1 + scale) if add_one else scale
    return x * s + shift


def apply_ada_layer_norm_continuous(
    x: Any,
    text_embeddings: Any,
    *,
    linear: Any,
    norm: Any,
    embedding_dim: int,
    silu: Any,
    pre_linear_dtype: Any | None = None,
) -> Any:
    """Shared AdaLayerNormContinuous forward math."""
    text_embeddings = silu(text_embeddings)
    if pre_linear_dtype is not None:
        text_embeddings = text_embeddings.astype(pre_linear_dtype)
    text_embeddings = linear(text_embeddings)
    chunk_size = embedding_dim
    scale = text_embeddings[:, :chunk_size]
    shift = text_embeddings[:, chunk_size:2 * chunk_size]
    return apply_scale_shift(norm(x), scale[:, None, :], shift[:, None, :], add_one=True)


def apply_ada_layer_norm_zero(
    x: Any,
    emb: Any,
    *,
    linear: Any,
    norm: Any,
    silu: Any,
) -> tuple[Any, Any, Any, Any, Any]:
    """Shared AdaLayerNormZero math (6-way modulation)."""
    e = linear(silu(emb))
    c = e.shape[-1] // 6
    shift_msa = e[:, :c]
    scale_msa = e[:, c:2 * c]
    gate_msa = e[:, 2 * c:3 * c]
    shift_mlp = e[:, 3 * c:4 * c]
    scale_mlp = e[:, 4 * c:5 * c]
    gate_mlp = e[:, 5 * c:6 * c]
    n = norm(x)
    n = apply_scale_shift(n, scale_msa[:, None, :], shift_msa[:, None, :], add_one=True)
    return n, gate_msa, shift_mlp, scale_mlp, gate_mlp


def apply_ada_layer_norm_zero_single(
    x: Any,
    emb: Any,
    *,
    linear: Any,
    norm: Any,
    silu: Any,
) -> tuple[Any, Any]:
    """Shared AdaLayerNormZeroSingle math (3-way modulation)."""
    e = linear(silu(emb))
    c = e.shape[-1] // 3
    shift_msa = e[:, :c]
    scale_msa = e[:, c:2 * c]
    gate = e[:, 2 * c:3 * c]
    n = norm(x)
    n = apply_scale_shift(n, scale_msa[:, None, :], shift_msa[:, None, :], add_one=True)
    return n, gate


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


def apply_rms_norm(x: Any, weight: Any, eps: float) -> Any:
    """Public helper for families needing explicit RMSNorm math."""
    return _rms_norm_apply(x, weight, eps)


def apply_layer_norm_fp32(layer: Any, x: Any) -> Any:
    """Run LayerNorm in fp32 then cast back to original dtype."""
    try:
        torch = importlib.import_module("torch")
        if isinstance(x, torch.Tensor):
            out = layer(x.float())
            return out.to(x.dtype)
    except ImportError:
        pass

    mx = importlib.import_module("mlx.core")
    out = layer(x.astype(mx.float32))
    return out.astype(x.dtype)


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
        return apply_ada_layer_norm_continuous(
            x,
            text_embeddings,
            linear=self.linear,
            norm=self.norm,
            embedding_dim=self.embedding_dim,
            silu=ctx.silu,
            pre_linear_dtype=None,
        )
