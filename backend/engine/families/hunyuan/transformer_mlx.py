"""
HunyuanVideo-1.5 Transformer3D — MLX port of diffusers ``HunyuanVideo15Transformer3DModel``.

Reference: ``diffusers/models/transformers/transformer_hunyuan_video15.py``
"""
from __future__ import annotations

import math
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from backend.engine.common._base import TransformerBase, _collect_params
from backend.engine.common.attention import _apply_rope
from backend.engine.common.cfg_batch import (
    HUNYUAN_CFG_TEXT_KEYS,
    broadcast_batch,
    predict_noise_cfg_batched,
)
from backend.engine.runtime._base import RuntimeContext


def _cfg_int(cfg: Any, name: str, default: int) -> int:
    return int(getattr(cfg, name, default))


def _cfg_float(cfg: Any, name: str, default: float) -> float:
    return float(getattr(cfg, name, default))


def _timesteps_proj(ctx: RuntimeContext, timesteps: Any, *, num_channels: int = 256) -> Any:
    """diffusers ``Timesteps`` (flip_sin_to_cos=True, downscale_freq_shift=0)."""
    timesteps = ctx.reshape(timesteps.astype(ctx.float32()), (-1,))
    half = num_channels // 2
    exp_arg = -math.log(10000.0) * ctx.arange(half, dtype=ctx.float32()) / float(half)
    emb = timesteps[:, None] * ctx.exp(exp_arg)[None, :]
    emb = ctx.concat([ctx.cos(emb), ctx.sin(emb)], axis=-1)
    return ctx.concat([emb[:, half:], emb[:, :half]], axis=-1)


class _TimestepEmbedding:
    """diffusers ``TimestepEmbedding``: linear_1 → SiLU → linear_2."""

    def __init__(self, ctx: RuntimeContext, in_channels: int, time_embed_dim: int):
        self.linear_1 = ctx.Linear(in_channels, time_embed_dim, bias=True)
        self.act = nn.SiLU()
        self.linear_2 = ctx.Linear(time_embed_dim, time_embed_dim, bias=True)

    def __call__(self, x: Any) -> Any:
        x = self.linear_1(x)
        x = self.act(x)
        return self.linear_2(x)


class _PixArtAlphaTextProjection:
    """``CombinedTimestepTextProjEmbeddings.text_embedder``."""

    def __init__(self, ctx: RuntimeContext, in_features: int, hidden_size: int):
        self.linear_1 = ctx.Linear(in_features, hidden_size, bias=True)
        self.act = nn.SiLU()
        self.linear_2 = ctx.Linear(hidden_size, hidden_size, bias=True)

    def __call__(self, x: Any) -> Any:
        x = self.linear_1(x)
        x = self.act(x)
        return self.linear_2(x)


class _CombinedTimestepTextProjEmbeddings:
    def __init__(self, ctx: RuntimeContext, embedding_dim: int, pooled_projection_dim: int):
        self.ctx = ctx
        self.timestep_embedder = _TimestepEmbedding(ctx, 256, embedding_dim)
        self.text_embedder = _PixArtAlphaTextProjection(ctx, pooled_projection_dim, embedding_dim)

    def __call__(self, timestep: Any, pooled_projection: Any) -> Any:
        ctx = self.ctx
        t_proj = _timesteps_proj(ctx, timestep)
        t_emb = self.timestep_embedder(t_proj)
        p_emb = self.text_embedder(pooled_projection)
        return t_emb + p_emb


class _HunyuanVideo15PatchEmbed:
    """``x_embedder`` — Conv3d patchify (diffusers ``HunyuanVideo15PatchEmbed``)."""

    def __init__(
        self,
        ctx: RuntimeContext,
        patch_size: tuple[int, int, int],
        in_channels: int,
        embed_dim: int,
    ):
        self.ctx = ctx
        self.proj = ctx.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True,
        )

    def __call__(self, hidden_states: Any) -> Any:
        ctx = self.ctx
        x = ctx.permute(hidden_states, (0, 2, 3, 4, 1))
        x = self.proj(x)
        B, T, H, W, C = x.shape
        return ctx.reshape(x, (B, T * H * W, C))


class _HunyuanVideo15AdaNorm:
    """Refiner block gate — ``norm_out`` in ``IndividualTokenRefinerBlock``."""

    def __init__(self, ctx: RuntimeContext, in_features: int, out_features: int):
        self.linear = ctx.Linear(in_features, out_features, bias=True)
        self.nonlinearity = nn.SiLU()

    def __call__(self, temb: Any) -> tuple[Any, Any]:
        v = self.linear(self.nonlinearity(temb))
        D = v.shape[-1] // 2
        gate_msa = v[:, :D][:, None, :]
        gate_mlp = v[:, D:][:, None, :]
        return gate_msa, gate_mlp


class _FeedForwardLinearSilu:
    """diffusers ``FeedForward`` with ``activation_fn='linear-silu'``."""

    def __init__(self, ctx: RuntimeContext, dim: int, mult: float = 4.0):
        inner = int(dim * mult)
        self.net = [
            ctx.Linear(dim, inner, bias=True),
            nn.SiLU(),
            ctx.Linear(inner, dim, bias=True),
        ]

    def __call__(self, x: Any) -> Any:
        x = self.net[0](x)
        x = self.net[1](x)
        return self.net[2](x)


class _FeedForwardGeluApprox:
    """diffusers ``FeedForward`` with ``activation_fn='gelu-approximate'``."""

    def __init__(self, ctx: RuntimeContext, dim: int, mult: float = 4.0):
        inner = int(dim * mult)
        self.net = [
            ctx.Linear(dim, inner, bias=True),
            ctx.GELU(approximate="tanh"),
            ctx.Linear(inner, dim, bias=True),
        ]

    def __call__(self, x: Any) -> Any:
        x = self.net[0](x)
        x = self.net[1](x)
        return self.net[2](x)


class _RefinerSelfAttention:
    """Self-attention in ``IndividualTokenRefinerBlock`` (no cross / added KV)."""

    def __init__(self, ctx: RuntimeContext, dim: int, heads: int, head_dim: int):
        self.ctx = ctx
        self.heads = heads
        self.head_dim = head_dim
        self.inner_dim = heads * head_dim
        self.scale = head_dim ** -0.5
        self.to_q = ctx.Linear(dim, self.inner_dim, bias=True)
        self.to_k = ctx.Linear(dim, self.inner_dim, bias=True)
        self.to_v = ctx.Linear(dim, self.inner_dim, bias=True)
        self.to_out = [ctx.Linear(self.inner_dim, dim, bias=True), ctx.Dropout(0.0)]

    def __call__(self, hidden_states: Any, attn_mask: Any | None = None) -> Any:
        ctx = self.ctx
        B, S, _ = hidden_states.shape
        q = self.to_q(hidden_states)
        k = self.to_k(hidden_states)
        v = self.to_v(hidden_states)
        q = ctx.reshape(q, (B, S, self.heads, self.head_dim))
        q = ctx.permute(q, (0, 2, 1, 3))
        k = ctx.reshape(k, (B, S, self.heads, self.head_dim))
        k = ctx.permute(k, (0, 2, 1, 3))
        v = ctx.reshape(v, (B, S, self.heads, self.head_dim))
        v = ctx.permute(v, (0, 2, 1, 3))
        out = ctx.attention(q, k, v, scale=self.scale, mask=attn_mask)
        out = ctx.permute(out, (0, 2, 1, 3))
        out = ctx.reshape(out, (B, S, self.inner_dim))
        out = self.to_out[0](out)
        return self.to_out[1](out)


class _IndividualTokenRefinerBlock:
    def __init__(
        self,
        ctx: RuntimeContext,
        num_heads: int,
        head_dim: int,
        mlp_ratio: float = 4.0,
    ):
        hidden = num_heads * head_dim
        self.norm1 = ctx.LayerNorm(hidden, eps=1e-6, affine=True, bias=True)
        self.attn = _RefinerSelfAttention(ctx, hidden, num_heads, head_dim)
        self.norm2 = ctx.LayerNorm(hidden, eps=1e-6, affine=True, bias=True)
        self.ff = _FeedForwardLinearSilu(ctx, hidden, mult=mlp_ratio)
        self.norm_out = _HunyuanVideo15AdaNorm(ctx, hidden, 2 * hidden)

    def __call__(
        self,
        hidden_states: Any,
        temb: Any,
        attn_mask: Any | None = None,
    ) -> Any:
        norm_h = self.norm1(hidden_states)
        attn_out = self.attn(norm_h, attn_mask)
        gate_msa, gate_mlp = self.norm_out(temb)
        hidden_states = hidden_states + attn_out * gate_msa
        ff_out = self.ff(self.norm2(hidden_states))
        return hidden_states + ff_out * gate_mlp


class _IndividualTokenRefiner:
    def __init__(
        self,
        ctx: RuntimeContext,
        num_heads: int,
        head_dim: int,
        num_layers: int,
        mlp_ratio: float = 4.0,
    ):
        self.ctx = ctx
        self.refiner_blocks = [
            _IndividualTokenRefinerBlock(ctx, num_heads, head_dim, mlp_ratio)
            for _ in range(num_layers)
        ]

    def __call__(
        self,
        hidden_states: Any,
        temb: Any,
        attention_mask: Any | None = None,
    ) -> Any:
        attn_mask = _build_refiner_attn_mask(self.ctx, attention_mask)
        for block in self.refiner_blocks:
            hidden_states = block(hidden_states, temb, attn_mask)
        return hidden_states


class _TokenRefiner:
    """``context_embedder``."""

    def __init__(
        self,
        ctx: RuntimeContext,
        in_channels: int,
        num_heads: int,
        head_dim: int,
        num_layers: int,
        mlp_ratio: float = 4.0,
    ):
        hidden = num_heads * head_dim
        self.ctx = ctx
        self.time_text_embed = _CombinedTimestepTextProjEmbeddings(ctx, hidden, in_channels)
        self.proj_in = ctx.Linear(in_channels, hidden, bias=True)
        self.token_refiner = _IndividualTokenRefiner(
            ctx, num_heads, head_dim, num_layers, mlp_ratio,
        )

    def __call__(
        self,
        hidden_states: Any,
        timestep: Any,
        attention_mask: Any | None = None,
    ) -> Any:
        ctx = self.ctx
        if attention_mask is None:
            pooled = ctx.mean(hidden_states, axis=1)
        else:
            mask_f = attention_mask.astype(ctx.float32())[:, :, None]
            pooled = ctx.sum(hidden_states * mask_f, axis=1) / mx.maximum(
                ctx.sum(mask_f, axis=1), 1e-6,
            )
        temb = self.time_text_embed(timestep, pooled)
        hidden_states = self.proj_in(hidden_states)
        return self.token_refiner(hidden_states, temb, attention_mask)


class _ByT5TextProjection:
    """``context_embedder_2``."""

    def __init__(self, ctx: RuntimeContext, in_features: int, hidden_size: int, out_features: int):
        self.norm = ctx.LayerNorm(in_features, eps=1e-6, affine=True, bias=True)
        self.linear_1 = ctx.Linear(in_features, hidden_size, bias=True)
        self.linear_2 = ctx.Linear(hidden_size, hidden_size, bias=True)
        self.linear_3 = ctx.Linear(hidden_size, out_features, bias=True)
        self.act_fn = ctx.GELU()

    def __call__(self, x: Any) -> Any:
        x = self.norm(x)
        x = self.linear_1(x)
        x = self.act_fn(x)
        x = self.linear_2(x)
        x = self.act_fn(x)
        return self.linear_3(x)


class _ImageProjection:
    """``image_embedder``."""

    def __init__(self, ctx: RuntimeContext, in_channels: int, hidden_size: int):
        self.norm_in = ctx.LayerNorm(in_channels, eps=1e-6, affine=True, bias=True)
        self.linear_1 = ctx.Linear(in_channels, in_channels, bias=True)
        self.act_fn = ctx.GELU()
        self.linear_2 = ctx.Linear(in_channels, hidden_size, bias=True)
        self.norm_out = ctx.LayerNorm(hidden_size, eps=1e-6, affine=True, bias=True)

    def __call__(self, x: Any) -> Any:
        x = self.norm_in(x)
        x = self.linear_1(x)
        x = self.act_fn(x)
        x = self.linear_2(x)
        return self.norm_out(x)


class _TimeEmbedding:
    """``time_embed``."""

    def __init__(self, ctx: RuntimeContext, embedding_dim: int, use_meanflow: bool = False):
        self.ctx = ctx
        self.timestep_embedder = _TimestepEmbedding(ctx, 256, embedding_dim)
        self.use_meanflow = use_meanflow
        self.timestep_embedder_r = (
            _TimestepEmbedding(ctx, 256, embedding_dim) if use_meanflow else None
        )

    def __call__(self, timestep: Any, timestep_r: Any | None = None) -> Any:
        ctx = self.ctx
        emb = self.timestep_embedder(_timesteps_proj(ctx, timestep))
        if timestep_r is not None and self.timestep_embedder_r is not None:
            emb = emb + self.timestep_embedder_r(_timesteps_proj(ctx, timestep_r))
        return emb


def _get_1d_rotary_pos_embed_np(dim: int, pos: np.ndarray, theta: float) -> tuple[np.ndarray, np.ndarray]:
    if dim % 2 != 0:
        raise ValueError(f"RoPE dim must be even, got {dim}")
    idx = np.arange(0, dim, 2, dtype=np.float64)
    inv = theta ** (-idx / float(dim))
    freqs = pos.reshape(-1, 1) * inv.reshape(1, -1)
    cos = np.cos(freqs)
    sin = np.sin(freqs)
    cos = np.repeat(cos, 2, axis=1).astype(np.float32)
    sin = np.repeat(sin, 2, axis=1).astype(np.float32)
    return cos, sin


class _RotaryPosEmbed:
    """``rope`` — 3-axis RoPE cos/sin for latent tokens."""

    def __init__(
        self,
        patch_size: int,
        patch_size_t: int,
        rope_dim: tuple[int, ...],
        theta: float = 256.0,
    ):
        self.patch_size = patch_size
        self.patch_size_t = patch_size_t
        self.rope_dim = rope_dim
        self.theta = theta

    def __call__(self, hidden_states: Any) -> tuple[Any, Any]:
        _, _, num_frames, height, width = hidden_states.shape
        rope_sizes = [
            num_frames // self.patch_size_t,
            height // self.patch_size,
            width // self.patch_size,
        ]
        grids = [
            np.arange(0, rope_sizes[i], dtype=np.float32)
            for i in range(3)
        ]
        mesh = np.meshgrid(grids[0], grids[1], grids[2], indexing="ij")
        cos_parts: list[np.ndarray] = []
        sin_parts: list[np.ndarray] = []
        for i in range(3):
            c, s = _get_1d_rotary_pos_embed_np(self.rope_dim[i], mesh[i].reshape(-1), self.theta)
            cos_parts.append(c)
            sin_parts.append(s)
        cos = np.concatenate(cos_parts, axis=1)
        sin = np.concatenate(sin_parts, axis=1)
        return mx.array(cos), mx.array(sin)


class _AdaLayerNormZero:
    """diffusers ``AdaLayerNormZero`` (Flux-style 6-way modulation)."""

    def __init__(self, ctx: RuntimeContext, dim: int):
        self.linear = ctx.Linear(dim, dim * 6, bias=True)
        self.norm = nn.LayerNorm(dim, eps=1e-6, affine=False)

    def __call__(self, x: Any, emb: Any) -> tuple[Any, Any, Any, Any, Any]:
        e = self.linear(nn.silu(emb))
        c = e.shape[-1] // 6
        shift_msa = e[:, 0 * c : 1 * c]
        scale_msa = e[:, 1 * c : 2 * c]
        gate_msa = e[:, 2 * c : 3 * c]
        shift_mlp = e[:, 3 * c : 4 * c]
        scale_mlp = e[:, 4 * c : 5 * c]
        gate_mlp = e[:, 5 * c : 6 * c]
        n = self.norm(x)
        n = n * (1 + scale_msa[:, None, :]) + shift_msa[:, None, :]
        return n, gate_msa, shift_mlp, scale_mlp, gate_mlp


class _JointAttention:
    """``HunyuanVideo15AttnProcessor2_0`` — image stream first, then text."""

    def __init__(self, ctx: RuntimeContext, dim: int, heads: int, head_dim: int):
        self.ctx = ctx
        self.heads = heads
        self.head_dim = head_dim
        self.inner_dim = heads * head_dim
        self.scale = head_dim ** -0.5
        self.to_q = ctx.Linear(dim, self.inner_dim, bias=True)
        self.to_k = ctx.Linear(dim, self.inner_dim, bias=True)
        self.to_v = ctx.Linear(dim, self.inner_dim, bias=True)
        self.add_q_proj = ctx.Linear(dim, self.inner_dim, bias=True)
        self.add_k_proj = ctx.Linear(dim, self.inner_dim, bias=True)
        self.add_v_proj = ctx.Linear(dim, self.inner_dim, bias=True)
        self.norm_q = ctx.RMSNorm(head_dim)
        self.norm_k = ctx.RMSNorm(head_dim)
        self.norm_added_q = ctx.RMSNorm(head_dim)
        self.norm_added_k = ctx.RMSNorm(head_dim)
        self.to_out = [ctx.Linear(self.inner_dim, dim, bias=True), ctx.Dropout(0.0)]
        self.to_add_out = ctx.Linear(self.inner_dim, dim, bias=True)

    def __call__(
        self,
        hidden_states: Any,
        encoder_hidden_states: Any,
        attention_mask: Any | None = None,
        image_rotary_emb: tuple[Any, Any] | None = None,
    ) -> tuple[Any, Any]:
        ctx = self.ctx
        B, img_len, _ = hidden_states.shape
        txt_len = int(encoder_hidden_states.shape[1])

        q = self.to_q(hidden_states)
        k = self.to_k(hidden_states)
        v = self.to_v(hidden_states)
        q = ctx.reshape(q, (B, img_len, self.heads, self.head_dim))
        q = ctx.permute(q, (0, 2, 1, 3))
        k = ctx.reshape(k, (B, img_len, self.heads, self.head_dim))
        k = ctx.permute(k, (0, 2, 1, 3))
        v = ctx.reshape(v, (B, img_len, self.heads, self.head_dim))
        v = ctx.permute(v, (0, 2, 1, 3))
        q = self.norm_q(q)
        k = self.norm_k(k)

        if image_rotary_emb is not None:
            cos, sin = image_rotary_emb
            cos_b = mx.reshape(cos, (1, 1, img_len, -1))
            sin_b = mx.reshape(sin, (1, 1, img_len, -1))
            q = _apply_rope(ctx, q, cos_b, sin_b)
            k = _apply_rope(ctx, k, cos_b, sin_b)

        eq = self.add_q_proj(encoder_hidden_states)
        ek = self.add_k_proj(encoder_hidden_states)
        ev = self.add_v_proj(encoder_hidden_states)
        eq = ctx.reshape(eq, (B, txt_len, self.heads, self.head_dim))
        eq = ctx.permute(eq, (0, 2, 1, 3))
        ek = ctx.reshape(ek, (B, txt_len, self.heads, self.head_dim))
        ek = ctx.permute(ek, (0, 2, 1, 3))
        ev = ctx.reshape(ev, (B, txt_len, self.heads, self.head_dim))
        ev = ctx.permute(ev, (0, 2, 1, 3))
        eq = self.norm_added_q(eq)
        ek = self.norm_added_k(ek)

        q = ctx.concat([q, eq], axis=2)
        k = ctx.concat([k, ek], axis=2)
        v = ctx.concat([v, ev], axis=2)

        total_len = img_len + txt_len
        mask = _build_joint_attn_mask(ctx, attention_mask, img_len, total_len)

        out = ctx.attention(q, k, v, scale=self.scale, mask=mask)
        out = ctx.permute(out, (0, 2, 1, 3))
        out = ctx.reshape(out, (B, total_len, self.inner_dim))

        hid = out[:, :img_len, :]
        enc = out[:, img_len:, :]
        hid = self.to_out[0](hid)
        hid = self.to_out[1](hid)
        enc = self.to_add_out(enc)
        return hid, enc


class _TransformerBlock:
    """``HunyuanVideo15TransformerBlock``."""

    def __init__(self, ctx: RuntimeContext, num_heads: int, head_dim: int, mlp_ratio: float):
        hidden = num_heads * head_dim
        self.norm1 = _AdaLayerNormZero(ctx, hidden)
        self.norm1_context = _AdaLayerNormZero(ctx, hidden)
        self.attn = _JointAttention(ctx, hidden, num_heads, head_dim)
        self.norm2 = ctx.LayerNorm(hidden, eps=1e-6, affine=False, bias=False)
        self.ff = _FeedForwardGeluApprox(ctx, hidden, mult=mlp_ratio)
        self.norm2_context = ctx.LayerNorm(hidden, eps=1e-6, affine=False, bias=False)
        self.ff_context = _FeedForwardGeluApprox(ctx, hidden, mult=mlp_ratio)

    def __call__(
        self,
        hidden_states: Any,
        encoder_hidden_states: Any,
        temb: Any,
        attention_mask: Any | None = None,
        freqs_cis: tuple[Any, Any] | None = None,
    ) -> tuple[Any, Any]:
        n_h, g_msa, s_mlp, sc_mlp, g_mlp = self.norm1(hidden_states, temb)
        n_e, c_g_msa, c_s_mlp, c_sc_mlp, c_g_mlp = self.norm1_context(encoder_hidden_states, temb)

        ah, ae = self.attn(n_h, n_e, attention_mask, freqs_cis)
        hidden_states = hidden_states + ah * g_msa[:, None, :]
        encoder_hidden_states = encoder_hidden_states + ae * c_g_msa[:, None, :]

        n_h2 = self.norm2(hidden_states)
        n_e2 = self.norm2_context(encoder_hidden_states)
        n_h2 = n_h2 * (1 + sc_mlp[:, None, :]) + s_mlp[:, None, :]
        n_e2 = n_e2 * (1 + c_sc_mlp[:, None, :]) + c_s_mlp[:, None, :]

        hidden_states = hidden_states + g_mlp[:, None, :] * self.ff(n_h2)
        encoder_hidden_states = encoder_hidden_states + c_g_mlp[:, None, :] * self.ff_context(n_e2)
        return hidden_states, encoder_hidden_states


class _AdaLayerNormContinuous:
    """``norm_out``."""

    def __init__(self, ctx: RuntimeContext, dim: int):
        self.silu = nn.SiLU()
        self.linear = ctx.Linear(dim, dim * 2, bias=True)
        self.norm = nn.LayerNorm(dim, eps=1e-6, affine=False)

    def __call__(self, x: Any, temb: Any) -> Any:
        v = self.linear(self.silu(temb))
        half = v.shape[-1] // 2
        shift = v[:, :half]
        scale = v[:, half:]
        x = self.norm(x)
        return x * (1 + scale[:, None, :]) + shift[:, None, :]


def _build_refiner_attn_mask(ctx: RuntimeContext | None, attention_mask: Any | None) -> Any | None:
    if ctx is None or attention_mask is None:
        return None
    mask = attention_mask.astype(mx.bool_)
    B, seq = mask.shape
    m1 = mx.reshape(mask, (B, 1, 1, seq))
    m1 = mx.broadcast_to(m1, (B, 1, seq, seq))
    m2 = mx.transpose(m1, (0, 1, 3, 2))
    return m1 & m2


def _build_joint_attn_mask(
    ctx: RuntimeContext,
    attention_mask: Any | None,
    img_len: int,
    total_len: int,
) -> Any | None:
    if attention_mask is None:
        return None
    mask = attention_mask.astype(mx.bool_)
    pad_len = total_len - int(mask.shape[1])
    if pad_len > 0:
        pad = mx.ones((int(mask.shape[0]), pad_len), dtype=mx.bool_)
        mask = ctx.concat([pad, mask], axis=1)
    B, seq = mask.shape
    m1 = mx.reshape(mask, (B, 1, 1, seq))
    m1 = mx.broadcast_to(m1, (B, 1, seq, seq))
    m2 = mx.transpose(m1, (0, 1, 3, 2))
    return m1 & m2


def _reorder_encoder_tokens(
    text: np.ndarray,
    text_mask: np.ndarray,
    text_2: np.ndarray,
    text_mask_2: np.ndarray,
    image: np.ndarray,
    image_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-sample token reorder matching diffusers forward (valid first, pad last)."""
    valid_img = image[image_mask]
    valid_t2 = text_2[text_mask_2]
    valid_t1 = text[text_mask]
    invalid_img = image[~image_mask]
    invalid_t2 = np.zeros_like(text_2[~text_mask_2])
    invalid_t1 = np.zeros_like(text[~text_mask])
    emb = np.concatenate(
        [valid_img, valid_t2, valid_t1, invalid_img, invalid_t2, invalid_t1],
        axis=0,
    )
    m = np.concatenate(
        [
            image_mask[image_mask],
            text_mask_2[text_mask_2],
            text_mask[text_mask],
            image_mask[~image_mask],
            text_mask_2[~text_mask_2],
            text_mask[~text_mask],
        ],
        axis=0,
    )
    return emb, m


def _stack_reordered_encoder(
    encoder_hidden_states: Any,
    encoder_attention_mask: Any,
    encoder_hidden_states_2: Any,
    encoder_attention_mask_2: Any,
    encoder_hidden_states_3: Any,
    encoder_attention_mask_3: Any,
) -> tuple[Any, Any]:
    """Batch-wise reorder + pad to common sequence length."""
    t_np = np.asarray(encoder_hidden_states)
    m_np = np.asarray(encoder_attention_mask).astype(bool)
    t2_np = np.asarray(encoder_hidden_states_2)
    m2_np = np.asarray(encoder_attention_mask_2).astype(bool)
    i_np = np.asarray(encoder_hidden_states_3)
    m3_np = np.asarray(encoder_attention_mask_3).astype(bool)

    B = t_np.shape[0]
    reordered_emb: list[np.ndarray] = []
    reordered_mask: list[np.ndarray] = []
    for b in range(B):
        emb, m = _reorder_encoder_tokens(
            t_np[b], m_np[b], t2_np[b], m2_np[b], i_np[b], m3_np[b],
        )
        reordered_emb.append(emb)
        reordered_mask.append(m)

    max_len = max(e.shape[0] for e in reordered_emb)
    dim = reordered_emb[0].shape[-1]
    dtype = t_np.dtype
    out_emb = np.zeros((B, max_len, dim), dtype=dtype)
    out_mask = np.zeros((B, max_len), dtype=np.bool_)
    for b, (emb, m) in enumerate(zip(reordered_emb, reordered_mask)):
        L = emb.shape[0]
        out_emb[b, :L] = emb
        out_mask[b, :L] = m
    return mx.array(out_emb), mx.array(out_mask)


class HunyuanVideoTransformer(TransformerBase):
    """HunyuanVideo-1.5 DiT — diffusers ``HunyuanVideo15Transformer3DModel`` layout."""

    def __init__(self, config: Any, ctx: RuntimeContext):
        self.config = config
        self.ctx = ctx

        heads = _cfg_int(config, "num_attention_heads", 16)
        head_dim = _cfg_int(config, "attention_head_dim", 128)
        inner_dim = _cfg_int(config, "inner_dim", heads * head_dim)
        if inner_dim != heads * head_dim:
            raise RuntimeError(
                f"HunyuanVideo inner_dim {inner_dim} != heads*head_dim {heads * head_dim}"
            )

        in_ch = _cfg_int(config, "in_channels", 65)
        out_ch = _cfg_int(config, "out_channels", 32)
        patch = _cfg_int(config, "patch_size", 1)
        patch_t = _cfg_int(config, "patch_size_t", 1)
        num_layers = _cfg_int(config, "num_layers", 54)
        num_refiner = _cfg_int(config, "num_refiner_layers", 2)
        mlp_ratio = _cfg_float(config, "mlp_ratio", 4.0)
        text_dim = _cfg_int(config, "text_embed_dim", 3584)
        text2_dim = _cfg_int(config, "text_embed_2_dim", 1472)
        image_dim = _cfg_int(config, "image_embed_dim", 1152)
        rope_theta = _cfg_float(config, "rope_theta", 256.0)
        rope_axes = getattr(config, "rope_axes_dim", (16, 56, 56))
        use_meanflow = bool(getattr(config, "use_meanflow", False))

        patch_size = (patch_t, patch, patch)

        self.x_embedder = _HunyuanVideo15PatchEmbed(ctx, patch_size, in_ch, inner_dim)
        self.image_embedder = _ImageProjection(ctx, image_dim, inner_dim)
        self.context_embedder = _TokenRefiner(
            ctx, text_dim, heads, head_dim, num_refiner, mlp_ratio,
        )
        self.context_embedder_2 = _ByT5TextProjection(ctx, text2_dim, 2048, inner_dim)
        self.time_embed = _TimeEmbedding(ctx, inner_dim, use_meanflow=use_meanflow)
        self.cond_type_embed = ctx.Embedding(3, inner_dim)
        self.rope = _RotaryPosEmbed(patch, patch_t, tuple(rope_axes), rope_theta)

        self.transformer_blocks = [
            _TransformerBlock(ctx, heads, head_dim, mlp_ratio)
            for _ in range(num_layers)
        ]
        self.norm_out = _AdaLayerNormContinuous(ctx, inner_dim)
        self.proj_out = ctx.Linear(
            inner_dim,
            patch_t * patch * patch * out_ch,
            bias=True,
        )

        self._patch_size = patch
        self._patch_size_t = patch_t
        self._out_channels = out_ch
        self._latent_channels = out_ch
        self._cond_latents: Any | None = None
        self._mask_concat: Any | None = None
        self._image_embeds: Any | None = None
        self._build_param_map()

    def prepare_conditioning(self, request: Any, bundle_root: str | None = None) -> dict[str, Any]:
        del bundle_root
        cond: dict[str, Any] = {}
        if getattr(request, "source_asset_id", None):
            cond["i2v_mode"] = True
        return cond

    def before_denoise(
        self,
        latents: Any,
        timesteps: Any,
        sigmas: Any,
        **cond: Any,
    ) -> tuple[Any, dict[str, Any]]:
        del timesteps, sigmas
        ctx = self.ctx
        B, C, T, H, W = latents.shape
        out_c = self._latent_channels
        if C != out_c:
            raise RuntimeError(
                f"HunyuanVideo noise latents expect {out_c} channels, got {C}"
            )

        cond_lat = cond.get("cond_latents")
        if cond_lat is None:
            cond_lat = ctx.zeros((B, out_c, T, H, W), dtype=latents.dtype)
        mask = cond.get("mask_concat")
        if mask is None:
            mask = ctx.zeros((B, 1, T, H, W), dtype=latents.dtype)

        self._cond_latents = cond_lat
        self._mask_concat = mask
        vision_tokens = int(getattr(self.config, "vision_num_semantic_tokens", 256))
        image_dim = int(getattr(self.config, "image_embed_dim", 1152))
        if cond.get("image_embeds") is not None:
            self._image_embeds = cond["image_embeds"]
        else:
            self._image_embeds = ctx.zeros((B, vision_tokens, image_dim), dtype=latents.dtype)
        return latents, cond

    def _build_model_input(self, latents: Any) -> Any:
        if self._cond_latents is None or self._mask_concat is None:
            raise RuntimeError("HunyuanVideo: call before_denoise before forward")
        batch_size = int(latents.shape[0])
        cond = broadcast_batch(self.ctx, self._cond_latents, batch_size)
        mask = broadcast_batch(self.ctx, self._mask_concat, batch_size)
        return self.ctx.concat([latents, cond, mask], axis=1)

    def predict_noise_cfg(
        self,
        latents_in: Any,
        t: Any,
        *,
        guidance: float,
        pos_kwargs: dict[str, Any],
        neg_kwargs: dict[str, Any],
        cfg_renorm: bool = False,
        cfg_renorm_min: float = 0.0,
    ) -> Any:
        return predict_noise_cfg_batched(
            self.forward,
            self.ctx,
            latents_in,
            t,
            guidance=guidance,
            pos_kwargs=pos_kwargs,
            neg_kwargs=neg_kwargs,
            text_keys=HUNYUAN_CFG_TEXT_KEYS,
            combine_cfg_noise=self.combine_cfg_noise,
            refine_cfg_noise=self.refine_cfg_noise,
            cfg_renorm=cfg_renorm,
            cfg_renorm_min=cfg_renorm_min,
        )

    def _build_param_map(self) -> None:
        if hasattr(self, "_param_map"):
            self._param_map.clear()
        else:
            self._param_map = {}
        _collect_params(self, "", self._param_map)

    def forward(
        self,
        latents: Any,
        timestep: Any,
        txt_embeds: Any | None = None,
        txt_attn_mask: Any | None = None,
        txt_embeds_2: Any | None = None,
        txt_attn_mask_2: Any | None = None,
        image_embeds: Any | None = None,
        *,
        timestep_r: Any | None = None,
        sigmas: Any | None = None,
        **_: Any,
    ) -> Any:
        ctx = self.ctx
        if latents.shape[1] == self._latent_channels:
            hidden_states = self._build_model_input(latents)
        else:
            hidden_states = latents
        if image_embeds is None:
            image_embeds = self._image_embeds
        if image_embeds is None:
            raise RuntimeError("HunyuanVideo requires `image_embeds` (zeros for T2V).")
        batch_size = int(hidden_states.shape[0])
        if int(image_embeds.shape[0]) == 1 and batch_size > 1:
            image_embeds = mx.repeat(image_embeds, batch_size, axis=0)
        elif int(image_embeds.shape[0]) != batch_size:
            raise RuntimeError(
                f"HunyuanVideo image_embeds batch {image_embeds.shape[0]} "
                f"!= latents batch {batch_size}"
            )
        if hidden_states.ndim != 5:
            raise RuntimeError(
                f"HunyuanVideo expects latents [B,C,T,H,W], got {hidden_states.shape}"
            )
        if txt_embeds is None:
            raise RuntimeError("HunyuanVideo requires Qwen-VL embeddings (`txt_embeds`).")
        if txt_attn_mask is None:
            raise RuntimeError("HunyuanVideo requires `txt_attn_mask`.")
        if txt_embeds_2 is None:
            raise RuntimeError("HunyuanVideo requires ByT5 embeddings (`txt_embeds_2`).")
        if txt_attn_mask_2 is None:
            raise RuntimeError("HunyuanVideo requires `txt_attn_mask_2`.")

        B, _, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self._patch_size_t, self._patch_size, self._patch_size

        freqs_cis = self.rope(hidden_states)
        temb = self.time_embed(timestep, timestep_r=timestep_r)

        hidden_states = self.x_embedder(hidden_states)
        encoder_hidden_states = self.context_embedder(txt_embeds, timestep, txt_attn_mask)

        seq1 = int(encoder_hidden_states.shape[1])
        cond0 = self.cond_type_embed(mx.zeros((B, seq1), dtype=mx.int32))
        encoder_hidden_states = encoder_hidden_states + cond0

        encoder_hidden_states_2 = self.context_embedder_2(txt_embeds_2)
        seq2 = int(encoder_hidden_states_2.shape[1])
        cond1 = self.cond_type_embed(mx.ones((B, seq2), dtype=mx.int32))
        encoder_hidden_states_2 = encoder_hidden_states_2 + cond1

        encoder_hidden_states_3 = self.image_embedder(image_embeds)
        is_t2v = bool(mx.all(image_embeds == 0))
        if is_t2v:
            encoder_hidden_states_3 = encoder_hidden_states_3 * 0.0
            encoder_attention_mask_3 = mx.zeros(
                (B, encoder_hidden_states_3.shape[1]), dtype=mx.bool_,
            )
        else:
            encoder_attention_mask_3 = mx.ones(
                (B, encoder_hidden_states_3.shape[1]), dtype=mx.bool_,
            )
        seq3 = int(encoder_hidden_states_3.shape[1])
        cond2 = self.cond_type_embed(2 * mx.ones((B, seq3), dtype=mx.int32))
        encoder_hidden_states_3 = encoder_hidden_states_3 + cond2

        encoder_hidden_states, encoder_attention_mask = _stack_reordered_encoder(
            encoder_hidden_states,
            txt_attn_mask,
            encoder_hidden_states_2,
            txt_attn_mask_2,
            encoder_hidden_states_3,
            encoder_attention_mask_3,
        )

        for block in self.transformer_blocks:
            hidden_states, encoder_hidden_states = block(
                hidden_states,
                encoder_hidden_states,
                temb,
                encoder_attention_mask,
                freqs_cis,
            )

        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)

        post_t = num_frames // p_t
        post_h = height // p_h
        post_w = width // p_w
        out_ch = self._out_channels

        out = ctx.reshape(
            hidden_states,
            (B, post_t, post_h, post_w, out_ch, p_t, p_h, p_w),
        )
        out = ctx.permute(out, (0, 4, 1, 5, 2, 6, 3, 7))
        out = ctx.reshape(out, (B, out_ch, post_t * p_t, post_h * p_h, post_w * p_w))
        return out

    def load_weights(
        self,
        weights: list[tuple[str, Any]],
        strict: bool = False,
        ctx: Any = None,
        *,
        bundle_affine_bits: int | None = None,
    ):
        """Load weights and cast floating params to bfloat16 (matches Flux2 / Qwen families)."""
        load_ctx = ctx if ctx is not None else self.ctx
        loaded, skipped = super().load_weights(
            weights,
            strict=strict,
            ctx=load_ctx,
            bundle_affine_bits=bundle_affine_bits,
        )
        for key, param in list(self._param_map.items()):
            if param.dtype != mx.bfloat16:
                new_param = param.astype(mx.bfloat16)
                self._param_map[key] = new_param
                parts = key.split(".")
                obj = self
                for part in parts[:-1]:
                    if part.isdigit():
                        obj = obj[int(part)]
                    else:
                        obj = getattr(obj, part)
                last = parts[-1]
                if hasattr(obj, last):
                    setattr(obj, last, new_param)
                elif hasattr(obj, "_parameters") and last in obj._parameters:
                    obj._parameters[last] = new_param
        mx.eval(*[p for _, p in self._param_map.items()])
        return loaded, skipped
