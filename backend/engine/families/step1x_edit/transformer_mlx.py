"""Step1X-Edit DiT + Qwen2Connector (MLX) — mirrors ``modules/model_edit.py`` + ``connector_edit.py``."""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as mx_nn

from backend.engine.common.ops.attention import scaled_dot_product_attention_bhsd_mx
from backend.engine.runtime._base import RuntimeContext


def _apply_rope(ctx: RuntimeContext, xq: mx.array, xk: mx.array, freqs_cis: mx.array) -> tuple[mx.array, mx.array]:
    xq_ = ctx.reshape(xq.astype(ctx.float32()), (*xq.shape[:-1], -1, 1, 2))
    xk_ = ctx.reshape(xk.astype(ctx.float32()), (*xk.shape[:-1], -1, 1, 2))
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return (
        ctx.reshape(xq_out, xq.shape).astype(xq.dtype),
        ctx.reshape(xk_out, xk.shape).astype(xk.dtype),
    )


def _rms_norm(x: mx.array, weight: mx.array, eps: float = 1e-6) -> mx.array:
    x32 = x.astype(mx.float32)
    rrms = mx.rsqrt(mx.mean(x32 * x32, axis=-1, keepdims=True) + eps)
    return (x32 * rrms).astype(x.dtype) * weight.astype(x.dtype)


def _layer_norm_no_affine(x: mx.array) -> mx.array:
    x32 = x.astype(mx.float32)
    mean = mx.mean(x32, axis=-1, keepdims=True)
    var = mx.mean(mx.square(x32 - mean), axis=-1, keepdims=True)
    return ((x32 - mean) / mx.sqrt(var + 1e-6)).astype(x.dtype)


def _attention_blhd(
    ctx: RuntimeContext,
    q: mx.array,
    k: mx.array,
    v: mx.array,
    *,
    attn_mask: mx.array | None = None,
) -> mx.array:
    q = ctx.permute(q, (0, 2, 1, 3))
    k = ctx.permute(k, (0, 2, 1, 3))
    v = ctx.permute(v, (0, 2, 1, 3))
    scale = float(q.shape[-1]) ** -0.5
    mask = None
    if attn_mask is not None:
        mask = mx.where(attn_mask > 0, mx.zeros_like(attn_mask), mx.full(attn_mask.shape, -1e9))
    out = scaled_dot_product_attention_bhsd_mx(mx, q, k, v, scale=scale, mask=mask)
    out = ctx.permute(out, (0, 2, 1, 3))
    b, s, h, d = out.shape
    return ctx.reshape(out, (b, s, h * d))


def _timestep_embedding(t: mx.array, dim: int, max_period: int = 10000) -> mx.array:
    t = t.astype(mx.float32) * 1000.0
    half = dim // 2
    freqs = mx.exp(-math.log(max_period) * mx.arange(0, half, dtype=mx.float32) / half)
    args = t[:, None] * freqs[None]
    emb = mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)
    if dim % 2:
        emb = mx.concatenate([emb, mx.zeros((emb.shape[0], 1), dtype=emb.dtype)], axis=-1)
    return emb


class _EmbedND:
    def __init__(self, ctx: RuntimeContext, dim: int, theta: int, axes_dim: list[int]):
        self.ctx = ctx
        self.theta = theta
        self.axes_dim = axes_dim

    @staticmethod
    def _rope_axis(ctx: RuntimeContext, pos: mx.array, dim: int, theta: int) -> mx.array:
        pos = pos.astype(mx.float32)
        scale = mx.arange(0, dim, 2, dtype=mx.float32) / dim
        omega = 1.0 / (theta**scale)
        out = pos[..., None].astype(mx.float32) * omega[None, None, :]
        cos_out = mx.cos(out)
        sin_out = mx.sin(out)
        stacked = mx.stack([cos_out, -sin_out, sin_out, cos_out], axis=-1)
        b, s, half, _, _ = stacked.shape
        return stacked.reshape(b, s, half, 2, 2)

    def __call__(self, ids: mx.array) -> mx.array:
        ctx = self.ctx
        parts = [self._rope_axis(ctx, ids[:, :, i], d, self.theta) for i, d in enumerate(self.axes_dim)]
        emb = ctx.concat(parts, axis=2)
        return ctx.expand_dims(emb, axis=1)


class _MLPEmbedder(mx_nn.Module):
    def __init__(self, ctx: RuntimeContext, in_dim: int, hidden_dim: int):
        super().__init__()
        self.in_layer = ctx.Linear(in_dim, hidden_dim, bias=True)
        self.out_layer = ctx.Linear(hidden_dim, hidden_dim, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        return self.out_layer(mx_nn.silu(self.in_layer(x)))


class _QKNorm(mx_nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query_norm = mx_nn.RMSNorm(dim, eps=1e-6)
        self.key_norm = mx_nn.RMSNorm(dim, eps=1e-6)

    def __call__(self, q: mx.array, k: mx.array, v: mx.array) -> tuple[mx.array, mx.array]:
        q = _rms_norm(q, self.query_norm.weight, eps=1e-6)
        k = _rms_norm(k, self.key_norm.weight, eps=1e-6)
        return q.astype(v.dtype), k.astype(v.dtype)


class _SelfAttention(mx_nn.Module):
    def __init__(self, ctx: RuntimeContext, dim: int, num_heads: int, qkv_bias: bool = True):
        super().__init__()
        self.ctx = ctx
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = ctx.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm = _QKNorm(self.head_dim)
        self.proj = ctx.Linear(dim, dim, bias=True)

    def __call__(self, x: mx.array, pe: mx.array) -> mx.array:
        ctx = self.ctx
        b, s, _ = x.shape
        qkv = self.qkv(x).reshape(b, s, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        q, k = self.norm(q, k, v)
        q, k = _apply_rope(ctx, q, k, pe)
        attn = _attention_blhd(ctx, q, k, v)
        return self.proj(attn)


class _Modulation(mx_nn.Module):
    def __init__(self, ctx: RuntimeContext, dim: int, *, double: bool):
        super().__init__()
        self.multiplier = 6 if double else 3
        self.lin = ctx.Linear(dim, self.multiplier * dim, bias=True)
        self.double = double

    def __call__(self, vec: mx.array):
        out = mx.split(mx_nn.silu(self.lin(vec))[:, None, :], self.multiplier, axis=-1)
        first = (out[0], out[1], out[2])
        second = (out[3], out[4], out[5]) if self.double else None
        return first, second


class _DoubleStreamBlock(mx_nn.Module):
    def __init__(self, ctx: RuntimeContext, hidden_size: int, num_heads: int, mlp_ratio: float, qkv_bias: bool):
        super().__init__()
        self.ctx = ctx
        self.num_heads = num_heads
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.img_mod = _Modulation(ctx, hidden_size, double=True)
        self.img_attn = _SelfAttention(ctx, hidden_size, num_heads, qkv_bias)
        self.img_mlp = [ctx.Linear(hidden_size, mlp_hidden, bias=True), ctx.Linear(mlp_hidden, hidden_size, bias=True)]
        self.txt_mod = _Modulation(ctx, hidden_size, double=True)
        self.txt_attn = _SelfAttention(ctx, hidden_size, num_heads, qkv_bias)
        self.txt_mlp = [ctx.Linear(hidden_size, mlp_hidden, bias=True), ctx.Linear(mlp_hidden, hidden_size, bias=True)]

    def __call__(self, img: mx.array, txt: mx.array, vec: mx.array, pe: mx.array) -> tuple[mx.array, mx.array]:
        ctx = self.ctx
        (img_s1, img_sc1, img_g1), (img_s2, img_sc2, img_g2) = self.img_mod(vec)
        (txt_s1, txt_sc1, txt_g1), (txt_s2, txt_sc2, txt_g2) = self.txt_mod(vec)

        img_mod = (1 + img_sc1) * _layer_norm_no_affine(img) + img_s1
        txt_mod = (1 + txt_sc1) * _layer_norm_no_affine(txt) + txt_s1

        b = img.shape[0]
        st = txt.shape[1]

        def _qkv(attn: _SelfAttention, x: mx.array):
            qkv = attn.qkv(x).reshape(b, -1, 3, self.num_heads, attn.head_dim)
            return qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

        iq, ik, iv = _qkv(self.img_attn, img_mod)
        tq, tk, tv = _qkv(self.txt_attn, txt_mod)
        iq, ik = self.img_attn.norm(iq, ik, iv)
        tq, tk = self.txt_attn.norm(tq, tk, tv)
        q = mx.concatenate([tq, iq], axis=1)
        k = mx.concatenate([tk, ik], axis=1)
        v = mx.concatenate([tv, iv], axis=1)
        q, k = _apply_rope(ctx, q, k, pe)
        attn = _attention_blhd(ctx, q, k, v)
        txt_attn = attn[:, :st]
        img_attn = attn[:, st:]
        img = img + img_g1 * self.img_attn.proj(img_attn)
        img_ff = (1 + img_sc2) * _layer_norm_no_affine(img) + img_s2
        img = img + img_g2 * self.img_mlp[1](mx_nn.gelu_approx(self.img_mlp[0](img_ff)))
        txt = txt + txt_g1 * self.txt_attn.proj(txt_attn)
        txt_ff = (1 + txt_sc2) * _layer_norm_no_affine(txt) + txt_s2
        txt = txt + txt_g2 * self.txt_mlp[1](mx_nn.gelu_approx(self.txt_mlp[0](txt_ff)))
        return img, txt


class _SingleStreamBlock(mx_nn.Module):
    def __init__(self, ctx: RuntimeContext, hidden_size: int, num_heads: int, mlp_ratio: float):
        super().__init__()
        self.ctx = ctx
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.hidden_size = hidden_size
        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.linear1 = ctx.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim, bias=True)
        self.linear2 = ctx.Linear(hidden_size + self.mlp_hidden_dim, hidden_size, bias=True)
        self.norm = _QKNorm(self.head_dim)
        self.modulation = _Modulation(ctx, hidden_size, double=False)

    def __call__(self, x: mx.array, vec: mx.array, pe: mx.array) -> mx.array:
        ctx = self.ctx
        (shift, scale, gate), _ = self.modulation(vec)
        x_mod = (1 + scale) * _layer_norm_no_affine(x) + shift
        qkv_mlp = self.linear1(x_mod)
        qkv = qkv_mlp[:, :, : 3 * self.hidden_size]
        mlp = qkv_mlp[:, :, 3 * self.hidden_size :]
        b, s, _ = x.shape
        qkv = qkv.reshape(b, s, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        q, k = self.norm(q, k, v)
        q, k = _apply_rope(ctx, q, k, pe)
        attn = _attention_blhd(ctx, q, k, v)
        mlp = mx_nn.gelu_approx(mlp)
        out = self.linear2(mx.concatenate([attn, mlp], axis=-1))
        return x + gate * out


class _LastLayer(mx_nn.Module):
    def __init__(self, ctx: RuntimeContext, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.linear = ctx.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = mx_nn.Module()
        self.adaLN_modulation.layers = [mx_nn.Identity(), ctx.Linear(hidden_size, 2 * hidden_size, bias=True)]

    def __call__(self, x: mx.array, vec: mx.array) -> mx.array:
        shift, scale = mx.split(self.adaLN_modulation.layers[1](mx_nn.silu(vec)), 2, axis=-1)
        x = (1 + scale[:, None, :]) * _layer_norm_no_affine(x) + shift[:, None, :]
        return self.linear(x)


class _MLP(mx_nn.Module):
    def __init__(self, ctx: RuntimeContext, in_channels: int, hidden_channels: int):
        super().__init__()
        self.fc1 = ctx.Linear(in_channels, hidden_channels, bias=True)
        self.fc2 = ctx.Linear(hidden_channels, in_channels, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(mx_nn.silu(self.fc1(x)))


def _connector_timestep_embedding(t: mx.array, dim: int, max_period: int = 10000) -> mx.array:
    half = dim // 2
    freqs = mx.exp(-math.log(max_period) * mx.arange(0, half, dtype=mx.float32) / half)
    args = t.astype(mx.float32)[:, None] * freqs[None]
    emb = mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)
    if dim % 2:
        emb = mx.concatenate([emb, mx.zeros((emb.shape[0], 1), dtype=emb.dtype)], axis=-1)
    return emb


class _TimestepEmbedderConnector(mx_nn.Module):
    def __init__(self, ctx: RuntimeContext, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = mx_nn.Module()
        self.mlp.layers = [
            ctx.Linear(frequency_embedding_size, hidden_size, bias=True),
            mx_nn.Identity(),
            ctx.Linear(hidden_size, hidden_size, bias=True),
        ]

    def __call__(self, t: mx.array) -> mx.array:
        emb = _connector_timestep_embedding(t, int(self.mlp.layers[0].weight.shape[-1]))
        h = self.mlp.layers[0](emb)
        h = mx_nn.silu(h)
        return self.mlp.layers[2](h)


class _TextProjection(mx_nn.Module):
    def __init__(self, ctx: RuntimeContext, in_channels: int, hidden_size: int):
        super().__init__()
        self.linear_1 = ctx.Linear(in_channels, hidden_size, bias=True)
        self.linear_2 = ctx.Linear(hidden_size, hidden_size, bias=True)

    def __call__(self, caption: mx.array) -> mx.array:
        return self.linear_2(mx_nn.silu(self.linear_1(caption)))


class _IndividualTokenRefinerBlock(mx_nn.Module):
    def __init__(self, ctx: RuntimeContext, hidden_size: int, heads_num: int):
        super().__init__()
        self.ctx = ctx
        self.heads_num = heads_num
        self.head_dim = hidden_size // heads_num
        self.norm1 = mx_nn.LayerNorm(hidden_size, eps=1e-6)
        self.self_attn_qkv = ctx.Linear(hidden_size, hidden_size * 3, bias=True)
        self.self_attn_q_norm = mx_nn.LayerNorm(self.head_dim, eps=1e-6)
        self.self_attn_k_norm = mx_nn.LayerNorm(self.head_dim, eps=1e-6)
        self.self_attn_proj = ctx.Linear(hidden_size, hidden_size, bias=True)
        self.norm2 = mx_nn.LayerNorm(hidden_size, eps=1e-6)
        self.mlp = _MLP(ctx, hidden_size, int(hidden_size * 4))
        self.adaLN_modulation = mx_nn.Module()
        self.adaLN_modulation.layers = [mx_nn.Identity(), ctx.Linear(hidden_size, 2 * hidden_size, bias=True)]

    def __call__(self, x: mx.array, c: mx.array, attn_mask: mx.array | None = None) -> mx.array:
        ctx = self.ctx
        gate_msa, gate_mlp = mx.split(self.adaLN_modulation.layers[1](mx_nn.silu(c)), 2, axis=-1)
        norm_x = self.norm1(x.astype(mx.float32)).astype(x.dtype)
        b, l, _ = norm_x.shape
        qkv = self.self_attn_qkv(norm_x).reshape(b, l, 3, self.heads_num, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        q = self.self_attn_q_norm(q.astype(mx.float32)).astype(v.dtype)
        k = self.self_attn_k_norm(k.astype(mx.float32)).astype(v.dtype)
        attn = _attention_blhd(ctx, q, k, v, attn_mask=attn_mask)
        x = x + gate_msa[:, None, :] * self.self_attn_proj(attn)
        x = x + gate_mlp[:, None, :] * self.mlp(self.norm2(x.astype(mx.float32)).astype(x.dtype))
        return x


def _build_self_attn_mask(mask: mx.array) -> mx.array:
    batch_size, seq_len = mask.shape
    m = mask.astype(mx.float32)
    m1 = mx.broadcast_to(mx.reshape(m, (batch_size, 1, 1, seq_len)), (batch_size, 1, seq_len, seq_len))
    m2 = mx.transpose(m1, (0, 1, 3, 2))
    return m1 * m2


class _SingleTokenRefiner(mx_nn.Module):
    def __init__(self, ctx: RuntimeContext, in_channels: int, hidden_size: int, heads_num: int, depth: int):
        super().__init__()
        self.input_embedder = ctx.Linear(in_channels, hidden_size, bias=True)
        self.t_embedder = _TimestepEmbedderConnector(ctx, hidden_size)
        self.c_embedder = _TextProjection(ctx, in_channels, hidden_size)
        self.individual_token_refiner = mx_nn.Module()
        self.individual_token_refiner.blocks = [
            _IndividualTokenRefinerBlock(ctx, hidden_size, heads_num) for _ in range(depth)
        ]

    def __call__(self, x: mx.array, t: mx.array, mask: mx.array | None = None) -> mx.array:
        timestep_aware = self.t_embedder(t)
        if mask is None:
            context_aware = mx.mean(x, axis=1)
        else:
            mask_f = mask.astype(mx.float32)[..., None]
            context_aware = mx.sum(x * mask_f, axis=1) / mx.maximum(mx.sum(mask_f, axis=1), 1e-6)
        c = timestep_aware + self.c_embedder(context_aware)
        x = self.input_embedder(x)
        attn_mask = _build_self_attn_mask(mask) if mask is not None else None
        for block in self.individual_token_refiner.blocks:
            x = block(x, c, attn_mask)
        return x


class _Qwen2Connector(mx_nn.Module):
    def __init__(self, ctx: RuntimeContext, version: str = "v1.1"):
        super().__init__()
        self.S = _SingleTokenRefiner(ctx, in_channels=3584, hidden_size=4096, heads_num=32, depth=2)
        self.global_proj_out = ctx.Linear(3584, 768, bias=True)
        self.version = version

    def __call__(self, x: mx.array, t: mx.array, mask: mx.array) -> tuple[mx.array, mx.array]:
        t = t * 1000.0
        mask_f = mask.astype(mx.float32)[..., None]
        x_mean = mx.sum(x * mask_f, axis=1) / mx.maximum(mx.sum(mask_f, axis=1), 1e-6)
        global_out = self.global_proj_out(x_mean)
        return self.S(x, t, mask), global_out


class Step1XEditMLX(mx_nn.Module):
    """Step1X-Edit DiT v1.1 — flat safetensor keys (``img_in.*``, ``double_blocks.*``, …)."""

    def __init__(
        self,
        ctx: RuntimeContext,
        *,
        in_channels: int = 64,
        out_channels: int = 64,
        vec_in_dim: int = 768,
        context_in_dim: int = 4096,
        hidden_size: int = 3072,
        mlp_ratio: float = 4.0,
        num_heads: int = 24,
        depth: int = 19,
        depth_single_blocks: int = 38,
        axes_dim: list[int] | None = None,
        theta: int = 10_000,
        qkv_bias: bool = True,
        version: str = "v1.1",
    ):
        super().__init__()
        axes_dim = axes_dim or [16, 56, 56]
        pe_dim = hidden_size // num_heads
        self.ctx = ctx
        self.pe_embedder = _EmbedND(ctx, pe_dim, theta, axes_dim)
        self.img_in = ctx.Linear(in_channels, hidden_size, bias=True)
        self.time_in = _MLPEmbedder(ctx, 256, hidden_size)
        self.vector_in = _MLPEmbedder(ctx, vec_in_dim, hidden_size)
        self.txt_in = ctx.Linear(context_in_dim, hidden_size, bias=True)
        self.double_blocks = [
            _DoubleStreamBlock(ctx, hidden_size, num_heads, mlp_ratio, qkv_bias) for _ in range(depth)
        ]
        self.single_blocks = [
            _SingleStreamBlock(ctx, hidden_size, num_heads, mlp_ratio) for _ in range(depth_single_blocks)
        ]
        self.final_layer = _LastLayer(ctx, hidden_size, 1, out_channels)
        self.connector = _Qwen2Connector(ctx, version=version)

    def __call__(
        self,
        img: mx.array,
        img_ids: mx.array,
        txt_ids: mx.array,
        timesteps: mx.array,
        llm_embedding: mx.array,
        t_vec: mx.array,
        mask: mx.array,
    ) -> mx.array:
        txt, y = self.connector(llm_embedding, t_vec, mask)
        img = self.img_in(img)
        vec = self.time_in(_timestep_embedding(timesteps, 256))
        vec = vec + self.vector_in(y)
        txt = self.txt_in(txt)
        ids = mx.concatenate([txt_ids, img_ids], axis=1)
        pe = self.pe_embedder(ids)
        for block in self.double_blocks:
            img, txt = block(img, txt, vec, pe)
        merged = mx.concatenate([txt, img], axis=1)
        for block in self.single_blocks:
            merged = block(merged, vec, pe)
        txt_len = txt.shape[1]
        return self.final_layer(merged[:, txt_len:, :], vec)
