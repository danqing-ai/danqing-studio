"""
Flux.1 Transformer — MM-DiT，与 diffusers ``FluxTransformer2DModel`` 对齐。

顺序：**Joint blocks（``transformer_blocks``）→ Single blocks（``single_transformer_blocks``）**，
``FluxPosEmbed`` 三轴 MRoPE（axes_dims=[16,56,56]）。
"""
from __future__ import annotations

from typing import Any

import mlx.core as mx
import numpy as np

from backend.engine.common._base import TransformerBase
from backend.engine.common.attention import apply_rope_interleaved_real as _apply_rope
from backend.engine.common.embeddings import PatchEmbed2D, TimestepEmbedding
from backend.engine.config.model_configs import Flux1Config
from backend.engine.runtime._base import RuntimeContext


def _scalar_to_float(x: Any) -> float:
    if isinstance(x, (float, int)):
        return float(x)
    if isinstance(x, mx.array):
        return float(np.asarray(x, dtype=np.float64).reshape(-1)[0])
    return float(np.asarray(x, dtype=np.float64).reshape(-1)[0])


class _Flux1PosEmbed:
    """diffusers ``FluxPosEmbed`` — axes_dims sum = head_dim."""

    def __init__(self, ctx: RuntimeContext, theta: float = 10000.0):
        self.ctx = ctx
        self.theta = theta
        self.axes_dims = [16, 56, 56]

    def forward(self, ids):
        ctx = self.ctx
        cos_list, sin_list = [], []
        for i, dim in enumerate(self.axes_dims):
            pos = ids[:, i].astype(ctx.float32())
            dtype_f32 = ctx.float32()
            freqs = 1.0 / (self.theta ** (ctx.arange(0, dim, 2, dtype=dtype_f32) / dim))
            args = pos[:, None] * freqs[None, :]
            cos_h = ctx.cos(args)
            sin_h = ctx.sin(args)
            cos_list.append(mx.repeat(cos_h, 2, axis=-1))
            sin_list.append(mx.repeat(sin_h, 2, axis=-1))
        cos = ctx.concat(cos_list, axis=-1)
        sin = ctx.concat(sin_list, axis=-1)
        cos = cos.reshape(1, 1, -1, cos.shape[-1])
        sin = sin.reshape(1, 1, -1, sin.shape[-1])
        return cos, sin


class _LayerNormNoParams:
    def __init__(self, dim: int, ctx: RuntimeContext, eps: float = 1e-6):
        self.eps = eps
        self.ctx = ctx

    def forward(self, x):
        mean = mx.mean(x, axis=-1, keepdims=True)
        var = mx.mean((x - mean) ** 2, axis=-1, keepdims=True)
        return (x - mean) / mx.sqrt(var + self.eps)


class _Flux1JointAttention:
    def __init__(self, dim: int, heads: int, ctx: RuntimeContext):
        nn = ctx
        self.ctx = ctx
        self.heads = heads
        self.dim_head = dim // heads
        self.scale = self.dim_head ** -0.5
        self.dim = dim

        self.to_q = nn.Linear(dim, dim, bias=True)
        self.to_k = nn.Linear(dim, dim, bias=True)
        self.to_v = nn.Linear(dim, dim, bias=True)
        self.to_out = nn.Linear(dim, dim, bias=True)

        self.add_q_proj = nn.Linear(dim, dim, bias=True)
        self.add_k_proj = nn.Linear(dim, dim, bias=True)
        self.add_v_proj = nn.Linear(dim, dim, bias=True)
        self.to_add_out = nn.Linear(dim, dim, bias=True)

        self.norm_q = nn.RMSNorm(self.dim_head, eps=1e-6)
        self.norm_k = nn.RMSNorm(self.dim_head, eps=1e-6)
        self.norm_added_q = nn.RMSNorm(self.dim_head, eps=1e-6)
        self.norm_added_k = nn.RMSNorm(self.dim_head, eps=1e-6)

    def forward(self, hidden_states, encoder_hidden_states,
                img_cos=None, img_sin=None, txt_cos=None, txt_sin=None):
        ctx = self.ctx
        B, S_img, _ = hidden_states.shape
        S_txt = encoder_hidden_states.shape[1]

        q = ctx.reshape(self.to_q(hidden_states), (B, S_img, self.heads, self.dim_head))
        k = ctx.reshape(self.to_k(hidden_states), (B, S_img, self.heads, self.dim_head))
        v = ctx.reshape(self.to_v(hidden_states), (B, S_img, self.heads, self.dim_head))
        q = self.norm_q(q)
        k = self.norm_k(k)

        q_txt = ctx.reshape(self.add_q_proj(encoder_hidden_states), (B, S_txt, self.heads, self.dim_head))
        k_txt = ctx.reshape(self.add_k_proj(encoder_hidden_states), (B, S_txt, self.heads, self.dim_head))
        v_txt = ctx.reshape(self.add_v_proj(encoder_hidden_states), (B, S_txt, self.heads, self.dim_head))
        q_txt = self.norm_added_q(q_txt)
        k_txt = self.norm_added_k(k_txt)

        q = ctx.permute(q, (0, 2, 1, 3))
        k = ctx.permute(k, (0, 2, 1, 3))
        v = ctx.permute(v, (0, 2, 1, 3))
        q_txt = ctx.permute(q_txt, (0, 2, 1, 3))
        k_txt = ctx.permute(k_txt, (0, 2, 1, 3))
        v_txt = ctx.permute(v_txt, (0, 2, 1, 3))

        if img_cos is not None:
            q = _apply_rope(ctx, q, img_cos, img_sin)
            k = _apply_rope(ctx, k, img_cos, img_sin)
        if txt_cos is not None:
            q_txt = _apply_rope(ctx, q_txt, txt_cos, txt_sin)
            k_txt = _apply_rope(ctx, k_txt, txt_cos, txt_sin)

        q_joint = ctx.concat([q_txt, q], axis=2)
        k_joint = ctx.concat([k_txt, k], axis=2)
        v_joint = ctx.concat([v_txt, v], axis=2)

        attn_out = ctx.attention(q_joint, k_joint, v_joint, scale=self.scale)
        attn_out = ctx.permute(attn_out, (0, 2, 1, 3))

        txt_out = attn_out[:, :S_txt, :, :].reshape(B, S_txt, self.dim)
        img_out = attn_out[:, S_txt:, :, :].reshape(B, S_img, self.dim)

        return self.to_out(img_out), self.to_add_out(txt_out)


class _Flux1SingleAttention:
    def __init__(self, dim: int, heads: int, ctx: RuntimeContext):
        nn = ctx
        self.ctx = ctx
        self.heads = heads
        self.dim_head = dim // heads
        self.scale = self.dim_head ** -0.5
        self.dim = dim

        self.to_q = nn.Linear(dim, dim, bias=True)
        self.to_k = nn.Linear(dim, dim, bias=True)
        self.to_v = nn.Linear(dim, dim, bias=True)
        self.norm_q = nn.RMSNorm(self.dim_head, eps=1e-6)
        self.norm_k = nn.RMSNorm(self.dim_head, eps=1e-6)

    def forward(self, hidden_states, cos=None, sin=None):
        ctx = self.ctx
        B, S, _ = hidden_states.shape

        q = ctx.reshape(self.to_q(hidden_states), (B, S, self.heads, self.dim_head))
        k = ctx.reshape(self.to_k(hidden_states), (B, S, self.heads, self.dim_head))
        v = ctx.reshape(self.to_v(hidden_states), (B, S, self.heads, self.dim_head))
        q = self.norm_q(q)
        k = self.norm_k(k)

        q = ctx.permute(q, (0, 2, 1, 3))
        k = ctx.permute(k, (0, 2, 1, 3))
        v = ctx.permute(v, (0, 2, 1, 3))

        if cos is not None:
            q = _apply_rope(ctx, q, cos, sin)
            k = _apply_rope(ctx, k, cos, sin)

        out = ctx.attention(q, k, v, scale=self.scale)
        out = ctx.permute(out, (0, 2, 1, 3))
        out = ctx.reshape(out, (B, S, self.dim))
        return out


class _Flux1FeedForward:
    def __init__(self, dim: int, ctx: RuntimeContext, mult: int = 4):
        nn = ctx
        self.ctx = ctx
        hidden_dim = int(dim * mult)
        self.net_0_proj = nn.Linear(dim, hidden_dim, bias=True)
        self.net_2 = nn.Linear(hidden_dim, dim, bias=True)

    def forward(self, x):
        ctx = self.ctx
        x = self.net_0_proj(x)
        x = ctx.gelu(x)
        return self.net_2(x)


class _AdaLayerNormZero:
    def __init__(self, dim: int, ctx: RuntimeContext):
        self.ctx = ctx
        self.linear = ctx.Linear(dim, dim * 6, bias=True)
        self.norm = _LayerNormNoParams(dim, ctx, eps=1e-6)

    def forward(self, x, emb):
        ctx = self.ctx
        e = self.linear(ctx.silu(emb))
        B, _ = e.shape
        dim = x.shape[-1]
        e = e.reshape(B, 6, dim)
        shift_msa = e[:, 0]
        scale_msa = e[:, 1]
        gate_msa = e[:, 2]
        shift_mlp = e[:, 3]
        scale_mlp = e[:, 4]
        gate_mlp = e[:, 5]
        n = self.norm.forward(x)
        n = n * (1 + scale_msa[:, None, :]) + shift_msa[:, None, :]
        return n, gate_msa, shift_mlp, scale_mlp, gate_mlp


class _Flux1JointBlock:
    def __init__(self, dim: int, heads: int, ctx: RuntimeContext):
        self.ctx = ctx
        self.norm1 = _AdaLayerNormZero(dim, ctx)
        self.norm1_context = _AdaLayerNormZero(dim, ctx)
        self.attn = _Flux1JointAttention(dim, heads, ctx)
        self.ff = _Flux1FeedForward(dim, ctx, mult=4)
        self.ff_context = _Flux1FeedForward(dim, ctx, mult=4)
        self.norm2 = _LayerNormNoParams(dim, ctx, eps=1e-6)
        self.norm2_context = _LayerNormNoParams(dim, ctx, eps=1e-6)

    def forward(self, hidden_states, encoder_hidden_states, temb,
                img_cos=None, img_sin=None, txt_cos=None, txt_sin=None):
        n_img, g_msa_i, s_mlp_i, sc_mlp_i, g_mlp_i = self.norm1.forward(hidden_states, temb)
        n_txt, g_msa_t, s_mlp_t, sc_mlp_t, g_mlp_t = self.norm1_context.forward(encoder_hidden_states, temb)

        img_out, txt_out = self.attn.forward(n_img, n_txt, img_cos, img_sin, txt_cos, txt_sin)
        hidden_states = hidden_states + g_msa_i[:, None, :] * img_out
        encoder_hidden_states = encoder_hidden_states + g_msa_t[:, None, :] * txt_out

        n_img = self.norm2(hidden_states)
        n_img = n_img * (1 + sc_mlp_i[:, None, :]) + s_mlp_i[:, None, :]
        n_txt = self.norm2_context(encoder_hidden_states)
        n_txt = n_txt * (1 + sc_mlp_t[:, None, :]) + s_mlp_t[:, None, :]

        hidden_states = hidden_states + g_mlp_i[:, None, :] * self.ff.forward(n_img)
        encoder_hidden_states = encoder_hidden_states + g_mlp_t[:, None, :] * self.ff_context.forward(n_txt)

        return encoder_hidden_states, hidden_states


class _AdaLayerNormZeroSingle:
    def __init__(self, dim: int, ctx: RuntimeContext):
        self.ctx = ctx
        self.norm = _LayerNormNoParams(dim, ctx, eps=1e-6)
        self.linear = ctx.Linear(dim, dim * 3, bias=True)

    def forward(self, x, emb):
        ctx = self.ctx
        e = self.linear(ctx.silu(emb))
        B, _, D = x.shape
        e = e.reshape(B, 3, D)
        shift_msa, scale_msa, gate = e[:, 0], e[:, 1], e[:, 2]
        n = self.norm.forward(x)
        n = n * (1 + scale_msa[:, None, :]) + shift_msa[:, None, :]
        return n, gate


class _Flux1SingleBlock:
    def __init__(self, dim: int, heads: int, ctx: RuntimeContext):
        nn = ctx
        self.ctx = ctx
        self.norm = _AdaLayerNormZeroSingle(dim, ctx)
        self.attn = _Flux1SingleAttention(dim, heads, ctx)
        self.proj_mlp = nn.Linear(dim, int(dim * 4), bias=True)
        self.proj_out = nn.Linear(int(dim * 4) + dim, dim, bias=True)

    def forward(self, x, temb, cos=None, sin=None):
        ctx = self.ctx
        n, gate = self.norm.forward(x, temb)
        attn_out = self.attn.forward(n, cos, sin)
        mlp_hidden = ctx.gelu(self.proj_mlp(n))
        combined = ctx.concat([attn_out, mlp_hidden], axis=-1)
        out = gate[:, None, :] * self.proj_out(combined)
        return x + out


class _AdaLayerNormContinuousOut:
    def __init__(self, dim: int, ctx: RuntimeContext):
        self.ctx = ctx
        self.norm = _LayerNormNoParams(dim, ctx, eps=1e-6)
        self.linear = ctx.Linear(dim, dim * 2, bias=True)

    def forward(self, x, c):
        ctx = self.ctx
        v = self.linear(ctx.silu(c))
        D = v.shape[-1] // 2
        scale = v[..., :D]
        shift = v[..., D:]
        x = self.norm.forward(x)
        return x * (1 + scale[:, None, :]) + shift[:, None, :]


class Flux1Transformer(TransformerBase):
    """Flux.1 — Joint MM-DiT 后再 Single 流；与 mflux / diffusers 块序一致。"""

    def __init__(self, config: Flux1Config, ctx: RuntimeContext):
        self.config = config
        self.ctx = ctx
        nn = ctx
        dim = config.hidden_dim
        heads = config.num_heads

        self.patch_embed = PatchEmbed2D(config.in_channels, dim, patch_size=1, ctx=ctx)
        self.txt_in = nn.Linear(config.text_dim, dim)
        self.clip_in = nn.Linear(config.clip_dim, dim) if config.clip_dim else None
        self.time_in = TimestepEmbedding(dim, ctx)
        self.vector_in = nn.Linear(config.pooled_dim, dim, bias=True)

        self.rope = _Flux1PosEmbed(ctx)

        self.transformer_blocks = [
            _Flux1JointBlock(dim, heads, ctx) for _ in range(config.num_joint_layers)
        ]
        self.single_transformer_blocks = [
            _Flux1SingleBlock(dim, heads, ctx) for _ in range(config.num_single_layers)
        ]

        self.norm_out = _AdaLayerNormContinuousOut(dim, ctx)
        self.proj_out = nn.Linear(dim, config.out_channels)

        self._build_param_map()

    def forward(self, latents, timestep, txt_embeds=None, clip_embeds=None,
                pooled_embeds=None, sigmas=None, **conditioning):
        ctx = self.ctx
        cfg = self.config
        B = latents.shape[0]
        _, _, H, W = latents.shape

        timestep_embed_value = conditioning.get("timestep_embed_value")
        if timestep_embed_value is not None:
            t_val = float(timestep_embed_value)
            t_batch = mx.full((B,), t_val, dtype=mx.float32)
        elif sigmas is not None:
            t_idx = int(timestep)
            n = int(sigmas.shape[0]) if hasattr(sigmas, "shape") else len(sigmas)
            sigma_t = sigmas[t_idx] if t_idx < n else sigmas[-1] if n > 0 else 1.0
            t_val = _scalar_to_float(sigma_t) * 1000.0
            t_batch = mx.full((B,), t_val, dtype=mx.float32)
        else:
            tv = timestep
            if isinstance(tv, mx.array):
                if tv.ndim == 0:
                    t_val = float(tv)
                else:
                    t_val = float(mx.reshape(tv, (-1,))[0])
            else:
                t_val = float(tv)
            if t_val <= 1.0 + 1e-5:
                t_val *= 1000.0
            t_batch = mx.full((B,), t_val, dtype=mx.float32)

        hidden_states = self.patch_embed(latents)
        img_seq_len = hidden_states.shape[1]

        if txt_embeds is not None:
            txt = self.txt_in(txt_embeds)
        else:
            txt = ctx.zeros((B, 0, cfg.hidden_dim))

        if clip_embeds is not None and self.clip_in is not None:
            txt = ctx.concat([txt, self.clip_in(clip_embeds)], axis=1)

        encoder_hidden_states = txt
        txt_len = encoder_hidden_states.shape[1]

        c = self.time_in(t_batch)
        if pooled_embeds is not None:
            c = c + self.vector_in(pooled_embeds)

        img_ids, txt_ids = self._gen_pos_ids(H, W, txt_len)
        txt_cos, txt_sin = self.rope.forward(txt_ids)
        img_cos, img_sin = self.rope.forward(img_ids)

        for block in self.transformer_blocks:
            encoder_hidden_states, hidden_states = block.forward(
                hidden_states, encoder_hidden_states, c,
                img_cos=img_cos, img_sin=img_sin,
                txt_cos=txt_cos, txt_sin=txt_sin,
            )

        x = ctx.concat([encoder_hidden_states, hidden_states], axis=1)
        all_ids = ctx.concat([txt_ids, img_ids], axis=0)
        all_cos, all_sin = self.rope.forward(all_ids)
        for block in self.single_transformer_blocks:
            x = block.forward(x, c, cos=all_cos, sin=all_sin)

        hidden_states = x[:, txt_len:]
        hidden_states = self.norm_out.forward(hidden_states, c)
        hidden_states = self.proj_out(hidden_states)

        hh = int(img_seq_len ** 0.5)
        ww = hh
        x = ctx.reshape(hidden_states, (B, hh, ww, cfg.out_channels))
        x = ctx.permute(x, (0, 3, 1, 2))
        return x

    def _gen_pos_ids(self, h: int, w: int, txt_len: int):
        ctx = self.ctx
        off = int(getattr(self.config, "max_seq_len", 512))
        img_h = mx.arange(0, h, 1, dtype=mx.int32)
        img_w = mx.arange(0, w, 1, dtype=mx.int32)
        h_grid = mx.reshape(mx.broadcast_to(img_h[:, None], (h, w)), (-1,))
        w_grid = mx.reshape(mx.broadcast_to(img_w[None, :], (h, w)), (-1,))
        ones = mx.ones(h * w, dtype=mx.int32)
        img_ids = mx.stack([ones, h_grid + off, w_grid + off], axis=1)
        txt_pos = mx.arange(0, txt_len, 1, dtype=mx.int32)
        zeros = mx.zeros(txt_len, dtype=mx.int32)
        txt_ids = mx.stack([zeros, txt_pos, txt_pos], axis=1)
        return img_ids, txt_ids
