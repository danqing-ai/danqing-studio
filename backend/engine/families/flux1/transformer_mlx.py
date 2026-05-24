"""
Flux.1 Transformer — MM-DiT，与 diffusers ``FluxTransformer2DModel`` 对齐。

顺序：**Joint blocks（``transformer_blocks``）→ Single blocks（``single_transformer_blocks``）**，
``FluxPosEmbed`` 三轴 MRoPE（axes_dims=[16,56,56]）。
"""
from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as mx_nn
import numpy as np

from backend.engine.common._base import TransformerBase, _collect_params
from backend.engine.common.attention import scaled_dot_product_attention_bhsd_mx
from backend.engine.common.embeddings import PatchEmbed2D, sinusoidal_timestep_proj
from backend.engine.common.norm import (
    apply_ada_layer_norm_zero,
    apply_ada_layer_norm_zero_single,
    apply_rms_norm,
    apply_scale_shift,
    unpack_modulation_2way,
)
from backend.engine.config.model_configs import Flux1Config
from backend.engine.runtime._base import RuntimeContext


def _rms_norm_fp32(norm: Any, x: Any) -> Any:
    """mflux ``AttentionUtils.process_qkv`` — RMSNorm in float32, cast back."""
    eps = float(getattr(norm, "eps", 1e-6))
    return apply_rms_norm(x, norm.weight, eps)


def _scalar_to_float(x: Any) -> float:
    if isinstance(x, (float, int)):
        return float(x)
    if isinstance(x, mx.array):
        return float(np.asarray(x, dtype=np.float64).reshape(-1)[0])
    return float(np.asarray(x, dtype=np.float64).reshape(-1)[0])


class _Flux1EmbedND:
    """mflux ``EmbedND`` — 2×2 RoPE blocks for joint + single attention."""

    def __init__(self, ctx: RuntimeContext, theta: float = 10000.0):
        self.ctx = ctx
        self.theta = theta
        self.axes_dims = [16, 56, 56]

    @staticmethod
    def _rope_axis(ctx: RuntimeContext, pos: Any, dim: int, theta: float) -> Any:
        pos = pos.astype(ctx.float32())
        batch_size, seq_length = int(pos.shape[0]), int(pos.shape[1])
        scale = ctx.arange(0, dim, 2, dtype=ctx.float32()) / dim
        omega = 1.0 / (theta ** scale)
        out = pos[:, :, None] * omega[None, None, :]
        cos_out = mx.cos(out)
        sin_out = mx.sin(out)
        stacked = mx.stack([cos_out, -sin_out, sin_out, cos_out], axis=-1)
        return ctx.reshape(stacked, (batch_size, seq_length, dim // 2, 2, 2))

    def forward(self, ids: Any) -> Any:
        ctx = self.ctx
        parts = [
            self._rope_axis(ctx, ids[:, :, i], dim, self.theta)
            for i, dim in enumerate(self.axes_dims)
        ]
        emb = ctx.concat(parts, axis=2)
        return mx.expand_dims(emb, axis=1)


def _apply_flux1_rope(ctx: RuntimeContext, xq: Any, xk: Any, freqs_cis: Any) -> tuple[Any, Any]:
    """mflux ``AttentionUtils.apply_rope`` — ``xq``/``xk`` are [B, H, S, D]."""
    xq_ = mx.reshape(xq.astype(mx.float32), (*xq.shape[:-1], -1, 1, 2))
    xk_ = mx.reshape(xk.astype(mx.float32), (*xk.shape[:-1], -1, 1, 2))
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return (
        mx.reshape(xq_out, xq.shape).astype(mx.float32),
        mx.reshape(xk_out, xk.shape).astype(mx.float32),
    )


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

        # mflux ``JointAttention``: ``nn.RMSNorm(128)`` — MLX default eps=1e-5 (not ctx.RMSNorm 1e-6)
        self.norm_q = mx_nn.RMSNorm(self.dim_head, eps=1e-5)
        self.norm_k = mx_nn.RMSNorm(self.dim_head, eps=1e-5)
        self.norm_added_q = mx_nn.RMSNorm(self.dim_head, eps=1e-5)
        self.norm_added_k = mx_nn.RMSNorm(self.dim_head, eps=1e-5)

    def forward(self, hidden_states, encoder_hidden_states, rotary_emb=None):
        ctx = self.ctx
        B, S_img, _ = hidden_states.shape
        S_txt = encoder_hidden_states.shape[1]

        def _qkv(img: Any, to_q, to_k, to_v, nq, nk) -> tuple[Any, Any, Any]:
            qq = ctx.reshape(to_q(img), (B, -1, self.heads, self.dim_head))
            kk = ctx.reshape(to_k(img), (B, -1, self.heads, self.dim_head))
            vv = ctx.reshape(to_v(img), (B, -1, self.heads, self.dim_head))
            qq = ctx.permute(qq, (0, 2, 1, 3))
            kk = ctx.permute(kk, (0, 2, 1, 3))
            vv = ctx.permute(vv, (0, 2, 1, 3))
            qq = _rms_norm_fp32(nq, qq)
            kk = _rms_norm_fp32(nk, kk)
            return qq, kk, vv

        q, k, v = _qkv(hidden_states, self.to_q, self.to_k, self.to_v, self.norm_q, self.norm_k)
        q_txt, k_txt, v_txt = _qkv(
            encoder_hidden_states,
            self.add_q_proj,
            self.add_k_proj,
            self.add_v_proj,
            self.norm_added_q,
            self.norm_added_k,
        )

        q_joint = ctx.concat([q_txt, q], axis=2)
        k_joint = ctx.concat([k_txt, k], axis=2)
        v_joint = ctx.concat([v_txt, v], axis=2)

        if rotary_emb is not None:
            q_joint, k_joint = _apply_flux1_rope(ctx, q_joint, k_joint, rotary_emb)

        # mflux ``AttentionUtils.compute_attention`` — flat [B, S, H*D] before split
        scale = float(q_joint.shape[-1]) ** -0.5
        attn_out = scaled_dot_product_attention_bhsd_mx(
            mx, q_joint, k_joint, v_joint, scale=scale
        )
        attn_out = mx.reshape(
            mx.transpose(attn_out, (0, 2, 1, 3)),
            (B, -1, self.heads * self.dim_head),
        )
        txt_raw = attn_out[:, :S_txt]
        img_raw = attn_out[:, S_txt:]

        return self.to_out(img_raw), self.to_add_out(txt_raw)


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
        self.norm_q = mx_nn.RMSNorm(self.dim_head, eps=1e-5)
        self.norm_k = mx_nn.RMSNorm(self.dim_head, eps=1e-5)

    def forward(self, hidden_states, rotary_emb=None):
        ctx = self.ctx
        B, S, _ = hidden_states.shape

        q = ctx.reshape(self.to_q(hidden_states), (B, S, self.heads, self.dim_head))
        k = ctx.reshape(self.to_k(hidden_states), (B, S, self.heads, self.dim_head))
        v = ctx.reshape(self.to_v(hidden_states), (B, S, self.heads, self.dim_head))
        q = ctx.permute(q, (0, 2, 1, 3))
        k = ctx.permute(k, (0, 2, 1, 3))
        v = ctx.permute(v, (0, 2, 1, 3))
        q = _rms_norm_fp32(self.norm_q, q)
        k = _rms_norm_fp32(self.norm_k, k)

        if rotary_emb is not None:
            q, k = _apply_flux1_rope(ctx, q, k, rotary_emb)

        scale = float(q.shape[-1]) ** -0.5
        out = scaled_dot_product_attention_bhsd_mx(mx, q, k, v, scale=scale)
        return mx.reshape(mx.transpose(out, (0, 2, 1, 3)), (B, S, self.dim))


class _Flux1FeedForward:
    def __init__(self, dim: int, ctx: RuntimeContext, mult: int = 4, *, approximate: str = "none"):
        nn = ctx
        self.ctx = ctx
        self._approximate = approximate
        hidden_dim = int(dim * mult)
        self.net_0_proj = nn.Linear(dim, hidden_dim, bias=True)
        self.net_2 = nn.Linear(hidden_dim, dim, bias=True)

    def forward(self, x):
        x = self.net_0_proj(x)
        if self._approximate == "gelu_approx":
            x = mx_nn.gelu_approx(x)
        else:
            x = mx_nn.gelu(x)
        return self.net_2(x)


class _AdaLayerNormZero:
    def __init__(self, dim: int, ctx: RuntimeContext):
        self.ctx = ctx
        self.linear = ctx.Linear(dim, dim * 6, bias=True)
        self.norm = mx_nn.LayerNorm(dim, eps=1e-6, affine=False)

    def forward(self, x, emb):
        return apply_ada_layer_norm_zero(
            x,
            emb,
            linear=self.linear,
            norm=self.norm,
            silu=mx_nn.silu,
        )


class _Flux1JointBlock:
    def __init__(self, dim: int, heads: int, ctx: RuntimeContext):
        self.ctx = ctx
        self.norm1 = _AdaLayerNormZero(dim, ctx)
        self.norm1_context = _AdaLayerNormZero(dim, ctx)
        self.attn = _Flux1JointAttention(dim, heads, ctx)
        self.ff = _Flux1FeedForward(dim, ctx, mult=4)
        self.ff_context = _Flux1FeedForward(dim, ctx, mult=4, approximate="gelu_approx")
        self.norm2 = mx_nn.LayerNorm(dim, eps=1e-6, affine=False)
        # mflux ``JointTransformerBlock.norm2_context``: LayerNorm(1536), not 3072
        self.norm2_context = mx_nn.LayerNorm(1536, eps=1e-6, affine=False)

    @staticmethod
    def _apply_norm_and_feed_forward(
        hidden_states: Any,
        attn_output: Any,
        gate_mlp: Any,
        gate_msa: Any,
        scale_mlp: Any,
        shift_mlp: Any,
        norm_layer: Any,
        ff_layer: _Flux1FeedForward,
    ) -> Any:
        """mflux ``JointTransformerBlock.apply_norm_and_feed_forward``."""
        attn_output = mx.expand_dims(gate_msa, axis=1) * attn_output
        hidden_states = hidden_states + attn_output
        norm_hidden_states = norm_layer(hidden_states)
        norm_hidden_states = apply_scale_shift(
            norm_hidden_states, scale_mlp[:, None], shift_mlp[:, None], add_one=True
        )
        ff_output = ff_layer.forward(norm_hidden_states)
        ff_output = mx.expand_dims(gate_mlp, axis=1) * ff_output
        return hidden_states + ff_output

    def forward(self, hidden_states, encoder_hidden_states, temb, rotary_emb=None):
        n_img, g_msa_i, s_mlp_i, sc_mlp_i, g_mlp_i = self.norm1.forward(hidden_states, temb)
        n_txt, g_msa_t, s_mlp_t, sc_mlp_t, g_mlp_t = self.norm1_context.forward(encoder_hidden_states, temb)

        img_out, txt_out = self.attn.forward(n_img, n_txt, rotary_emb)
        hidden_states = self._apply_norm_and_feed_forward(
            hidden_states, img_out, g_mlp_i, g_msa_i, sc_mlp_i, s_mlp_i, self.norm2, self.ff,
        )
        encoder_hidden_states = self._apply_norm_and_feed_forward(
            encoder_hidden_states,
            txt_out,
            g_mlp_t,
            g_msa_t,
            sc_mlp_t,
            s_mlp_t,
            self.norm2_context,
            self.ff_context,
        )
        return encoder_hidden_states, hidden_states


class _AdaLayerNormZeroSingle:
    def __init__(self, dim: int, ctx: RuntimeContext):
        self.ctx = ctx
        self.norm = mx_nn.LayerNorm(dim, eps=1e-6, affine=False)
        self.linear = ctx.Linear(dim, dim * 3, bias=True)

    def forward(self, x, emb):
        return apply_ada_layer_norm_zero_single(
            x,
            emb,
            linear=self.linear,
            norm=self.norm,
            silu=mx_nn.silu,
        )


class _Flux1SingleBlock:
    def __init__(self, dim: int, heads: int, ctx: RuntimeContext):
        nn = ctx
        self.ctx = ctx
        self.norm = _AdaLayerNormZeroSingle(dim, ctx)
        self.attn = _Flux1SingleAttention(dim, heads, ctx)
        self.proj_mlp = nn.Linear(dim, int(dim * 4), bias=True)
        self.proj_out = nn.Linear(int(dim * 4) + dim, dim, bias=True)

    def forward(self, x, temb, rotary_emb=None):
        residual = x
        n, gate = self.norm.forward(x, temb)
        attn_out = self.attn.forward(n, rotary_emb)
        mlp_hidden = mx_nn.gelu_approx(self.proj_mlp(n))
        combined = self.ctx.concat([attn_out, mlp_hidden], axis=-1)
        out = gate[:, None, :] * self.proj_out(combined)
        return residual + out


class _AdaLayerNormContinuousOut:
    def __init__(self, dim: int, ctx: RuntimeContext):
        self.ctx = ctx
        self.norm = mx_nn.LayerNorm(dim, eps=1e-6, affine=False)
        # mflux ``AdaLayerNormContinuous``: Linear(..., bias=False); checkpoint bias is unused
        self.linear = ctx.Linear(dim, dim * 2, bias=False)

    def forward(self, x, c):
        ctx = self.ctx
        v = self.linear(ctx.silu(c).astype(mx.bfloat16))
        scale, shift = unpack_modulation_2way(v)
        x = self.norm(x)
        return apply_scale_shift(x, scale[:, None, :], shift[:, None, :], add_one=True)


def _pack_flux1_latents(ctx: RuntimeContext, latents: Any) -> Any:
    """[B, 16, H, W] → [B, (H//2)*(W//2), 64]（mflux ``FluxLatentCreator.pack_latents``）。"""
    B, c, h, w = latents.shape
    if int(c) != 16:
        raise RuntimeError(f"Flux1 pack expects 16 VAE latent channels, got {c}")
    x = ctx.reshape(latents, (B, 16, h // 2, 2, w // 2, 2))
    x = ctx.permute(x, (0, 2, 4, 1, 3, 5))
    return ctx.reshape(x, (B, (h // 2) * (w // 2), 64))


def _unpack_flux1_latents(ctx: RuntimeContext, tokens: Any, h: int, w: int) -> Any:
    """[B, (H//2)*(W//2), 64] → [B, 16, H, W]。"""
    B, seq, c = tokens.shape
    if int(c) != 64:
        raise RuntimeError(f"Flux1 unpack expects 64-dim tokens, got {c}")
    ph, pw = h // 2, w // 2
    if ph * pw != int(seq):
        raise RuntimeError(
            f"Flux1 unpack: token count {seq} != (H//2)*(W//2)={(ph * pw)} for H={h} W={w}"
        )
    x = ctx.reshape(tokens, (B, ph, pw, 16, 2, 2))
    x = ctx.permute(x, (0, 3, 1, 4, 2, 5))
    return ctx.reshape(x, (B, 16, h, w))


class _Flux1TimestepMLP:
    """diffusers ``timestep_embedder`` / mflux ``TimestepEmbedder``。"""

    def __init__(self, dim: int, ctx: RuntimeContext):
        nn = ctx
        self.ctx = ctx
        self.mlp = nn.Sequential(
            nn.Linear(256, dim, bias=True),
            nn.SiLU(),
            nn.Linear(dim, dim, bias=True),
        )

    def forward(self, sample: Any) -> Any:
        return self.mlp(sample)


def _flatten_mlx_sequential_in_param_map(param_map: dict) -> None:
    """Expand ``*.layers`` list-of-dicts (MLX Sequential) into ``*.layers.N.weight`` keys."""
    to_delete: list[str] = []
    to_add: dict[str, Any] = {}
    for key, val in param_map.items():
        if not key.endswith(".layers") or not isinstance(val, list):
            continue
        to_delete.append(key)
        for idx, layer_params in enumerate(val):
            if not isinstance(layer_params, dict):
                continue
            for pname, ptensor in layer_params.items():
                to_add[f"{key}.{idx}.{pname}"] = ptensor
    for key in to_delete:
        del param_map[key]
    param_map.update(to_add)


class Flux1Transformer(TransformerBase):
    """Flux.1 — Joint MM-DiT 后再 Single 流；与 mflux / diffusers 块序一致。"""

    def __init__(self, config: Flux1Config, ctx: RuntimeContext):
        self.config = config
        self.ctx = ctx
        nn = ctx
        dim = config.hidden_dim
        heads = config.num_heads

        # 权重来自 diffusers ``x_embedder`` Linear(64→dim)；VAE 侧为 16ch，pack 后再过此线性层
        self.patch_embed = PatchEmbed2D(64, dim, patch_size=1, ctx=ctx)
        self.txt_in = nn.Linear(config.text_dim, dim)
        # diffusers FluxTransformer2DModel：CLIP 仅 pooled 进 time_text_embed，无 clip token 流
        self.clip_in = nn.Linear(config.clip_dim, dim) if config.clip_dim else None
        self.time_in = _Flux1TimestepMLP(dim, ctx)
        self.vector_in = nn.Sequential(
            nn.Linear(config.pooled_dim, dim, bias=True),
            nn.SiLU(),
            nn.Linear(dim, dim, bias=True),
        )
        if config.supports_guidance:
            self.guidance_in = _Flux1TimestepMLP(dim, ctx)
        else:
            self.guidance_in = None

        self.pos_embed = _Flux1EmbedND(ctx)

        self.transformer_blocks = [
            _Flux1JointBlock(dim, heads, ctx) for _ in range(config.num_joint_layers)
        ]
        self.single_transformer_blocks = [
            _Flux1SingleBlock(dim, heads, ctx) for _ in range(config.num_single_layers)
        ]

        self.norm_out = _AdaLayerNormContinuousOut(dim, ctx)
        self.proj_out = nn.Linear(dim, config.out_channels)

        self._build_param_map()

    def _build_param_map(self):
        """Flatten MLX ``nn.Sequential`` ``layers`` list entries for diffusers weight keys."""
        if hasattr(self, "_param_map"):
            self._param_map.clear()
        else:
            self._param_map = {}
        _collect_params(self, "", self._param_map)
        _flatten_mlx_sequential_in_param_map(self._param_map)

    def _patch_embed_packed(self, tokens: Any) -> Any:
        """diffusers ``x_embedder`` Linear(64, dim) — 权重在 ``patch_embed.proj`` Conv1x1。"""
        ctx = self.ctx
        w = self.patch_embed.proj.weight
        w = ctx.reshape(w, (w.shape[0], -1))
        b = self.patch_embed.proj.bias
        return tokens @ w.T + b

    def forward(self, latents, timestep, txt_embeds=None, clip_embeds=None,
                pooled_embeds=None, sigmas=None, **conditioning):
        ctx = self.ctx
        cfg = self.config
        if latents.ndim != 4:
            raise RuntimeError(
                f"Flux1Transformer expects NCHW latents [B,C,H,W], got shape={tuple(latents.shape)}"
            )
        B = latents.shape[0]
        _, _, H, W = latents.shape

        timestep_embed_value = conditioning.get("timestep_embed_value")
        if timestep_embed_value is not None:
            t_val = float(timestep_embed_value)
        elif sigmas is not None:
            t_idx = int(timestep)
            n = int(sigmas.shape[0]) if hasattr(sigmas, "shape") else len(sigmas)
            sigma_t = sigmas[t_idx] if t_idx < n else sigmas[-1] if n > 0 else 1.0
            t_val = _scalar_to_float(sigma_t) * 1000.0
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
        # mflux ``compute_text_embeddings``: timestep as ModelConfig.precision (bfloat16)
        t_batch = mx.full((B,), t_val, dtype=mx.bfloat16)

        hidden_states = self._patch_embed_packed(_pack_flux1_latents(ctx, latents))
        img_seq_len = hidden_states.shape[1]

        if txt_embeds is not None:
            txt = self.txt_in(txt_embeds)
        else:
            txt = ctx.zeros((B, 0, cfg.hidden_dim))

        if clip_embeds is not None and self.clip_in is not None:
            txt = ctx.concat([txt, self.clip_in(clip_embeds)], axis=1)

        encoder_hidden_states = txt
        txt_len = encoder_hidden_states.shape[1]

        t_proj = sinusoidal_timestep_proj(ctx, t_batch, 256, flip_sin_to_cos=True)
        c = self.time_in.forward(t_proj)
        guidance_scale = conditioning.get("guidance_scale")
        if self.guidance_in is not None and guidance_scale is not None:
            g_val = float(guidance_scale) * 1000.0
            g_batch = mx.full((B,), g_val, dtype=mx.bfloat16)
            c = c + self.guidance_in.forward(
                sinusoidal_timestep_proj(ctx, g_batch, 256, flip_sin_to_cos=True)
            )
        if pooled_embeds is not None:
            c = c + self.vector_in(pooled_embeds)
        c = c.astype(mx.bfloat16)

        txt_ids, img_ids = self._prepare_pos_ids(H, W, txt_len)
        rotary_emb = self.pos_embed.forward(mx.concatenate([txt_ids, img_ids], axis=1))

        for block in self.transformer_blocks:
            encoder_hidden_states, hidden_states = block.forward(
                hidden_states, encoder_hidden_states, c, rotary_emb=rotary_emb,
            )
            if getattr(ctx, "backend", None) == "mlx":
                ctx.eval(encoder_hidden_states, hidden_states)

        x = ctx.concat([encoder_hidden_states, hidden_states], axis=1)
        for block in self.single_transformer_blocks:
            x = block.forward(x, c, rotary_emb=rotary_emb)
            if getattr(ctx, "backend", None) == "mlx":
                ctx.eval(x)

        hidden_states = x[:, txt_len:]
        hidden_states = self.norm_out.forward(hidden_states, c)
        hidden_states = self.proj_out(hidden_states)
        return _unpack_flux1_latents(ctx, hidden_states, H, W)

    def _prepare_pos_ids(self, latent_h: int, latent_w: int, txt_len: int):
        """mflux ``_prepare_text_ids`` + ``_prepare_latent_image_ids``（packed 格 H//2 × W//2）。"""
        ph, pw = latent_h // 2, latent_w // 2
        txt_ids = mx.zeros((1, txt_len, 3), dtype=mx.int32)
        img_h = mx.arange(0, ph, dtype=mx.int32)
        img_w = mx.arange(0, pw, dtype=mx.int32)
        h_grid = mx.reshape(mx.broadcast_to(img_h[:, None], (ph, pw)), (-1,))
        w_grid = mx.reshape(mx.broadcast_to(img_w[None, :], (ph, pw)), (-1,))
        zeros_img = mx.zeros(ph * pw, dtype=mx.int32)
        img_ids = mx.stack([zeros_img, h_grid, w_grid], axis=1)
        img_ids = mx.reshape(img_ids, (1, ph * pw, 3))
        return txt_ids, img_ids

    def load_weights(self, weights, strict=False, ctx=None, *, bundle_affine_bits=None):
        """Load weights and cast to bfloat16 (matches mflux / Flux2 reference precision)."""
        load_ctx = ctx if ctx is not None else self.ctx
        loaded, skipped = super().load_weights(
            weights,
            strict=strict,
            ctx=load_ctx,
            bundle_affine_bits=bundle_affine_bits,
        )
        self._cast_param_map_dtype(mx.bfloat16)
        return loaded, skipped
