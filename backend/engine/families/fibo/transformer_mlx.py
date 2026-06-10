"""
FIBO Transformer — Bria4Transformer2DModel compatible.

Joint MM-DiT (8 blocks) → Single DiT (38 blocks).
Diffusers weight keys match directly (via _collect_params list traversal).
"""
from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as mx_nn
import numpy as np

from backend.engine.common.model.base import TransformerBase, _collect_params
from backend.engine.common.ops.attention import scaled_dot_product_attention_bhsd_mx
from backend.engine.common.ops.cfg_batch import FIBO_CFG_TEXT_KEYS, predict_noise_cfg_batched
from backend.engine.common.ops.embeddings import sinusoidal_timestep_proj
from backend.engine.common.ops.norm import (
    apply_ada_layer_norm_continuous,
    apply_ada_layer_norm_zero,
    apply_ada_layer_norm_zero_single,
)
from backend.engine.config.model_configs import FIBOConfig
from backend.engine.runtime._base import RuntimeContext


def _fibo_apply_rotary_emb(ctx: RuntimeContext, x: Any, cos: Any, sin: Any) -> Any:
    bsz, seq_len, num_heads, head_dim = x.shape
    cos = ctx.expand_dims(ctx.expand_dims(cos, axis=0), axis=2)
    sin = ctx.expand_dims(ctx.expand_dims(sin, axis=0), axis=2)
    x2 = x.reshape(bsz, seq_len, num_heads, -1, 2)
    x_real = x2[..., 0]
    x_imag = x2[..., 1]
    x_rotated = ctx.stack([-x_imag, x_real], axis=-1).reshape(bsz, seq_len, num_heads, head_dim)
    return (x.astype(ctx.float32()) * cos + x_rotated.astype(ctx.float32()) * sin).astype(x.dtype)


class _FiboEmbedND:
    """FiboEmbedND — 3-axis RoPE (axes_dim=[16,56,56])."""

    def __init__(self, ctx: RuntimeContext, theta: float = 10000.0):
        self.ctx = ctx
        self.theta = theta
        self.axes_dim = [16, 56, 56]

    def forward(self, ids: Any) -> tuple[Any, Any]:
        ctx = self.ctx
        if ids.ndim == 3 and int(ids.shape[0]) == 1:
            ids = ids[0]
        pos = ids.astype(ctx.float32())
        cos_out = []
        sin_out = []
        for i, dim in enumerate(self.axes_dim):
            cos_axis, sin_axis = self._rope_1d(pos[:, i], dim)
            cos_out.append(cos_axis)
            sin_out.append(sin_axis)
        return ctx.concat(cos_out, axis=-1), ctx.concat(sin_out, axis=-1)

    def _rope_1d(self, pos: Any, dim: int) -> tuple[Any, Any]:
        ctx = self.ctx
        pos = pos.astype(ctx.float32())
        if pos.ndim != 1:
            pos = ctx.reshape(pos, (-1,))
        freqs = 1.0 / (self.theta ** (ctx.arange(0, dim, 2, dtype=ctx.float32()) / dim))
        angles = pos[:, None] * freqs[None, :]
        cos_base = ctx.cos(angles)
        sin_base = ctx.sin(angles)
        cos = ctx.reshape(ctx.stack([cos_base, cos_base], axis=-1), (pos.shape[0], -1))
        sin = ctx.reshape(ctx.stack([sin_base, sin_base], axis=-1), (pos.shape[0], -1))
        return cos, sin


class _TimestepEmbedder:
    """Wrapper to match diffusers weight key path."""

    def __init__(self, dim: int):
        self.linear_1 = mx_nn.Linear(256, dim, bias=True)
        self.linear_2 = mx_nn.Linear(dim, dim, bias=True)


class _BriaFiboTimestepProjEmbeddings:
    """BriaFiboTimestepProjEmbeddings — sinusoidal + MLP."""

    def __init__(self, dim: int, ctx: RuntimeContext):
        self.ctx = ctx
        self.timestep_embedder = _TimestepEmbedder(dim)

    def forward(self, sample: Any) -> Any:
        x = mx_nn.silu(self.timestep_embedder.linear_1(sample))
        return self.timestep_embedder.linear_2(x)


class _FiboAdaLayerNormZero:
    """FiboAdaLayerNormZero — delegates to ``common/norm.apply_ada_layer_norm_zero``."""

    def __init__(self, dim: int, ctx: RuntimeContext):
        self.ctx = ctx
        self.linear = mx_nn.Linear(dim, dim * 6, bias=True)
        self.norm = mx_nn.LayerNorm(dim, eps=1e-6, affine=False)

    def forward(self, hidden_states: Any, text_embeddings: Any) -> tuple[Any, ...]:
        return apply_ada_layer_norm_zero(
            hidden_states, text_embeddings,
            linear=self.linear, norm=self.norm, silu=mx_nn.silu,
        )


class _FiboAdaLayerNormZeroSingle:
    """AdaLayerNormZeroSingle — delegates to ``common/norm.apply_ada_layer_norm_zero_single``."""

    def __init__(self, dim: int, ctx: RuntimeContext):
        self.ctx = ctx
        self.linear = mx_nn.Linear(dim, dim * 3, bias=True)
        self.norm = mx_nn.LayerNorm(dim, eps=1e-6, affine=False)

    def forward(self, hidden_states: Any, text_embeddings: Any) -> tuple[Any, Any]:
        return apply_ada_layer_norm_zero_single(
            hidden_states, text_embeddings,
            linear=self.linear, norm=self.norm, silu=mx_nn.silu,
        )


class _FiboJointAttention:
    """FiboJointAttention — joint QKV for image + context streams."""

    def __init__(self, dim: int, num_heads: int, head_dim: int, ctx: RuntimeContext):
        self.ctx = ctx
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = dim

        self.to_q = mx_nn.Linear(dim, dim, bias=True)
        self.to_k = mx_nn.Linear(dim, dim, bias=True)
        self.to_v = mx_nn.Linear(dim, dim, bias=True)
        self.norm_q = mx_nn.RMSNorm(head_dim, eps=1e-6)
        self.norm_k = mx_nn.RMSNorm(head_dim, eps=1e-6)

        self.add_q_proj = mx_nn.Linear(dim, dim, bias=True)
        self.add_k_proj = mx_nn.Linear(dim, dim, bias=True)
        self.add_v_proj = mx_nn.Linear(dim, dim, bias=True)
        self.norm_added_q = mx_nn.RMSNorm(head_dim, eps=1e-6)
        self.norm_added_k = mx_nn.RMSNorm(head_dim, eps=1e-6)

        self.to_out = [mx_nn.Linear(dim, dim, bias=True)]
        self.to_add_out = mx_nn.Linear(dim, dim, bias=True)

    def forward(
        self,
        hidden_states: Any,
        encoder_hidden_states: Any,
        image_rotary_emb: tuple[Any, Any],
        attention_mask: Any | None = None,
    ) -> tuple[Any, Any]:
        ctx = self.ctx
        batch_size, seq_img, _ = hidden_states.shape
        _, seq_ctx, _ = encoder_hidden_states.shape
        cos, sin = image_rotary_emb

        query = self.to_q(hidden_states)
        key = self.to_k(hidden_states)
        value = self.to_v(hidden_states)

        enc_query = self.add_q_proj(encoder_hidden_states)
        enc_key = self.add_k_proj(encoder_hidden_states)
        enc_value = self.add_v_proj(encoder_hidden_states)

        query = ctx.reshape(query, (batch_size, seq_img, self.num_heads, self.head_dim))
        key = ctx.reshape(key, (batch_size, seq_img, self.num_heads, self.head_dim))
        value = ctx.reshape(value, (batch_size, seq_img, self.num_heads, self.head_dim))

        enc_query = ctx.reshape(enc_query, (batch_size, seq_ctx, self.num_heads, self.head_dim))
        enc_key = ctx.reshape(enc_key, (batch_size, seq_ctx, self.num_heads, self.head_dim))
        enc_value = ctx.reshape(enc_value, (batch_size, seq_ctx, self.num_heads, self.head_dim))

        query = self.norm_q(query.astype(ctx.float32())).astype(query.dtype)
        key = self.norm_k(key.astype(ctx.float32())).astype(key.dtype)
        enc_query = self.norm_added_q(enc_query.astype(ctx.float32())).astype(enc_query.dtype)
        enc_key = self.norm_added_k(enc_key.astype(ctx.float32())).astype(enc_key.dtype)

        query = ctx.concat([enc_query, query], axis=1)
        key = ctx.concat([enc_key, key], axis=1)
        value = ctx.concat([enc_value, value], axis=1)

        query = _fibo_apply_rotary_emb(ctx, query, cos, sin)
        key = _fibo_apply_rotary_emb(ctx, key, cos, sin)

        query_bhsd = ctx.permute(query, (0, 2, 1, 3))
        key_bhsd = ctx.permute(key, (0, 2, 1, 3))
        value_bhsd = ctx.permute(value, (0, 2, 1, 3))

        scale = 1.0 / ctx.sqrt(ctx.array(self.head_dim, dtype=query_bhsd.dtype))
        attn_output = scaled_dot_product_attention_bhsd_mx(
            mx, query_bhsd, key_bhsd, value_bhsd, scale=scale, mask=attention_mask
        )

        attn_output = ctx.permute(attn_output, (0, 2, 1, 3))
        attn_output = ctx.reshape(attn_output, (batch_size, seq_img + seq_ctx, self.inner_dim))

        context_attn_output = attn_output[:, :seq_ctx, :]
        hidden_attn_output = attn_output[:, seq_ctx:, :]

        hidden_attn_output = self.to_out[0](hidden_attn_output)
        context_attn_output = self.to_add_out(context_attn_output)
        return hidden_attn_output, context_attn_output


class _FiboSingleAttention:
    """FiboSingleAttention — single-stream attention."""

    def __init__(self, dim: int, num_heads: int, head_dim: int, ctx: RuntimeContext):
        self.ctx = ctx
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = dim

        self.to_q = mx_nn.Linear(dim, dim, bias=True)
        self.to_k = mx_nn.Linear(dim, dim, bias=True)
        self.to_v = mx_nn.Linear(dim, dim, bias=True)
        self.norm_q = mx_nn.RMSNorm(head_dim, eps=1e-6)
        self.norm_k = mx_nn.RMSNorm(head_dim, eps=1e-6)

    def forward(
        self,
        hidden_states: Any,
        image_rotary_emb: tuple[Any, Any],
        attention_mask: Any | None = None,
    ) -> Any:
        ctx = self.ctx
        batch_size, seq_len, _ = hidden_states.shape
        cos, sin = image_rotary_emb

        query = self.to_q(hidden_states)
        key = self.to_k(hidden_states)
        value = self.to_v(hidden_states)

        query = ctx.reshape(query, (batch_size, seq_len, self.num_heads, self.head_dim))
        key = ctx.reshape(key, (batch_size, seq_len, self.num_heads, self.head_dim))
        value = ctx.reshape(value, (batch_size, seq_len, self.num_heads, self.head_dim))

        query = self.norm_q(query.astype(ctx.float32())).astype(query.dtype)
        key = self.norm_k(key.astype(ctx.float32())).astype(key.dtype)

        query = _fibo_apply_rotary_emb(ctx, query, cos, sin)
        key = _fibo_apply_rotary_emb(ctx, key, cos, sin)

        query_bhsd = ctx.permute(query, (0, 2, 1, 3))
        key_bhsd = ctx.permute(key, (0, 2, 1, 3))
        value_bhsd = ctx.permute(value, (0, 2, 1, 3))

        scale = 1.0 / ctx.sqrt(ctx.array(self.head_dim, dtype=query_bhsd.dtype))
        attn_output = scaled_dot_product_attention_bhsd_mx(
            mx, query_bhsd, key_bhsd, value_bhsd, scale=scale, mask=attention_mask
        )

        attn_output = ctx.permute(attn_output, (0, 2, 1, 3))
        return ctx.reshape(attn_output, (batch_size, seq_len, self.inner_dim))


class _FiboGELU:
    """FiboGELU — Linear + gelu_approx."""

    def __init__(self, dim_in: int, dim_out: int):
        self.proj = mx_nn.Linear(dim_in, dim_out, bias=True)

    def __call__(self, x: Any) -> Any:
        return mx_nn.gelu_approx(self.proj(x))

    def forward(self, x: Any) -> Any:
        return mx_nn.gelu_approx(self.proj(x))


class _FiboFeedForward:
    """FiboFeedForward — GELU-approx + Linear."""

    def __init__(self, dim: int, ctx: RuntimeContext, mult: float = 4.0):
        inner_dim = int(dim * mult)
        self.net = [
            _FiboGELU(dim, inner_dim),
            mx_nn.Dropout(0.0),
            mx_nn.Linear(inner_dim, dim, bias=True),
        ]

    def forward(self, x: Any) -> Any:
        for layer in self.net:
            x = layer(x)
        return x


class _FiboJointTransformerBlock:
    """FiboJointTransformerBlock."""

    def __init__(self, dim: int, num_heads: int, head_dim: int, ctx: RuntimeContext):
        self.ctx = ctx
        self.norm1 = _FiboAdaLayerNormZero(dim, ctx)
        self.norm1_context = _FiboAdaLayerNormZero(dim, ctx)
        self.attn = _FiboJointAttention(dim, num_heads, head_dim, ctx)
        self.norm2 = mx_nn.LayerNorm(dim, eps=1e-6, affine=False)
        self.ff = _FiboFeedForward(dim, ctx)
        self.norm2_context = mx_nn.LayerNorm(dim, eps=1e-6, affine=False)
        self.ff_context = _FiboFeedForward(dim, ctx)

    def forward(
        self,
        hidden_states: Any,
        encoder_hidden_states: Any,
        temb: Any,
        image_rotary_emb: tuple[Any, Any],
        attention_mask: Any | None = None,
    ) -> tuple[Any, Any]:
        ctx = self.ctx
        norm_hidden, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1.forward(
            hidden_states, temb
        )
        norm_encoder, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context.forward(
            encoder_hidden_states, temb
        )

        attn_out, ctx_attn_out = self.attn.forward(
            norm_hidden, norm_encoder, image_rotary_emb, attention_mask
        )

        attn_out = ctx.expand_dims(gate_msa, axis=1) * attn_out
        hidden_states = hidden_states + attn_out
        norm_hidden = self.norm2(hidden_states)
        norm_hidden = norm_hidden * (1 + scale_mlp[:, None, :]) + shift_mlp[:, None, :]
        ff_out = self.ff.forward(norm_hidden)
        hidden_states = hidden_states + ctx.expand_dims(gate_mlp, axis=1) * ff_out

        ctx_attn_out = ctx.expand_dims(c_gate_msa, axis=1) * ctx_attn_out
        encoder_hidden_states = encoder_hidden_states + ctx_attn_out
        norm_encoder = self.norm2_context(encoder_hidden_states)
        norm_encoder = norm_encoder * (1 + c_scale_mlp[:, None, :]) + c_shift_mlp[:, None, :]
        encoder_hidden_states = encoder_hidden_states + ctx.expand_dims(c_gate_mlp, axis=1) * self.ff_context.forward(
            norm_encoder
        )
        return encoder_hidden_states, hidden_states


class _FiboSingleTransformerBlock:
    """FiboSingleTransformerBlock."""

    def __init__(self, dim: int, num_heads: int, head_dim: int, ctx: RuntimeContext):
        self.ctx = ctx
        self.norm = _FiboAdaLayerNormZeroSingle(dim, ctx)
        self.attn = _FiboSingleAttention(dim, num_heads, head_dim, ctx)
        self.proj_mlp = mx_nn.Linear(dim, int(dim * 4), bias=True)
        self.proj_out = mx_nn.Linear(int(dim * 4) + dim, dim, bias=True)

    def forward(
        self,
        hidden_states: Any,
        temb: Any,
        image_rotary_emb: tuple[Any, Any],
        attention_mask: Any | None = None,
    ) -> Any:
        ctx = self.ctx
        residual = hidden_states
        norm_hidden, gate = self.norm.forward(hidden_states, temb)
        attn_out = self.attn.forward(norm_hidden, image_rotary_emb, attention_mask)
        mlp_hidden = mx_nn.gelu_approx(self.proj_mlp(norm_hidden))
        combined = ctx.concat([attn_out, mlp_hidden], axis=-1)
        hidden_states = ctx.expand_dims(gate, axis=1) * self.proj_out(combined)
        return residual + hidden_states


class _AdaLayerNormContinuousOut:
    """AdaLayerNormContinuousOut — delegates to ``common/norm.apply_ada_layer_norm_continuous``."""

    def __init__(self, dim: int, ctx: RuntimeContext):
        self.ctx = ctx
        self._dim = dim
        self.norm = mx_nn.LayerNorm(dim, eps=1e-6, affine=False)
        self.linear = mx_nn.Linear(dim, dim * 2, bias=False)

    def forward(self, x: Any, c: Any) -> Any:
        return apply_ada_layer_norm_continuous(
            x, c,
            linear=self.linear, norm=self.norm,
            embedding_dim=self._dim, silu=mx_nn.silu,
            pre_linear_dtype=self.ctx.bfloat16(),
        )


class _BriaFiboTextProjection:
    """BriaFiboTextProjection — Linear wrapper for caption_projection."""

    def __init__(self, in_features: int, hidden_size: int):
        self.linear = mx_nn.Linear(in_features, hidden_size, bias=False)

    def __call__(self, caption: Any) -> Any:
        return self.linear(caption)

    def forward(self, caption: Any) -> Any:
        return self.linear(caption)


class FIBODiTMLX(TransformerBase):
    """FIBO / Bria4Transformer2DModel — Joint + Single DiT."""

    def __init__(self, config: FIBOConfig, ctx: RuntimeContext):
        self.config = config
        self.ctx = ctx
        nn = ctx
        dim = config.hidden_dim
        heads = config.num_heads
        head_dim = config.head_dim

        self.pos_embed = _FiboEmbedND(ctx)
        self.x_embedder = nn.Linear(config.in_channels, dim, bias=True)
        self.time_embed = _BriaFiboTimestepProjEmbeddings(dim, ctx)
        self.context_embedder = nn.Linear(config.text_dim, dim, bias=True)

        self.transformer_blocks = [
            _FiboJointTransformerBlock(dim, heads, head_dim, ctx)
            for _ in range(config.num_joint_layers)
        ]
        self.single_transformer_blocks = [
            _FiboSingleTransformerBlock(dim, heads, head_dim, ctx)
            for _ in range(config.num_single_layers)
        ]

        self.norm_out = _AdaLayerNormContinuousOut(dim, ctx)
        self.proj_out = nn.Linear(dim, config.out_channels, bias=True)
        self.caption_projection = [
            _BriaFiboTextProjection(config.text_encoder_dim, dim // 2)
            for _ in range(config.num_joint_layers + config.num_single_layers)
        ]

        self._build_param_map()

    def forward_cfg(
        self,
        latents: Any,
        timestep: Any,
        txt_embeds: Any,
        neg_embeds: Any | None,
        guidance: float,
        sigmas: Any | None = None,
        *,
        cfg_renorm: bool = False,
        cfg_renorm_min: float = 0.0,
        **conditioning: Any,
    ) -> Any:
        """Batched CFG — Reference stacks [uncond, cond] on batch axis 0."""
        if neg_embeds is None and not (
            txt_embeds is not None and int(txt_embeds.shape[0]) == 2
        ):
            return self.forward(
                latents,
                timestep,
                txt_embeds=txt_embeds,
                sigmas=sigmas,
                **conditioning,
            )
        if (
            neg_embeds is None
            and txt_embeds is not None
            and int(txt_embeds.shape[0]) == 2
        ):
            batched_latents = self.ctx.concat([latents, latents], axis=0)
            noise = self.forward(
                batched_latents,
                timestep,
                txt_embeds=txt_embeds,
                sigmas=sigmas,
                **conditioning,
            )
            noise_uncond = noise[0:1]
            noise_cond = noise[1:2]
            noise_pred = self.combine_cfg_noise(noise_cond, noise_uncond, guidance)
            if cfg_renorm:
                noise_pred = self.refine_cfg_noise(
                    noise_cond, noise_pred, cfg_renorm_min=cfg_renorm_min,
                )
            return noise_pred

        pos_kwargs: dict[str, Any] = {
            "txt_embeds": txt_embeds,
            "sigmas": sigmas,
            **conditioning,
        }
        neg_kwargs: dict[str, Any] = {
            "txt_embeds": neg_embeds,
            "sigmas": sigmas,
            **conditioning,
        }
        return predict_noise_cfg_batched(
            self.forward,
            self.ctx,
            latents,
            timestep,
            guidance=float(guidance),
            pos_kwargs=pos_kwargs,
            neg_kwargs=neg_kwargs,
            text_keys=FIBO_CFG_TEXT_KEYS,
            combine_cfg_noise=self.combine_cfg_noise,
            refine_cfg_noise=self.refine_cfg_noise,
            cfg_renorm=cfg_renorm,
            cfg_renorm_min=cfg_renorm_min,
        )

    def _build_param_map(self):
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
        sigmas: Any | None = None,
        text_encoder_layers: list[Any] | None = None,
        **conditioning: Any,
    ) -> Any:
        ctx = self.ctx
        cfg = self.config
        dim = cfg.hidden_dim
        B = latents.shape[0]
        _, _, H, W = latents.shape
        text_encoder_layers = conditioning.get("text_encoder_layers", text_encoder_layers)
        conditioning_latents = conditioning.get("conditioning_latents")
        conditioning_image_ids = conditioning.get("conditioning_image_ids")

        # Pack noise latents [B,48,H,W] → [B, H*W, 48]; optional FIBO-Edit concat on sequence axis.
        noise_packed = ctx.permute(latents, (0, 2, 3, 1))
        noise_packed = ctx.reshape(noise_packed, (B, H * W, cfg.in_channels))
        img_seq_len = noise_packed.shape[1]
        cond_seq_len = 0
        if conditioning_latents is not None:
            cond = conditioning_latents
            if int(cond.shape[0]) == 1 and B > 1:
                cond = ctx.broadcast_to(cond, (B, *cond.shape[1:]))
            _, _, c_h, c_w = cond.shape
            if c_h != H or c_w != W:
                raise RuntimeError(
                    f"FIBO edit conditioning latents spatial {c_h}x{c_w} != noise latents {H}x{W}"
                )
            cond_packed = ctx.permute(cond, (0, 2, 3, 1))
            cond_packed = ctx.reshape(cond_packed, (B, c_h * c_w, cfg.in_channels))
            cond_seq_len = int(cond_packed.shape[1])
            hidden_states = ctx.concat([noise_packed, cond_packed], axis=1)
        else:
            hidden_states = noise_packed

        hidden_states = self.x_embedder(hidden_states)

        if txt_embeds is not None:
            encoder_hidden_states = self.context_embedder(txt_embeds)
            txt_len = encoder_hidden_states.shape[1]
        else:
            encoder_hidden_states = ctx.zeros((B, 0, dim))
            txt_len = 0

        # Time embedding
        timestep_embed_value = conditioning.get("timestep_embed_value")
        if timestep_embed_value is not None:
            t_val = float(timestep_embed_value)
        elif sigmas is not None:
            t_idx = int(timestep)
            n = int(sigmas.shape[0]) if hasattr(sigmas, "shape") else len(sigmas)
            sigma_t = sigmas[t_idx] if t_idx < n else sigmas[-1] if n > 0 else 1.0
            t_val = float(ctx.reshape(ctx.array(sigma_t), (-1,))[0]) * 1000.0
        else:
            tv = timestep
            if isinstance(tv, mx.array):
                t_val = float(ctx.reshape(tv, (-1,))[0])
            else:
                t_val = float(tv)
            if t_val <= 1.0 + 1e-5:
                t_val *= 1000.0
        t_batch = ctx.full((B,), t_val, dtype=ctx.bfloat16())
        t_proj = sinusoidal_timestep_proj(ctx, t_batch, 256, flip_sin_to_cos=True)
        temb = self.time_embed.forward(t_proj).astype(ctx.bfloat16())

        # RoPE
        txt_ids = ctx.zeros((txt_len, 3), dtype=ctx.float32())
        img_h = ctx.arange(0, H, dtype=ctx.float32())
        img_w = ctx.arange(0, W, dtype=ctx.float32())
        img_h = ctx.reshape(ctx.broadcast_to(ctx.expand_dims(img_h, axis=1), (H, W)), (-1,))
        img_w = ctx.reshape(ctx.broadcast_to(ctx.expand_dims(img_w, axis=0), (H, W)), (-1,))
        img_ids = ctx.stack([ctx.zeros((H * W,), dtype=ctx.float32()), img_h, img_w], axis=1)
        if conditioning_image_ids is not None:
            cond_ids = conditioning_image_ids
            if cond_ids.ndim == 3:
                cond_ids = cond_ids[0]
            img_ids = ctx.concat([img_ids, cond_ids.astype(ctx.float32())], axis=0)
        ids = ctx.concat([txt_ids, img_ids], axis=0)
        ids = ctx.expand_dims(ids, axis=0)
        cos, sin = self.pos_embed.forward(ids)

        # Attention mask — Reference: full bidirectional over prompt + latent (+ conditioning) tokens
        prompt_mask = ctx.ones((B, txt_len), dtype=ctx.float32())
        latent_mask = ctx.ones((B, img_seq_len), dtype=ctx.float32())
        if cond_seq_len > 0:
            cond_mask = ctx.ones((B, cond_seq_len), dtype=ctx.float32())
            attention_mask_2d = ctx.concat([prompt_mask, latent_mask, cond_mask], axis=1)
        else:
            attention_mask_2d = ctx.concat([prompt_mask, latent_mask], axis=1)
        attn_matrix = ctx.einsum("bi,bj->bij", attention_mask_2d, attention_mask_2d)
        min_dtype = float(np.finfo(np.float32).min)
        attn_matrix = ctx.where(
            attn_matrix == 1,
            ctx.zeros_like(attn_matrix),
            ctx.ones_like(attn_matrix) * min_dtype,
        )
        attn_matrix = ctx.expand_dims(attn_matrix, axis=1).astype(ctx.bfloat16())

        # Caption projection — Reference projects all layers once, then splices per block
        total_layers = cfg.num_joint_layers + cfg.num_single_layers
        projected_layers: list[Any | None]
        if text_encoder_layers is not None and len(text_encoder_layers) > 0:
            if len(text_encoder_layers) >= total_layers:
                text_encoder_layers = text_encoder_layers[len(text_encoder_layers) - total_layers :]
            else:
                text_encoder_layers = text_encoder_layers + [text_encoder_layers[-1]] * (
                    total_layers - len(text_encoder_layers)
                )
            projected_layers = [
                self.caption_projection[i](layer)
                for i, layer in enumerate(text_encoder_layers)
            ]
        else:
            projected_layers = [None] * total_layers

        # Joint blocks
        for i, block in enumerate(self.transformer_blocks):
            if projected_layers[i] is not None:
                encoder_half = encoder_hidden_states[:, :, : dim // 2]
                encoder_hidden_states = ctx.concat(
                    [encoder_half, projected_layers[i]], axis=-1
                )
            encoder_hidden_states, hidden_states = block.forward(
                hidden_states, encoder_hidden_states, temb, (cos, sin), attn_matrix
            )
            if getattr(ctx, "backend", None) == "mlx":
                ctx.eval(encoder_hidden_states, hidden_states)

        # Single blocks
        x = ctx.concat([encoder_hidden_states, hidden_states], axis=1)
        for i, block in enumerate(self.single_transformer_blocks):
            block_idx = cfg.num_joint_layers + i
            if projected_layers[block_idx] is not None:
                enc_len = encoder_hidden_states.shape[1]
                enc_part = x[:, :enc_len, :]
                img_part = x[:, enc_len:, :]
                enc_half = enc_part[:, :, : dim // 2]
                enc_part = ctx.concat([enc_half, projected_layers[block_idx]], axis=-1)
                x = ctx.concat([enc_part, img_part], axis=1)
            x = block.forward(x, temb, (cos, sin), attn_matrix)
            if getattr(ctx, "backend", None) == "mlx":
                ctx.eval(x)

        hidden_states = x[:, txt_len:, :]
        if cond_seq_len > 0:
            hidden_states = hidden_states[:, :img_seq_len, :]
        hidden_states = self.norm_out.forward(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)

        # Unpack latents [B, H*W, 48] → [B, 48, H, W]
        hidden_states = ctx.reshape(hidden_states, (B, H, W, cfg.out_channels))
        return ctx.permute(hidden_states, (0, 3, 1, 2))

    def load_weights(
        self,
        weights,
        strict=False,
        ctx=None,
        *,
        bundle_affine_bits=None,
        inference_mode=None,
    ):
        load_ctx = ctx if ctx is not None else self.ctx
        loaded, skipped = super().load_weights(
            weights,
            strict=strict,
            ctx=load_ctx,
            bundle_affine_bits=bundle_affine_bits,
            inference_mode=inference_mode,
        )
        if inference_mode is None or getattr(inference_mode, "kind", "dense") != "quantized":
            self._cast_param_map_dtype(load_ctx.bfloat16())
        return loaded, skipped
