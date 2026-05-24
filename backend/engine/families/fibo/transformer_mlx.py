"""
FIBO Transformer — Bria4Transformer2DModel compatible.

Joint MM-DiT (8 blocks) → Single DiT (38 blocks).
Diffusers weight keys match directly (via _collect_params list traversal).
"""
from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as mx_nn

from backend.engine.common._base import TransformerBase, _collect_params
from backend.engine.common.cfg_batch import FIBO_CFG_TEXT_KEYS, predict_noise_cfg_batched
from backend.engine.common.embeddings import sinusoidal_timestep_proj
from backend.engine.config.model_configs import FIBOConfig
from backend.engine.runtime._base import RuntimeContext


class _FiboEmbedND:
    """mflux FiboEmbedND — 3-axis RoPE (axes_dim=[16,56,56])."""

    def __init__(self, ctx: RuntimeContext, theta: float = 10000.0):
        self.ctx = ctx
        self.theta = theta
        self.axes_dim = [16, 56, 56]

    def forward(self, ids: Any) -> tuple[Any, Any]:
        if ids.ndim == 3 and int(ids.shape[0]) == 1:
            ids = ids[0]
        pos = ids.astype(mx.float32)
        cos_out = []
        sin_out = []
        for i, dim in enumerate(self.axes_dim):
            cos_axis, sin_axis = self._rope_1d(pos[:, i], dim)
            cos_out.append(cos_axis)
            sin_out.append(sin_axis)
        return mx.concatenate(cos_out, axis=-1), mx.concatenate(sin_out, axis=-1)

    def _rope_1d(self, pos: Any, dim: int) -> tuple[Any, Any]:
        pos = pos.astype(mx.float32)
        if pos.ndim != 1:
            pos = mx.reshape(pos, (-1,))
        freqs = 1.0 / (self.theta ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))
        angles = pos[:, None] * freqs[None, :]
        cos_base = mx.cos(angles)
        sin_base = mx.sin(angles)
        cos = mx.reshape(mx.stack([cos_base, cos_base], axis=-1), (pos.shape[0], -1))
        sin = mx.reshape(mx.stack([sin_base, sin_base], axis=-1), (pos.shape[0], -1))
        return cos, sin


class _TimestepEmbedder:
    """Wrapper to match diffusers weight key path."""

    def __init__(self, dim: int):
        self.linear_1 = mx_nn.Linear(256, dim, bias=True)
        self.linear_2 = mx_nn.Linear(dim, dim, bias=True)


class _BriaFiboTimestepProjEmbeddings:
    """mflux BriaFiboTimestepProjEmbeddings — sinusoidal + MLP."""

    def __init__(self, dim: int, ctx: RuntimeContext):
        self.ctx = ctx
        self.timestep_embedder = _TimestepEmbedder(dim)

    def forward(self, sample: Any) -> Any:
        x = mx_nn.silu(self.timestep_embedder.linear_1(sample))
        return self.timestep_embedder.linear_2(x)


class _FiboAdaLayerNormZero:
    """mflux FiboAdaLayerNormZero — SiLU + Linear → 6 chunks."""

    def __init__(self, dim: int, ctx: RuntimeContext):
        self.ctx = ctx
        self.linear = mx_nn.Linear(dim, dim * 6, bias=True)

    def forward(self, hidden_states: Any, text_embeddings: Any) -> tuple[Any, ...]:
        emb = self.linear(mx_nn.silu(text_embeddings))
        chunk = hidden_states.shape[-1]
        shift_msa = emb[:, 0 * chunk : 1 * chunk]
        scale_msa = emb[:, 1 * chunk : 2 * chunk]
        gate_msa = emb[:, 2 * chunk : 3 * chunk]
        shift_mlp = emb[:, 3 * chunk : 4 * chunk]
        scale_mlp = emb[:, 4 * chunk : 5 * chunk]
        gate_mlp = emb[:, 5 * chunk : 6 * chunk]
        norm = self._layer_norm(hidden_states)
        hidden_states = norm * (1 + scale_msa[:, None, :]) + shift_msa[:, None, :]
        return hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp

    @staticmethod
    def _layer_norm(x: Any) -> Any:
        x_f32 = x.astype(mx.float32)
        mean = mx.mean(x_f32, axis=-1, keepdims=True)
        var = mx.mean((x_f32 - mean) ** 2, axis=-1, keepdims=True)
        y = (x_f32 - mean) / mx.sqrt(var + 1e-6)
        return y.astype(x.dtype)


class _FiboAdaLayerNormZeroSingle:
    """mflux AdaLayerNormZeroSingle — SiLU + Linear → shift/scale/gate."""

    def __init__(self, dim: int, ctx: RuntimeContext):
        self.ctx = ctx
        self.linear = mx_nn.Linear(dim, dim * 3, bias=True)
        self.norm = mx_nn.LayerNorm(dim, eps=1e-6, affine=False)

    def forward(self, hidden_states: Any, text_embeddings: Any) -> tuple[Any, Any]:
        emb = self.linear(mx_nn.silu(text_embeddings))
        chunk = hidden_states.shape[-1]
        shift_msa = emb[:, 0 * chunk : 1 * chunk]
        scale_msa = emb[:, 1 * chunk : 2 * chunk]
        gate_msa = emb[:, 2 * chunk : 3 * chunk]
        hidden_states = self.norm(hidden_states) * (1 + scale_msa[:, None, :]) + shift_msa[:, None, :]
        return hidden_states, gate_msa


class _FiboJointAttention:
    """mflux FiboJointAttention — joint QKV for image + context streams."""

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
        batch_size, seq_img, _ = hidden_states.shape
        _, seq_ctx, _ = encoder_hidden_states.shape
        cos, sin = image_rotary_emb

        query = self.to_q(hidden_states)
        key = self.to_k(hidden_states)
        value = self.to_v(hidden_states)

        enc_query = self.add_q_proj(encoder_hidden_states)
        enc_key = self.add_k_proj(encoder_hidden_states)
        enc_value = self.add_v_proj(encoder_hidden_states)

        query = mx.reshape(query, (batch_size, seq_img, self.num_heads, self.head_dim))
        key = mx.reshape(key, (batch_size, seq_img, self.num_heads, self.head_dim))
        value = mx.reshape(value, (batch_size, seq_img, self.num_heads, self.head_dim))

        enc_query = mx.reshape(enc_query, (batch_size, seq_ctx, self.num_heads, self.head_dim))
        enc_key = mx.reshape(enc_key, (batch_size, seq_ctx, self.num_heads, self.head_dim))
        enc_value = mx.reshape(enc_value, (batch_size, seq_ctx, self.num_heads, self.head_dim))

        query = self.norm_q(query.astype(mx.float32)).astype(query.dtype)
        key = self.norm_k(key.astype(mx.float32)).astype(key.dtype)
        enc_query = self.norm_added_q(enc_query.astype(mx.float32)).astype(enc_query.dtype)
        enc_key = self.norm_added_k(enc_key.astype(mx.float32)).astype(enc_key.dtype)

        query = mx.concatenate([enc_query, query], axis=1)
        key = mx.concatenate([enc_key, key], axis=1)
        value = mx.concatenate([enc_value, value], axis=1)

        query = self._apply_rotary_emb(query, cos, sin)
        key = self._apply_rotary_emb(key, cos, sin)

        query_bhsd = mx.transpose(query, (0, 2, 1, 3))
        key_bhsd = mx.transpose(key, (0, 2, 1, 3))
        value_bhsd = mx.transpose(value, (0, 2, 1, 3))

        scale = 1.0 / mx.sqrt(mx.array(self.head_dim, dtype=query_bhsd.dtype))
        from backend.engine.common.attention import scaled_dot_product_attention_bhsd_mx

        attn_output = scaled_dot_product_attention_bhsd_mx(
            mx, query_bhsd, key_bhsd, value_bhsd, scale=scale, mask=attention_mask
        )

        attn_output = mx.transpose(attn_output, (0, 2, 1, 3))
        attn_output = mx.reshape(attn_output, (batch_size, seq_img + seq_ctx, self.inner_dim))

        context_attn_output = attn_output[:, :seq_ctx, :]
        hidden_attn_output = attn_output[:, seq_ctx:, :]

        hidden_attn_output = self.to_out[0](hidden_attn_output)
        context_attn_output = self.to_add_out(context_attn_output)
        return hidden_attn_output, context_attn_output

    @staticmethod
    def _apply_rotary_emb(x: Any, cos: Any, sin: Any) -> Any:
        bsz, seq_len, num_heads, head_dim = x.shape
        cos = mx.expand_dims(mx.expand_dims(cos, axis=0), axis=2)
        sin = mx.expand_dims(mx.expand_dims(sin, axis=0), axis=2)
        x2 = x.reshape(bsz, seq_len, num_heads, -1, 2)
        x_real = x2[..., 0]
        x_imag = x2[..., 1]
        x_rotated = mx.stack([-x_imag, x_real], axis=-1).reshape(bsz, seq_len, num_heads, head_dim)
        return (x.astype(mx.float32) * cos + x_rotated.astype(mx.float32) * sin).astype(x.dtype)


class _FiboSingleAttention:
    """mflux FiboSingleAttention — single-stream attention."""

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
        batch_size, seq_len, _ = hidden_states.shape
        cos, sin = image_rotary_emb

        query = self.to_q(hidden_states)
        key = self.to_k(hidden_states)
        value = self.to_v(hidden_states)

        query = mx.reshape(query, (batch_size, seq_len, self.num_heads, self.head_dim))
        key = mx.reshape(key, (batch_size, seq_len, self.num_heads, self.head_dim))
        value = mx.reshape(value, (batch_size, seq_len, self.num_heads, self.head_dim))

        query = self.norm_q(query.astype(mx.float32)).astype(query.dtype)
        key = self.norm_k(key.astype(mx.float32)).astype(key.dtype)

        query = _FiboJointAttention._apply_rotary_emb(query, cos, sin)
        key = _FiboJointAttention._apply_rotary_emb(key, cos, sin)

        query_bhsd = mx.transpose(query, (0, 2, 1, 3))
        key_bhsd = mx.transpose(key, (0, 2, 1, 3))
        value_bhsd = mx.transpose(value, (0, 2, 1, 3))

        scale = 1.0 / mx.sqrt(mx.array(self.head_dim, dtype=query_bhsd.dtype))
        from backend.engine.common.attention import scaled_dot_product_attention_bhsd_mx

        attn_output = scaled_dot_product_attention_bhsd_mx(
            mx, query_bhsd, key_bhsd, value_bhsd, scale=scale, mask=attention_mask
        )

        attn_output = mx.transpose(attn_output, (0, 2, 1, 3))
        attn_output = mx.reshape(attn_output, (batch_size, seq_len, self.inner_dim))
        return attn_output


class _FiboGELU:
    """mflux FiboGELU — Linear + gelu_approx."""

    def __init__(self, dim_in: int, dim_out: int):
        self.proj = mx_nn.Linear(dim_in, dim_out, bias=True)

    def __call__(self, x: Any) -> Any:
        return mx_nn.gelu_approx(self.proj(x))

    def forward(self, x: Any) -> Any:
        return mx_nn.gelu_approx(self.proj(x))


class _FiboFeedForward:
    """mflux FiboFeedForward — GELU-approx + Linear."""

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
    """mflux FiboJointTransformerBlock."""

    def __init__(self, dim: int, num_heads: int, head_dim: int, ctx: RuntimeContext):
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
        norm_hidden, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1.forward(
            hidden_states, temb
        )
        norm_encoder, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context.forward(
            encoder_hidden_states, temb
        )

        attn_out, ctx_attn_out = self.attn.forward(
            norm_hidden, norm_encoder, image_rotary_emb, attention_mask
        )

        attn_out = mx.expand_dims(gate_msa, axis=1) * attn_out
        hidden_states = hidden_states + attn_out
        norm_hidden = self.norm2(hidden_states)
        norm_hidden = norm_hidden * (1 + scale_mlp[:, None, :]) + shift_mlp[:, None, :]
        ff_out = self.ff.forward(norm_hidden)
        hidden_states = hidden_states + mx.expand_dims(gate_mlp, axis=1) * ff_out

        ctx_attn_out = mx.expand_dims(c_gate_msa, axis=1) * ctx_attn_out
        encoder_hidden_states = encoder_hidden_states + ctx_attn_out
        norm_encoder = self.norm2_context(encoder_hidden_states)
        norm_encoder = norm_encoder * (1 + c_scale_mlp[:, None, :]) + c_shift_mlp[:, None, :]
        encoder_hidden_states = encoder_hidden_states + mx.expand_dims(c_gate_mlp, axis=1) * self.ff_context.forward(
            norm_encoder
        )
        return encoder_hidden_states, hidden_states


class _FiboSingleTransformerBlock:
    """mflux FiboSingleTransformerBlock."""

    def __init__(self, dim: int, num_heads: int, head_dim: int, ctx: RuntimeContext):
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
        residual = hidden_states
        norm_hidden, gate = self.norm.forward(hidden_states, temb)
        attn_out = self.attn.forward(norm_hidden, image_rotary_emb, attention_mask)
        mlp_hidden = mx_nn.gelu_approx(self.proj_mlp(norm_hidden))
        combined = mx.concatenate([attn_out, mlp_hidden], axis=-1)
        hidden_states = mx.expand_dims(gate, axis=1) * self.proj_out(combined)
        return residual + hidden_states


class _AdaLayerNormContinuousOut:
    """mflux AdaLayerNormContinuousOut — SiLU + Linear → scale/shift."""

    def __init__(self, dim: int, ctx: RuntimeContext):
        self.ctx = ctx
        self.norm = mx_nn.LayerNorm(dim, eps=1e-6, affine=False)
        self.linear = mx_nn.Linear(dim, dim * 2, bias=False)

    def forward(self, x: Any, c: Any) -> Any:
        v = self.linear(mx_nn.silu(c).astype(mx.bfloat16))
        scale = v[:, : x.shape[-1]]
        shift = v[:, x.shape[-1] :]
        x = self.norm(x)
        return x * (1 + scale[:, None, :]) + shift[:, None, :]


class _BriaFiboTextProjection:
    """mflux BriaFiboTextProjection — Linear wrapper for caption_projection."""

    def __init__(self, in_features: int, hidden_size: int):
        self.linear = mx_nn.Linear(in_features, hidden_size, bias=False)

    def __call__(self, caption: Any) -> Any:
        return self.linear(caption)

    def forward(self, caption: Any) -> Any:
        return self.linear(caption)


class FIBOTransformer(TransformerBase):
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
        """Batched CFG — mflux stacks [uncond, cond] on batch axis 0."""
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

        # Pack latents [B,48,H,W] → [B, H*W, 48]
        hidden_states = mx.transpose(latents, (0, 2, 3, 1))
        hidden_states = mx.reshape(hidden_states, (B, H * W, cfg.in_channels))
        hidden_states = self.x_embedder(hidden_states)
        img_seq_len = hidden_states.shape[1]

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
            t_val = float(mx.reshape(ctx.array(sigma_t), (-1,))[0]) * 1000.0
        else:
            tv = timestep
            if isinstance(tv, mx.array):
                t_val = float(mx.reshape(tv, (-1,))[0])
            else:
                t_val = float(tv)
            if t_val <= 1.0 + 1e-5:
                t_val *= 1000.0
        t_batch = mx.full((B,), t_val, dtype=mx.bfloat16)
        t_proj = sinusoidal_timestep_proj(ctx, t_batch, 256, flip_sin_to_cos=True)
        temb = self.time_embed.forward(t_proj).astype(mx.bfloat16)

        # RoPE
        txt_ids = mx.zeros((txt_len, 3), dtype=mx.float32)
        img_h = mx.arange(0, H, dtype=mx.float32)
        img_w = mx.arange(0, W, dtype=mx.float32)
        img_h = mx.reshape(mx.broadcast_to(img_h[:, None], (H, W)), (-1,))
        img_w = mx.reshape(mx.broadcast_to(img_w[None, :], (H, W)), (-1,))
        img_ids = mx.stack([mx.zeros(H * W, dtype=mx.float32), img_h, img_w], axis=1)
        ids = mx.concatenate([txt_ids, img_ids], axis=0)
        ids = mx.expand_dims(ids, axis=0)
        cos, sin = self.pos_embed.forward(ids)

        # Attention mask — mflux: full bidirectional over prompt + latent tokens
        prompt_mask = mx.ones((B, txt_len), dtype=mx.float32)
        latent_mask = mx.ones((B, img_seq_len), dtype=mx.float32)
        attention_mask_2d = mx.concatenate([prompt_mask, latent_mask], axis=1)
        attn_matrix = mx.einsum("bi,bj->bij", attention_mask_2d, attention_mask_2d)
        min_dtype = mx.finfo(mx.float32).min
        attn_matrix = mx.where(
            attn_matrix == 1,
            mx.zeros_like(attn_matrix),
            mx.ones_like(attn_matrix) * min_dtype,
        )
        attn_matrix = mx.expand_dims(attn_matrix, axis=1).astype(mx.bfloat16)

        # Caption projection (mflux applies before DiT blocks)
        total_layers = cfg.num_joint_layers + cfg.num_single_layers
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
                encoder_hidden_states = mx.concatenate([encoder_half, projected_layers[i]], axis=-1)
            encoder_hidden_states, hidden_states = block.forward(
                hidden_states, encoder_hidden_states, temb, (cos, sin), attn_matrix
            )
            if getattr(ctx, "backend", None) == "mlx":
                ctx.eval(encoder_hidden_states, hidden_states)

        # Single blocks
        x = mx.concatenate([encoder_hidden_states, hidden_states], axis=1)
        for i, block in enumerate(self.single_transformer_blocks):
            block_idx = cfg.num_joint_layers + i
            if projected_layers[block_idx] is not None:
                enc_len = encoder_hidden_states.shape[1]
                enc_part = x[:, :enc_len, :]
                img_part = x[:, enc_len:, :]
                enc_half = enc_part[:, :, : dim // 2]
                enc_part = mx.concatenate([enc_half, projected_layers[block_idx]], axis=-1)
                x = mx.concatenate([enc_part, img_part], axis=1)
            x = block.forward(x, temb, (cos, sin), attn_matrix)
            if getattr(ctx, "backend", None) == "mlx":
                ctx.eval(x)

        hidden_states = x[:, txt_len:, :]
        hidden_states = self.norm_out.forward(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)

        # Unpack latents [B, H*W, 48] → [B, 48, H, W]
        hidden_states = mx.reshape(hidden_states, (B, H, W, cfg.out_channels))
        hidden_states = mx.transpose(hidden_states, (0, 3, 1, 2))
        return hidden_states

    def load_weights(self, weights, strict=False, ctx=None, *, bundle_affine_bits=None):
        load_ctx = ctx if ctx is not None else self.ctx
        loaded, skipped = super().load_weights(
            weights, strict=strict, ctx=load_ctx, bundle_affine_bits=bundle_affine_bits
        )
        self._cast_param_map_dtype(mx.bfloat16)
        return loaded, skipped
