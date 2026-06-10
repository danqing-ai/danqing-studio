"""CogView4 Transformer — MLX port of diffusers ``CogView4Transformer2DModel``."""
from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from backend.engine.common.model.base import TransformerBase
from backend.engine.common.ops.attention import apply_rope_real_unbind_dim2, scaled_dot_product_attention_bhsd_mx
from backend.engine.common.ops.embeddings import sinusoidal_timestep_proj
from backend.engine.config.model_configs import CogView4Config
from backend.engine.runtime._base import RuntimeContext


def _gelu_approx(x: Any) -> Any:
    return nn.gelu_approx(x)


class _CogView4RotaryPosEmbed:
    """2-axis RoPE for latent patches (diffusers ``CogView4RotaryPosEmbed``)."""

    def __init__(self, dim: int, patch_size: int, rope_axes_dim: tuple[int, int], theta: float = 10000.0):
        self.dim = int(dim)
        self.patch_size = int(patch_size)
        self.rope_axes_dim = (int(rope_axes_dim[0]), int(rope_axes_dim[1]))
        self.theta = float(theta)

    def forward(self, ctx: RuntimeContext, hidden_states: Any) -> tuple[Any, Any]:
        _, _, height, width = (int(hidden_states.shape[0]), int(hidden_states.shape[1]),
                              int(hidden_states.shape[2]), int(hidden_states.shape[3]))
        height = height // self.patch_size
        width = width // self.patch_size
        dim_h = self.dim // 2
        dim_w = self.dim // 2
        h_inv = 1.0 / (self.theta ** (np.arange(0, dim_h, 2, dtype=np.float32)[: (dim_h // 2)] / dim_h))
        w_inv = 1.0 / (self.theta ** (np.arange(0, dim_w, 2, dtype=np.float32)[: (dim_w // 2)] / dim_w))
        h_seq = np.arange(self.rope_axes_dim[0], dtype=np.float32)
        w_seq = np.arange(self.rope_axes_dim[1], dtype=np.float32)
        freqs_h = np.outer(h_seq, h_inv)
        freqs_w = np.outer(w_seq, w_inv)
        inner_h = (np.arange(height, dtype=np.float32) * self.rope_axes_dim[0] / height).astype(np.int64)
        inner_w = (np.arange(width, dtype=np.float32) * self.rope_axes_dim[1] / width).astype(np.int64)
        freqs_h = freqs_h[inner_h]
        freqs_w = freqs_w[inner_w]
        freqs_h = freqs_h[:, None, :]
        freqs_w = freqs_w[None, :, :]
        freqs_h = np.broadcast_to(freqs_h, (height, width, freqs_h.shape[-1]))
        freqs_w = np.broadcast_to(freqs_w, (height, width, freqs_w.shape[-1]))
        freqs = np.concatenate([freqs_h, freqs_w], axis=-1)
        freqs = np.concatenate([freqs, freqs], axis=-1)
        freqs = freqs.reshape(height * width, -1)
        cos = ctx.array(np.cos(freqs), dtype=ctx.float32())
        sin = ctx.array(np.sin(freqs), dtype=ctx.float32())
        return cos, sin


class _CogView4PatchEmbed(nn.Module):
    def __init__(self, in_channels: int, hidden_size: int, patch_size: int, text_hidden_size: int):
        super().__init__()
        self.patch_size = int(patch_size)
        self.proj = nn.Linear(in_channels * patch_size * patch_size, hidden_size, bias=True)
        self.text_proj = nn.Linear(text_hidden_size, hidden_size, bias=True)

    def __call__(self, hidden_states: Any, encoder_hidden_states: Any) -> tuple[Any, Any]:
        p = self.patch_size
        b, c, h, w = (int(hidden_states.shape[0]), int(hidden_states.shape[1]),
                      int(hidden_states.shape[2]), int(hidden_states.shape[3]))
        ph, pw = h // p, w // p
        x = hidden_states.reshape(b, c, ph, p, pw, p)
        x = x.transpose(0, 2, 4, 1, 3, 5).reshape(b, ph * pw, c * p * p)
        x = self.proj(x)
        enc = self.text_proj(encoder_hidden_states)
        return x, enc


class _TimestepEmbedder(nn.Module):
    def __init__(self, in_channels: int, time_embed_dim: int):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim, bias=True)
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim, bias=True)

    def __call__(self, sample: Any) -> Any:
        x = nn.silu(self.linear_1(sample))
        return self.linear_2(x)


class _ConditionEmbedder(nn.Module):
    """PixArtAlphaTextProjection — silu between linears."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear_1 = nn.Linear(in_features, out_features, bias=True)
        self.linear_2 = nn.Linear(out_features, out_features, bias=True)

    def __call__(self, x: Any) -> Any:
        return self.linear_2(nn.silu(self.linear_1(x)))


class _CogView4TimeConditionEmbed(nn.Module):
    def __init__(self, cfg: CogView4Config):
        super().__init__()
        inner = cfg.num_attention_heads * cfg.attention_head_dim
        pooled = 3 * 2 * cfg.condition_dim
        self.timestep_embedder = _TimestepEmbedder(inner, cfg.time_embed_dim)
        self.condition_embedder = _ConditionEmbedder(pooled, cfg.time_embed_dim)
        self._condition_dim = cfg.condition_dim
        self._timesteps_dim = inner

    def __call__(
        self,
        ctx: RuntimeContext,
        timestep: Any,
        original_size: Any,
        target_size: Any,
        crop_coords: Any,
        hidden_dtype: Any,
    ) -> Any:
        t_proj = sinusoidal_timestep_proj(
            ctx, timestep.astype(ctx.float32()), self._timesteps_dim, flip_sin_to_cos=True, downscale_freq_shift=0.0
        )
        def _cond_proj(coords: Any) -> Any:
            batch = int(coords.shape[0])
            flat = coords.reshape(batch, -1).astype(ctx.float32())
            parts = [
                sinusoidal_timestep_proj(
                    ctx,
                    flat[:, i],
                    self._condition_dim,
                    flip_sin_to_cos=True,
                    downscale_freq_shift=0.0,
                )
                for i in range(int(flat.shape[1]))
            ]
            return ctx.concat(parts, axis=-1)

        cond_parts = [_cond_proj(original_size), _cond_proj(crop_coords), _cond_proj(target_size)]
        condition_proj = ctx.concat(cond_parts, axis=-1)
        t_emb = self.timestep_embedder(t_proj.astype(hidden_dtype))
        c_emb = self.condition_embedder(condition_proj.astype(hidden_dtype))
        return t_emb + c_emb


class _CogView4AdaLayerNormZero(nn.Module):
    def __init__(self, embedding_dim: int, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-5, affine=False)
        self.norm_context = nn.LayerNorm(dim, eps=1e-5, affine=False)
        self.linear = nn.Linear(embedding_dim, 12 * dim, bias=True)

    def __call__(self, hidden_states: Any, encoder_hidden_states: Any, temb: Any) -> tuple[Any, ...]:
        n_h = self.norm(hidden_states)
        n_c = self.norm_context(encoder_hidden_states)
        emb = self.linear(temb)
        c = emb.shape[-1] // 12
        parts = [emb[:, i * c:(i + 1) * c] for i in range(12)]
        (
            shift_msa, c_shift_msa, scale_msa, c_scale_msa, gate_msa, c_gate_msa,
            shift_mlp, c_shift_mlp, scale_mlp, c_scale_mlp, gate_mlp, c_gate_mlp,
        ) = parts
        hidden_states = n_h * (1 + scale_msa[:, None, :]) + shift_msa[:, None, :]
        encoder_hidden_states = n_c * (1 + c_scale_msa[:, None, :]) + c_shift_msa[:, None, :]
        return (
            hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp,
            encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp,
        )


class _CogView4JointAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, head_dim: int):
        super().__init__()
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.to_q = nn.Linear(dim, dim, bias=True)
        self.to_k = nn.Linear(dim, dim, bias=True)
        self.to_v = nn.Linear(dim, dim, bias=True)
        self.to_out = [nn.Linear(dim, dim, bias=True)]
        self.norm_q = nn.LayerNorm(head_dim, eps=1e-5, affine=False)
        self.norm_k = nn.LayerNorm(head_dim, eps=1e-5, affine=False)

    def __call__(
        self,
        ctx: RuntimeContext,
        hidden_states: Any,
        encoder_hidden_states: Any,
        *,
        image_rotary_emb: tuple[Any, Any] | None,
        attention_mask: Any | None = None,
    ) -> tuple[Any, Any]:
        b = int(encoder_hidden_states.shape[0])
        text_len = int(encoder_hidden_states.shape[1])
        mixed = ctx.concat([encoder_hidden_states, hidden_states], axis=1)
        h, d = self.num_heads, self.head_dim
        q = self.to_q(mixed).reshape(b, -1, h, d).transpose(0, 2, 1, 3)
        k = self.to_k(mixed).reshape(b, -1, h, d).transpose(0, 2, 1, 3)
        v = self.to_v(mixed).reshape(b, -1, h, d).transpose(0, 2, 1, 3)
        q = self.norm_q(q)
        k = self.norm_k(k)
        if image_rotary_emb is not None:
            cos, sin = image_rotary_emb
            cos = ctx.expand_dims(ctx.expand_dims(cos, axis=0), axis=1)
            sin = ctx.expand_dims(ctx.expand_dims(sin, axis=0), axis=1)
            q_img = apply_rope_real_unbind_dim2(ctx, q[:, :, text_len:, :], cos, sin)
            k_img = apply_rope_real_unbind_dim2(ctx, k[:, :, text_len:, :], cos, sin)
            q = ctx.concat([q[:, :, :text_len, :], q_img], axis=2)
            k = ctx.concat([k[:, :, :text_len, :], k_img], axis=2)
        mask = None
        if attention_mask is not None:
            text_mask = attention_mask.astype(ctx.float32())
            img_len = int(hidden_states.shape[1])
            ones_img = ctx.ones((b, img_len), dtype=ctx.float32())
            mix = ctx.concat([text_mask, ones_img], axis=1)
            attn_matrix = ctx.einsum("bi,bj->bij", mix, mix)
            min_v = float(np.finfo(np.float32).min)
            mask = ctx.where(attn_matrix > 0, ctx.zeros_like(attn_matrix), ctx.full(attn_matrix.shape, min_v))
            mask = ctx.expand_dims(mask, axis=1).astype(q.dtype)
        scale = 1.0 / (d ** 0.5)
        out = scaled_dot_product_attention_bhsd_mx(mx, q, k, v, scale=scale, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(b, text_len + int(hidden_states.shape[1]), h * d)
        out = self.to_out[0](out)
        enc_out, img_out = out[:, :text_len, :], out[:, text_len:, :]
        return img_out, enc_out


class _CogView4FFGelu(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(dim, dim * 4, bias=True)

    def __call__(self, x: Any) -> Any:
        return _gelu_approx(self.proj(x))


class _CogView4FF(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = [_CogView4FFGelu(dim), nn.Identity(), nn.Linear(dim * 4, dim, bias=True)]

    def __call__(self, x: Any) -> Any:
        return self.net[2](self.net[0](x))


class _CogView4Block(nn.Module):
    def __init__(self, cfg: CogView4Config):
        super().__init__()
        dim = cfg.num_attention_heads * cfg.attention_head_dim
        self.norm1 = _CogView4AdaLayerNormZero(cfg.time_embed_dim, dim)
        self.attn1 = _CogView4JointAttention(dim, cfg.num_attention_heads, cfg.attention_head_dim)
        self.norm2 = nn.LayerNorm(dim, eps=1e-5, affine=False)
        self.norm2_context = nn.LayerNorm(dim, eps=1e-5, affine=False)
        self.ff = _CogView4FF(dim)

    def __call__(
        self,
        ctx: RuntimeContext,
        hidden_states: Any,
        encoder_hidden_states: Any,
        temb: Any,
        image_rotary_emb: tuple[Any, Any] | None,
        attention_mask: Any | None,
    ) -> tuple[Any, Any]:
        (
            norm_h, gate_msa, shift_mlp, scale_mlp, gate_mlp,
            norm_c, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp,
        ) = self.norm1(hidden_states, encoder_hidden_states, temb)
        attn_h, attn_c = self.attn1(
            ctx, norm_h, norm_c, image_rotary_emb=image_rotary_emb, attention_mask=attention_mask
        )
        hidden_states = hidden_states + attn_h * gate_msa[:, None, :]
        encoder_hidden_states = encoder_hidden_states + attn_c * c_gate_msa[:, None, :]
        norm_h = self.norm2(hidden_states) * (1 + scale_mlp[:, None, :]) + shift_mlp[:, None, :]
        norm_c = self.norm2_context(encoder_hidden_states) * (1 + c_scale_mlp[:, None, :]) + c_shift_mlp[:, None, :]
        hidden_states = hidden_states + self.ff(norm_h) * gate_mlp[:, None, :]
        encoder_hidden_states = encoder_hidden_states + self.ff(norm_c) * c_gate_mlp[:, None, :]
        return hidden_states, encoder_hidden_states


class _CogView4FinalNorm(nn.Module):
    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.linear = nn.Linear(cond_dim, dim * 2, bias=True)
        self.norm = nn.LayerNorm(dim, eps=1e-5, affine=False)

    def __call__(self, x: Any, cond: Any) -> Any:
        emb = self.linear(cond.astype(x.dtype))
        dim = int(x.shape[-1])
        scale = emb[:, :dim]
        shift = emb[:, dim:2 * dim]
        return self.norm(x) * (1 + scale[:, None, :]) + shift[:, None, :]


class _CogView4DiTCore(nn.Module):
    def __init__(self, cfg: CogView4Config, ctx: RuntimeContext):
        super().__init__()
        inner = cfg.num_attention_heads * cfg.attention_head_dim
        self.cfg = cfg
        self.ctx = ctx
        self.rope = _CogView4RotaryPosEmbed(
            cfg.attention_head_dim, cfg.patch_size, cfg.rope_axes_dim, theta=10000.0,
        )
        self.patch_embed = _CogView4PatchEmbed(
            cfg.in_channels, inner, cfg.patch_size, cfg.text_embed_dim,
        )
        self.time_condition_embed = _CogView4TimeConditionEmbed(cfg)
        self.transformer_blocks = [_CogView4Block(cfg) for _ in range(cfg.num_layers)]
        self.norm_out = _CogView4FinalNorm(inner, cfg.time_embed_dim)
        self.proj_out = nn.Linear(inner, cfg.patch_size * cfg.patch_size * cfg.out_channels, bias=True)

    def __call__(
        self,
        hidden_states: Any,
        encoder_hidden_states: Any,
        timestep: Any,
        original_size: Any,
        target_size: Any,
        crop_coords: Any,
        attention_mask: Any | None = None,
    ) -> Any:
        ctx = self.ctx
        cfg = self.cfg
        p = cfg.patch_size
        b, c, h, w = (int(hidden_states.shape[0]), int(hidden_states.shape[1]),
                      int(hidden_states.shape[2]), int(hidden_states.shape[3]))
        ph, pw = h // p, w // p
        cos, sin = self.rope.forward(ctx, hidden_states)
        image_rotary_emb = (cos, sin)
        hidden_states, encoder_hidden_states = self.patch_embed(hidden_states, encoder_hidden_states)
        temb = self.time_condition_embed(
            ctx, timestep, original_size, target_size, crop_coords, hidden_states.dtype,
        )
        temb = nn.silu(temb)
        for block in self.transformer_blocks:
            hidden_states, encoder_hidden_states = block(
                ctx, hidden_states, encoder_hidden_states, temb, image_rotary_emb, attention_mask,
            )
        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states.reshape(b, ph, pw, -1, p, p)
        return hidden_states.transpose(0, 3, 1, 4, 2, 5).reshape(b, cfg.out_channels, h, w)


class CogView4DiTMLX(TransformerBase):
    """CogView4-6B DiT — joint text/image attention with SDXL micro-conditioning."""

    def __init__(self, config: CogView4Config | Any, ctx: RuntimeContext):
        super().__init__()
        self.ctx = ctx
        if isinstance(config, CogView4Config):
            cfg = config
        else:
            cfg = CogView4Config()
        self.config = cfg
        self._core = _CogView4DiTCore(cfg, ctx)
        self._build_param_map()

    def sanitize(self, weights: dict[str, Any]) -> dict[str, Any]:
        from backend.engine.families.cogview4.weights import remap_cogview4_weights

        return remap_cogview4_weights(weights)

    def _build_param_map(self) -> None:
        from backend.engine.families.ernie_image.transformer_mlx import _flatten_mlx_module_params

        self._param_map = {}
        core = self._core
        for attr in ("patch_embed", "time_condition_embed", "norm_out", "proj_out"):
            _flatten_mlx_module_params(getattr(core, attr), attr, self._param_map)
        for i, block in enumerate(core.transformer_blocks):
            bp = f"transformer_blocks.{i}"
            _flatten_mlx_module_params(block.norm1, f"{bp}.norm1", self._param_map)
            _flatten_mlx_module_params(block.attn1.to_q, f"{bp}.attn1.to_q", self._param_map)
            _flatten_mlx_module_params(block.attn1.to_k, f"{bp}.attn1.to_k", self._param_map)
            _flatten_mlx_module_params(block.attn1.to_v, f"{bp}.attn1.to_v", self._param_map)
            _flatten_mlx_module_params(block.attn1.norm_q, f"{bp}.attn1.norm_q", self._param_map)
            _flatten_mlx_module_params(block.attn1.norm_k, f"{bp}.attn1.norm_k", self._param_map)
            _flatten_mlx_module_params(block.attn1.to_out[0], f"{bp}.attn1.to_out.0", self._param_map)
            _flatten_mlx_module_params(block.norm2, f"{bp}.norm2", self._param_map)
            _flatten_mlx_module_params(block.norm2_context, f"{bp}.norm2_context", self._param_map)
            _flatten_mlx_module_params(block.ff.net[0], f"{bp}.ff.net.0", self._param_map)
            _flatten_mlx_module_params(block.ff.net[2], f"{bp}.ff.net.2", self._param_map)

    def load_weights(
        self,
        weights,
        strict=False,
        ctx=None,
        *,
        bundle_affine_bits=None,
        inference_mode=None,
    ):
        loaded, skipped = super().load_weights(
            weights,
            strict=strict,
            ctx=ctx,
            bundle_affine_bits=bundle_affine_bits,
            inference_mode=inference_mode,
        )
        self._build_param_map()
        return loaded, skipped

    def parameters(self):
        return list(self._param_map.items())

    def prepare_conditioning(self, request: Any, bundle_root: str | None = None) -> dict[str, Any]:
        del bundle_root
        from backend.core.contracts import parse_size

        w, h = parse_size(getattr(request, "size", "1024x1024"))
        return {
            "original_size": (int(w), int(h)),
            "target_size": (int(w), int(h)),
            "crop_coords": (0, 0),
        }

    def _resolve_timestep_batch(self, b: int, timestep: Any, sigmas: Any | None, cond: dict[str, Any]) -> Any:
        ctx = self.ctx
        t_embed = cond.get("timestep_embed_value")
        if t_embed is not None:
            t_val = float(t_embed)
        elif sigmas is not None:
            t_idx = int(timestep)
            n = int(sigmas.shape[0]) if hasattr(sigmas, "shape") else len(sigmas)
            sigma_t = sigmas[t_idx] if t_idx < n else sigmas[-1] if n > 0 else 1.0
            t_val = float(ctx.reshape(ctx.array(sigma_t), (-1,))[0]) * 1000.0
        else:
            tv = timestep
            t_val = float(ctx.reshape(tv, (-1,))[0]) if hasattr(tv, "shape") else float(tv)
            if t_val <= 1.0 + 1e-5:
                t_val *= 1000.0
        t_arr = ctx.full((b,), t_val, dtype=ctx.float32())
        return t_arr

    def _size_tensor(self, b: int, value: Any, dtype: Any) -> Any:
        ctx = self.ctx
        if isinstance(value, (tuple, list)) and len(value) == 2:
            row = ctx.array([[float(value[0]), float(value[1])]], dtype=ctx.float32())
        elif hasattr(value, "shape"):
            row = value.astype(ctx.float32())
        else:
            raise RuntimeError(f"CogView4: invalid size conditioning {value!r}")
        if int(row.shape[0]) == 1 and b > 1:
            row = ctx.broadcast_to(row, (b, 2))
        return row.astype(dtype)

    def forward(
        self,
        latents: Any,
        timestep: Any,
        txt_embeds: Any = None,
        sigmas: Any | None = None,
        **cond: Any,
    ) -> Any:
        if txt_embeds is None:
            raise RuntimeError("CogView4 requires txt_embeds")
        ctx = self.ctx
        b = int(latents.shape[0])
        if int(txt_embeds.shape[0]) == 1 and b > 1:
            txt_embeds = ctx.broadcast_to(txt_embeds, (b, *txt_embeds.shape[1:]))
        t_batch = self._resolve_timestep_batch(b, timestep, sigmas, cond)
        original_size = self._size_tensor(b, cond.get("original_size", cond.get("target_size")), latents.dtype)
        target_size = self._size_tensor(b, cond.get("target_size", cond.get("original_size")), latents.dtype)
        crop_coords = self._size_tensor(b, cond.get("crop_coords", (0, 0)), latents.dtype)
        out = self._core(
            latents.astype(mx.bfloat16),
            txt_embeds.astype(mx.bfloat16),
            t_batch,
            original_size,
            target_size,
            crop_coords,
            attention_mask=cond.get("attention_mask"),
        )
        return out.astype(latents.dtype)
