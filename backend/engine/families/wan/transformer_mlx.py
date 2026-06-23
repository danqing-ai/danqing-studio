"""Wan DiT — MLX implementation (ported from Wan2.2-mlx ``WanModel``)."""
from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn

from backend.engine.common.model.base import TransformerBase
from backend.engine.common.ops.cfg_batch import (
    TEXT_KEYS_MINIMAL,
    predict_noise_cfg_batched,
)
from backend.engine.common.ops.attention import (
    build_key_padding_mask_from_lengths,
    wan_attention,
)
from backend.engine.common.ops.embeddings import (
    factorized_rope_apply,
    factorized_rope_concat_params,
    factorized_rope_precompute_cos_sin,
    pad_ragged_2d_sequences,
    sinusoidal_embedding_1d,
)
from backend.engine.common.ops.norm import (
    apply_layer_norm_fp32,
    apply_scale_shift,
    unpack_modulation_2table,
    unpack_modulation_6table,
)
from backend.engine.config.model_configs import WanConfig
from backend.engine.runtime._base import RuntimeContext

from backend.engine.common.codecs.text_encoders.qwen3_mlx import MlxRMSNorm

class WanLayerNorm(nn.LayerNorm):
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = False):
        super().__init__(dims=dim, eps=eps, affine=elementwise_affine)

    def __call__(self, x: mx.array) -> mx.array:
        return apply_layer_norm_fp32(lambda y: super(WanLayerNorm, self).__call__(y), x)


class WanSelfAttention(nn.Module):
    def __init__(self, ctx: RuntimeContext, dim: int, num_heads: int, qk_norm: bool, eps: float):
        super().__init__()
        self.ctx = ctx
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = MlxRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = MlxRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def __call__(
        self,
        x: mx.array,
        grid_sizes: list[tuple[int, int, int]],
        freqs: mx.array,
        *,
        rope_cos_sin: tuple[mx.array, mx.array] | None = None,
        attn_mask: mx.array | None = None,
    ) -> mx.array:
        b, s, n, d = x.shape[0], x.shape[1], self.num_heads, self.head_dim
        w_dtype = self.q.weight.dtype
        x_w = x.astype(w_dtype)
        fp32 = self.ctx.float32()
        q = self.norm_q(self.q(x_w)).reshape(b, s, n, d)
        k = self.norm_k(self.k(x_w)).reshape(b, s, n, d)
        v = self.v(x_w).reshape(b, s, n, d)
        q = factorized_rope_apply(
            mx, q.astype(fp32), grid_sizes, freqs, precomputed_cos_sin=rope_cos_sin
        ).astype(w_dtype)
        k = factorized_rope_apply(
            mx, k.astype(fp32), grid_sizes, freqs, precomputed_cos_sin=rope_cos_sin
        ).astype(w_dtype)
        if attn_mask is not None:
            out = wan_attention(self.ctx, q, k, v, mask=attn_mask)
        else:
            out = wan_attention(self.ctx, q, k, v)
        return self.o(out.reshape(b, s, -1))


class WanCrossAttention(WanSelfAttention):
    def cross_kv(self, context: mx.array) -> tuple[mx.array, mx.array]:
        """Precompute K/V for fixed text context (reused across denoise steps)."""
        b, n, d = context.shape[0], self.num_heads, self.head_dim
        k = self.norm_k(self.k(context)).reshape(b, -1, n, d)
        v = self.v(context).reshape(b, -1, n, d)
        return k, v

    def __call__(
        self,
        x: mx.array,
        context: mx.array | None = None,
        *,
        cross_kv: tuple[mx.array, mx.array] | None = None,
    ) -> mx.array:
        b, n, d = x.shape[0], self.num_heads, self.head_dim
        q = self.norm_q(self.q(x)).reshape(b, -1, n, d)
        if cross_kv is not None:
            k, v = cross_kv
        else:
            if context is None:
                raise RuntimeError("WanCrossAttention requires context or cross_kv")
            k, v = self.cross_kv(context)
        out = wan_attention(self.ctx, q, k, v)
        return self.o(out.reshape(b, -1, n * d))


class WanFFN(nn.Module):
    def __init__(self, dim: int, ffn_dim: int):
        super().__init__()
        self.layer_0 = nn.Linear(dim, ffn_dim)
        self.layer_2 = nn.Linear(ffn_dim, dim)

    def __call__(self, x: mx.array) -> mx.array:
        return self.layer_2(nn.GELU(approx="tanh")(self.layer_0(x)))


class WanAttentionBlock(nn.Module):
    def __init__(
        self,
        ctx: RuntimeContext,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        qk_norm: bool,
        cross_attn_norm: bool,
        eps: float,
    ):
        super().__init__()
        self.ctx = ctx
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(ctx, dim, num_heads, qk_norm, eps)
        self.norm3 = WanLayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WanCrossAttention(ctx, dim, num_heads, qk_norm, eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = WanFFN(dim, ffn_dim)
        self.modulation = ctx.zeros((1, 6, dim), dtype=ctx.float32())

    def __call__(
        self,
        x: mx.array,
        e: mx.array,
        grid_sizes: list[tuple[int, int, int]],
        freqs: mx.array,
        context: mx.array,
        *,
        cross_kv: tuple[mx.array, mx.array] | None = None,
        rope_cos_sin: tuple[mx.array, mx.array] | None = None,
        attn_mask: mx.array | None = None,
    ) -> mx.array:
        ctx = self.ctx
        fp32 = ctx.float32()
        mod = ctx.expand_dims(self.modulation.astype(fp32), 0) + e.astype(fp32)
        e0, e1, e2, e3, e4, e5 = unpack_modulation_6table(mod)
        y = self.self_attn(
            apply_scale_shift(self.norm1(x).astype(fp32), e1, e0, add_one=True),
            grid_sizes,
            freqs,
            rope_cos_sin=rope_cos_sin,
            attn_mask=attn_mask,
        )
        x = x + y * e2
        x = x + self.cross_attn(self.norm3(x), context, cross_kv=cross_kv)
        y = self.ffn(
            apply_scale_shift(self.norm2(x).astype(fp32), e4, e3, add_one=True)
        )
        return x + y * e5


class WanModelMLX(TransformerBase):
    """Wan video DiT — ``VideoPipeline`` contract: latents ``[B,C,T,H,W]``."""

    def __init__(self, config: WanConfig, ctx: RuntimeContext, num_frames: int = 81):
        self.config = config
        self.ctx = ctx
        self._num_frames = num_frames
        pt, ph, pw = config.patch_size
        patch_dim = int(config.dim_in) * int(pt) * int(ph) * int(pw)

        self.patch_embedding = nn.Linear(patch_dim, config.dim)
        self._patch_size = config.patch_size
        self.text_embedding = ctx.ModuleList([
            nn.Linear(config.text_dim, config.dim),
            nn.Linear(config.dim, config.dim),
        ])
        self.time_embedding = ctx.ModuleList([
            nn.Linear(config.freq_dim, config.dim),
            nn.Linear(config.dim, config.dim),
        ])
        self.time_projection = nn.Linear(config.dim, config.dim * 6)
        self.blocks = ctx.ModuleList([
            WanAttentionBlock(
                ctx, config.dim, config.ffn_dim, config.num_heads,
                config.qk_norm, config.cross_attn_norm, config.eps,
            )
            for _ in range(config.depth)
        ])
        self.head_norm = WanLayerNorm(config.dim, config.eps)
        self.head = nn.Linear(config.dim, ph * pw * pt * config.dim_out)
        self.head_modulation = ctx.zeros((1, 2, config.dim), dtype=ctx.float32())
        self.patch_size = self._patch_size
        self.text_len = config.text_len
        self.out_dim = config.dim_out

        d = config.dim // config.num_heads
        self._freqs = factorized_rope_concat_params(
            mx,
            1024,
            [d - 4 * (d // 6), 2 * (d // 6), 2 * (d // 6)],
        )

        self._rope_cos_sin: tuple[mx.array, mx.array] | None = None
        self._rope_grid_key: tuple[int, int, int] | None = None
        self._i2v_cond: Any | None = None
        self._i2v_mask: Any | None = None
        self._i2v_side: Any | None = None
        self._compiled_forward = None
        self._text_cache_key: tuple[int, ...] | None = None
        self._cached_context: mx.array | None = None
        self._cached_cross_kv: list[tuple[mx.array, mx.array]] | None = None
        self._build_param_map()

    def invalidate_text_cache(self) -> None:
        """Drop cross-attn K/V cache (new prompt or after weight load)."""
        self._text_cache_key = None
        self._cached_context = None
        self._cached_cross_kv = None
        self._rope_cos_sin = None
        self._rope_grid_key = None

    def after_load_weights(self, bundle_root=None) -> None:
        super().after_load_weights(bundle_root)
        self._compiled_forward = None
        self.invalidate_text_cache()
        if getattr(self.ctx, "backend", None) != "mlx":
            return
        allow_compile = bool(getattr(self.config, "use_mlx_compile", False))
        if getattr(self.config, "step_distill", False):
            allow_compile = bool(getattr(self.config, "use_mlx_compile_step_distill", True))
        if not allow_compile:
            return
        try:
            self._compiled_forward = self.ctx.compile(self._forward_compute)
        except Exception:
            self._compiled_forward = None

    def sanitize(self, weights: dict) -> dict:
        """Transform checkpoint keys to match ``WanModelMLX._param_map``.

        Handles both original Wan keys and diffusers-format checkpoints,
        including Conv3d → Linear patch embedding reshape.
        """
        from backend.engine.families.wan.weights import remap_wan_weights

        return remap_wan_weights(weights)

    def _build_param_map(self) -> None:
        self._param_map = {}
        self._param_map["patch_embedding.weight"] = self.patch_embedding.weight
        self._param_map["patch_embedding.bias"] = self.patch_embedding.bias
        self._param_map["text_embedding.0.weight"] = self.text_embedding[0].weight
        self._param_map["text_embedding.0.bias"] = self.text_embedding[0].bias
        self._param_map["text_embedding.2.weight"] = self.text_embedding[1].weight
        self._param_map["text_embedding.2.bias"] = self.text_embedding[1].bias
        self._param_map["time_embedding.0.weight"] = self.time_embedding[0].weight
        self._param_map["time_embedding.0.bias"] = self.time_embedding[0].bias
        self._param_map["time_embedding.2.weight"] = self.time_embedding[1].weight
        self._param_map["time_embedding.2.bias"] = self.time_embedding[1].bias
        self._param_map["time_projection.1.weight"] = self.time_projection.weight
        self._param_map["time_projection.1.bias"] = self.time_projection.bias
        self._param_map["head.1.weight"] = self.head.weight
        self._param_map["head.1.bias"] = self.head.bias
        self._param_map["head.modulation"] = self.head_modulation
        for i, blk in enumerate(self.blocks):
            prefix = f"blocks.{i}"
            self._param_map[f"{prefix}.modulation"] = blk.modulation
            for part in ("self_attn", "cross_attn"):
                attn = getattr(blk, part)
                for w in ("q", "k", "v", "o"):
                    lin = getattr(attn, w)
                    self._param_map[f"{prefix}.{part}.{w}.weight"] = lin.weight
                    self._param_map[f"{prefix}.{part}.{w}.bias"] = lin.bias
                if hasattr(attn, "norm_q") and hasattr(attn.norm_q, "weight"):
                    self._param_map[f"{prefix}.{part}.norm_q.weight"] = attn.norm_q.weight
                if hasattr(attn, "norm_k") and hasattr(attn.norm_k, "weight"):
                    self._param_map[f"{prefix}.{part}.norm_k.weight"] = attn.norm_k.weight
            if hasattr(blk.norm3, "weight"):
                self._param_map[f"{prefix}.norm3.weight"] = blk.norm3.weight
                self._param_map[f"{prefix}.norm3.bias"] = blk.norm3.bias
            self._param_map[f"{prefix}.ffn.layer_0.weight"] = blk.ffn.layer_0.weight
            self._param_map[f"{prefix}.ffn.layer_0.bias"] = blk.ffn.layer_0.bias
            self._param_map[f"{prefix}.ffn.layer_2.weight"] = blk.ffn.layer_2.weight
            self._param_map[f"{prefix}.ffn.layer_2.bias"] = blk.ffn.layer_2.bias

    def set_i2v_state(
        self,
        cond: Any | None,
        mask: Any | None,
        *,
        side: Any | None = None,
    ) -> None:
        self._i2v_cond = cond
        self._i2v_mask = mask
        self._i2v_side = side

    def reblend_i2v_latents(self, latents: Any) -> Any:
        if self._i2v_side is not None:
            return latents
        if self._i2v_cond is None or self._i2v_mask is None:
            return latents
        from .conditioning import prepare_ti2v_i2v_latents
        return prepare_ti2v_i2v_latents(self.ctx, latents, self._i2v_cond, self._i2v_mask)

    def forward(
        self,
        latents: Any,
        timestep: Any,
        txt_embeds: Any | None = None,
        *,
        timestep_per_token: Any | None = None,
        seq_len: int | None = None,
        **_: Any,
    ) -> Any:
        if txt_embeds is None:
            raise RuntimeError("Wan requires T5 embeddings (`txt_embeds`).")
        key = tuple(int(x) for x in txt_embeds.shape) + (id(txt_embeds),)
        if not (
            self._text_cache_key == key
            and self._cached_context is not None
            and self._cached_cross_kv is not None
        ):
            cfg = self.config
            batch = pad_ragged_2d_sequences(
                self.ctx,
                [txt_embeds[i] for i in range(int(txt_embeds.shape[0]))],
                target_len=cfg.text_len,
                dtype=txt_embeds.dtype,
                pad_value=0.0,
            )
            context = self.text_embedding[1](nn.GELU(approx="tanh")(self.text_embedding[0](batch)))
            self._cached_context = context
            self._cached_cross_kv = [blk.cross_attn.cross_kv(context) for blk in self.blocks]
            self._text_cache_key = key
        if self._compiled_forward is not None:
            return self._compiled_forward(latents, timestep, timestep_per_token, seq_len)
        return self._forward_compute(latents, timestep, timestep_per_token, seq_len)

    def _forward_compute(
        self,
        latents: Any,
        timestep: Any,
        timestep_per_token: Any | None,
        seq_len: int | None,
    ) -> Any:
        ctx = self.ctx
        context = self._cached_context
        cross_kv_list = self._cached_cross_kv
        if context is None or cross_kv_list is None:
            raise RuntimeError("Wan: text context cache missing; call forward() with txt_embeds first.")
        if latents.ndim != 5:
            raise RuntimeError(f"Wan expects latents [B,C,T,H,W], got {latents.shape}")

        b = int(latents.shape[0])
        per_token = timestep_per_token is not None
        patches = []
        grid_sizes_list: list[tuple[int, int, int]] = []
        seq_lens_list: list[int] = []
        pt, ph, pw = self._patch_size
        for i in range(b):
            sample = latents[i]
            if self._i2v_side is not None:
                side = self._i2v_side[i] if int(getattr(self._i2v_side, "ndim", 0)) == 5 else self._i2v_side
                sample = ctx.concat([sample, side], axis=0)
            c, f, h, w = (int(sample.shape[j]) for j in range(4))
            if f % pt != 0 or h % ph != 0 or w % pw != 0:
                raise RuntimeError(
                    f"Wan latent shape [C,T,H,W]=[{c},{f},{h},{w}] is not divisible by "
                    f"patch_size={self._patch_size} (need T%{pt}==0, H%{ph}==0, W%{pw}==0). "
                    f"Snap pixel size to multiples of vae_scale×patch "
                    f"(e.g. 480×704 not 480×720 for vae_scale=16)."
                )
            f_out, h_out, w_out = f // pt, h // ph, w // pw
            patch = sample.reshape(c, f_out, pt, h_out, ph, w_out, pw)
            patch = patch.transpose(1, 3, 5, 0, 2, 4, 6)
            flat = patch.reshape(f_out * h_out * w_out, -1)
            flat = self.patch_embedding(flat).astype(self.patch_embedding.weight.dtype)
            grid = (f_out, h_out, w_out)
            patches.append(flat)
            grid_sizes_list.append(grid)
            seq_lens_list.append(int(flat.shape[0]))
        if seq_len is None:
            seq_len = max(seq_lens_list)
        x = pad_ragged_2d_sequences(ctx, patches, target_len=int(seq_len))

        t_in = timestep_per_token if per_token else timestep
        if t_in is None:
            raise RuntimeError("Wan forward requires timestep or timestep_per_token")
        if getattr(t_in, "ndim", 0) == 0:
            t_in = ctx.reshape(t_in, (1,))
        if int(getattr(t_in, "shape", (1,))[0]) == 1 and b > 1:
            t_in = ctx.repeat(t_in, b, axis=0)

        cfg = self.config
        ndim = getattr(t_in, "ndim", 0)
        if per_token:
            if ndim == 0:
                raise RuntimeError("Wan per-token timesteps require a 2D tensor [B, L]")
            if ndim == 1:
                t_in = ctx.reshape(t_in, (1, -1))
            bt = int(t_in.shape[0])
            seq_tok = int(t_in.shape[1])
            flat_t = ctx.reshape(t_in, (-1,))
            emb = sinusoidal_embedding_1d(ctx, cfg.freq_dim, flat_t)
            emb = ctx.reshape(emb, (bt, seq_tok, cfg.freq_dim)).astype(ctx.float32())
            e = self.time_embedding[1](nn.silu(self.time_embedding[0](emb)))
            e0 = self.time_projection(nn.silu(e))
            e0 = ctx.reshape(e0, (bt, seq_tok, 6, cfg.dim))
        else:
            if ndim == 0:
                t_b = ctx.reshape(t_in, (1,))
            elif ndim == 1:
                t_b = t_in
            elif ndim == 2 and int(t_in.shape[1]) == 1:
                t_b = ctx.reshape(t_in, (-1,))
            else:
                raise RuntimeError(
                    f"Wan scalar timestep expected [B] or scalar, got shape {getattr(t_in, 'shape', ())}"
                )
            t_b = t_b.astype(ctx.float32())
            emb = sinusoidal_embedding_1d(ctx, cfg.freq_dim, t_b).astype(ctx.float32())
            e = self.time_embedding[1](nn.silu(self.time_embedding[0](emb)))
            e0 = self.time_projection(nn.silu(e))
            e0 = ctx.reshape(e0, (int(t_b.shape[0]), 1, 6, cfg.dim))
        freqs = self._freqs
        rope_key = grid_sizes_list[0]
        if self._rope_grid_key == rope_key and self._rope_cos_sin is not None:
            rope_cos_sin = self._rope_cos_sin
        else:
            w_dtype = self.patch_embedding.weight.dtype
            rope_cos_sin = factorized_rope_precompute_cos_sin(
                mx, grid_sizes_list, self._freqs, dtype=w_dtype
            )
            self._rope_grid_key = rope_key
            self._rope_cos_sin = rope_cos_sin

        if all(sl >= seq_len for sl in seq_lens_list):
            attn_mask = None
        else:
            lens = ctx.array(seq_lens_list[:b], dtype=ctx.int32())
            attn_mask = build_key_padding_mask_from_lengths(ctx, lens, seq_len, self.patch_embedding.weight.dtype)

        for blk, cross_kv in zip(self.blocks, cross_kv_list):
            x = blk(
                x,
                e0,
                grid_sizes_list,
                freqs,
                context,
                cross_kv=cross_kv,
                rope_cos_sin=rope_cos_sin,
                attn_mask=attn_mask,
            )

        if e.ndim == 2:
            e_h = e[:, None, :]
        else:
            e_h = e
        fp32 = ctx.float32()
        mod = self.head_modulation.astype(fp32)[:, None, :, :] + e_h.astype(fp32)[:, :, None, :]
        e_shift, e_scale = unpack_modulation_2table(mod)
        x = self.head_norm(x).astype(fp32)
        x = self.head(apply_scale_shift(x, e_scale, e_shift, add_one=True))
        c = self.out_dim
        pt, ph, pw = self.patch_size
        outs = []
        for bi in range(int(x.shape[0])):
            u = x[bi]
            f, h, w = grid_sizes_list[bi]
            tok = u[: f * h * w].reshape(f, h, w, pt, ph, pw, c)
            tok = ctx.einsum("fhwpqrc->cfphqwr", tok)
            outs.append(tok.reshape(c, f * pt, h * ph, w * pw))
        return ctx.stack(outs, axis=0)

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
            text_keys=TEXT_KEYS_MINIMAL,
            combine_cfg_noise=self.combine_cfg_noise,
            refine_cfg_noise=self.refine_cfg_noise,
            cfg_renorm=cfg_renorm,
            cfg_renorm_min=cfg_renorm_min,
        )
