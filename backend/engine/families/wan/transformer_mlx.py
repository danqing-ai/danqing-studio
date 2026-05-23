"""Wan DiT — MLX implementation (ported from Wan2.2-mlx ``WanModel``)."""
from __future__ import annotations

import math
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from backend.engine.common._base import TransformerBase
from backend.engine.common.cfg_batch import (
    TEXT_KEYS_MINIMAL,
    predict_noise_cfg_batched,
)
from backend.engine.config.model_configs import WanConfig
from backend.engine.runtime._base import RuntimeContext

from .attention_mlx import _seq_lens_from_grid_sizes, wan_attention


def _sinusoidal_embedding_1d(ctx: RuntimeContext, dim: int, position: Any) -> Any:
    if dim % 2 != 0:
        raise ValueError("dim must be even")
    half = dim // 2
    position = position.astype(ctx.float32())
    freqs = ctx.power(
        ctx.array(10000.0, dtype=ctx.float32()),
        -ctx.arange(half, dtype=ctx.float32()) / half,
    )
    sinusoid = ctx.outer(position, freqs)
    return ctx.concat([ctx.cos(sinusoid), ctx.sin(sinusoid)], axis=-1)


def _rope_params(max_seq_len: int, dim: int, theta: float = 10000.0) -> mx.array:
    freqs = mx.outer(
        mx.arange(max_seq_len, dtype=mx.float32),
        1.0 / mx.power(theta, mx.arange(0, dim, 2, dtype=mx.float32) / dim),
    )
    return mx.cos(freqs) + 1j * mx.sin(freqs)


def _repeat_axis(a: mx.array, repeats: int, axis: int) -> mx.array:
    return mx.repeat(a, repeats, axis=axis)


def _split_rope_freqs(freqs: mx.array, c: int) -> tuple[mx.array, mx.array, mx.array]:
    """Split RoPE freqs into temporal / height / width parts (MLX split uses indices)."""
    s0 = c - 2 * (c // 3)
    s1 = c // 3
    return mx.split(freqs, [s0, s0 + s1], axis=1)


def _rope_freqs_for_grid(
    f: int, h: int, w: int, freq_parts: tuple[mx.array, mx.array, mx.array], c: int,
) -> mx.array:
    fp0, fp1, fp2 = freq_parts
    seq_len = f * h * w
    return mx.concatenate([
        _repeat_axis(_repeat_axis(fp0[:f].reshape(f, 1, 1, -1), h, axis=1), w, axis=2),
        _repeat_axis(_repeat_axis(fp1[:h].reshape(1, h, 1, -1), f, axis=0), w, axis=2),
        _repeat_axis(_repeat_axis(fp2[:w].reshape(1, 1, w, -1), f, axis=0), h, axis=1),
    ], axis=-1).reshape(seq_len, 1, c)


def _rope_apply_one(
    x_row: mx.array, seq_len: int, freqs_i: mx.array, n: int, c: int, pad_len: int,
) -> mx.array:
    x_i = mx.view(x_row[:seq_len], mx.complex64).reshape(seq_len, n, c)
    x_i = mx.view(x_i * freqs_i, mx.float32).reshape(seq_len, n, -1)
    if seq_len < pad_len:
        x_i = mx.concatenate([x_i, x_row[seq_len:]], axis=0)
    return x_i


def _rope_apply(x: mx.array, grid_sizes: mx.array, freqs: mx.array) -> mx.array:
    n, c = x.shape[2], x.shape[3] // 2
    freq_parts = _split_rope_freqs(freqs, c)
    grids = grid_sizes.tolist()
    pad_len = int(x.shape[1])
    if len(grids) == 1 or all(g == grids[0] for g in grids[1:]):
        f, h, w = (int(v) for v in grids[0])
        seq_len = f * h * w
        freqs_i = _rope_freqs_for_grid(f, h, w, freq_parts, c)
        return mx.stack([
            _rope_apply_one(x[i], seq_len, freqs_i, n, c, pad_len) for i in range(int(x.shape[0]))
        ]).astype(mx.float32)
    output = []
    for i, (f, h, w) in enumerate(grids):
        seq_len = int(f * h * w)
        freqs_i = _rope_freqs_for_grid(int(f), int(h), int(w), freq_parts, c)
        output.append(_rope_apply_one(x[i], seq_len, freqs_i, n, c, pad_len))
    return mx.stack(output).astype(mx.float32)


class WanRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        norm = x.astype(mx.float32)
        norm = norm * mx.rsqrt(norm.square().mean(axis=-1, keepdims=True) + self.eps)
        return norm.astype(x.dtype) * self.weight


class WanLayerNorm(nn.LayerNorm):
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = False):
        super().__init__(dims=dim, eps=eps, affine=elementwise_affine)

    def __call__(self, x: mx.array) -> mx.array:
        return super().__call__(x.astype(mx.float32)).astype(x.dtype)


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
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def __call__(self, x: mx.array, grid_sizes: mx.array, freqs: mx.array) -> mx.array:
        b, s, n, d = x.shape[0], x.shape[1], self.num_heads, self.head_dim
        seq_lens = _seq_lens_from_grid_sizes(grid_sizes)
        q = self.norm_q(self.q(x)).reshape(b, s, n, d)
        k = self.norm_k(self.k(x)).reshape(b, s, n, d)
        v = self.v(x).reshape(b, s, n, d)
        out = wan_attention(
            self.ctx,
            _rope_apply(q, grid_sizes, freqs),
            _rope_apply(k, grid_sizes, freqs),
            v,
            k_lens=seq_lens,
        )
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
        return self.layer_2(nn.gelu(self.layer_0(x)))


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
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(ctx, dim, num_heads, qk_norm, eps)
        self.norm3 = WanLayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WanCrossAttention(ctx, dim, num_heads, qk_norm, eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = WanFFN(dim, ffn_dim)
        self.modulation = mx.zeros((1, 6, dim))

    def __call__(
        self,
        x: mx.array,
        e: mx.array,
        grid_sizes: mx.array,
        freqs: mx.array,
        context: mx.array,
        *,
        cross_kv: tuple[mx.array, mx.array] | None = None,
    ) -> mx.array:
        e = mx.expand_dims(self.modulation, 0) + e
        e = mx.split(e, 6, axis=2)
        y = self.self_attn(
            self.norm1(x).astype(mx.float32) * (1 + mx.squeeze(e[1], axis=2)) + mx.squeeze(e[0], axis=2),
            grid_sizes,
            freqs,
        )
        x = x + y * mx.squeeze(e[2], axis=2)
        x = x + self.cross_attn(self.norm3(x), context, cross_kv=cross_kv)
        y = self.ffn(
            self.norm2(x).astype(mx.float32) * (1 + mx.squeeze(e[4], axis=2)) + mx.squeeze(e[3], axis=2)
        )
        return x + y * mx.squeeze(e[5], axis=2)


class WanModelMLX(TransformerBase):
    """Wan video DiT — ``VideoPipeline`` contract: latents ``[B,C,T,H,W]``."""

    def __init__(self, config: WanConfig, ctx: RuntimeContext, num_frames: int = 81):
        self.config = config
        self.ctx = ctx
        self._num_frames = num_frames
        pt, ph, pw = config.patch_size

        self.patch_embedding = nn.Conv3d(
            config.dim_in,
            config.dim,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )
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
        self.head = nn.Linear(config.dim, ph * pw * pt * config.dim_out)
        self.head_modulation = mx.zeros((1, 2, config.dim))
        self.patch_size = config.patch_size
        self.text_len = config.text_len
        self.out_dim = config.dim_out

        d = config.dim // config.num_heads
        self._freqs = mx.concatenate([
            _rope_params(1024, d - 4 * (d // 6)),
            _rope_params(1024, 2 * (d // 6)),
            _rope_params(1024, 2 * (d // 6)),
        ], axis=1)

        self._i2v_cond: Any | None = None
        self._i2v_mask: Any | None = None
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

    def after_load_weights(self, bundle_root=None) -> None:
        super().after_load_weights(bundle_root)
        self._compiled_forward = None
        self.invalidate_text_cache()
        if getattr(self.ctx, "backend", None) != "mlx":
            return
        if not getattr(self.config, "use_mlx_compile", True):
            return
        try:
            self._compiled_forward = self.ctx.compile(self._forward_compute)
        except Exception:
            self._compiled_forward = None

    def _text_cache_key_for(self, txt_embeds: Any) -> tuple[int, ...]:
        sh = tuple(int(x) for x in txt_embeds.shape)
        return sh + (id(txt_embeds),)

    def _get_context_and_cross_kv(
        self, txt_embeds: Any,
    ) -> tuple[mx.array, list[tuple[mx.array, mx.array]]]:
        key = self._text_cache_key_for(txt_embeds)
        if self._text_cache_key == key and self._cached_context is not None and self._cached_cross_kv is not None:
            return self._cached_context, self._cached_cross_kv
        context = self._apply_text_embed(txt_embeds)
        cross_kv = [blk.cross_attn.cross_kv(context) for blk in self.blocks]
        self._text_cache_key = key
        self._cached_context = context
        self._cached_cross_kv = cross_kv
        return context, cross_kv

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
        self._param_map["head.head.weight"] = self.head.weight
        self._param_map["head.head.bias"] = self.head.bias
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

    def parameters(self):
        return list(self._param_map.items())

    def set_i2v_state(self, cond: Any | None, mask: Any | None) -> None:
        self._i2v_cond = cond
        self._i2v_mask = mask

    def reblend_i2v_latents(self, latents: Any) -> Any:
        if self._i2v_cond is None or self._i2v_mask is None:
            return latents
        from .conditioning import prepare_ti2v_i2v_latents
        return prepare_ti2v_i2v_latents(self.ctx, latents, self._i2v_cond, self._i2v_mask)

    def _ncthw_to_conv_input(self, sample: Any) -> Any:
        """``[C,T,H,W]`` → ``[1,T,H,W,C]`` for MLX Conv3d."""
        return self.ctx.expand_dims(self.ctx.permute(sample, (1, 2, 3, 0)), 0)

    def _apply_text_embed(self, txt_embeds: Any) -> Any:
        ctx = self.ctx
        cfg = self.config
        b = int(txt_embeds.shape[0])
        padded = []
        for i in range(b):
            u = txt_embeds[i]
            seq = int(u.shape[0])
            if seq >= cfg.text_len:
                padded.append(u[: cfg.text_len])
            else:
                pad = ctx.zeros((cfg.text_len - seq, u.shape[-1]), dtype=u.dtype)
                padded.append(ctx.concat([u, pad], axis=0))
        batch = ctx.stack(padded)
        return self.text_embedding[1](nn.gelu(self.text_embedding[0](batch)))

    def _time_paths(self, t: Any, seq_len: int) -> tuple[Any, Any]:
        ctx = self.ctx
        cfg = self.config
        ndim = getattr(t, "ndim", 0)
        if ndim == 0:
            t = ctx.broadcast_to(ctx.reshape(t, (1, 1)), (1, seq_len))
        elif ndim == 1:
            t = ctx.broadcast_to(ctx.reshape(t, (-1, 1)), (int(t.shape[0]), seq_len))
        elif ndim == 2 and int(t.shape[1]) == 1:
            t = ctx.broadcast_to(t, (int(t.shape[0]), seq_len))
        bt = int(t.shape[0])
        flat = ctx.reshape(t, (-1,))
        emb = _sinusoidal_embedding_1d(ctx, cfg.freq_dim, flat)
        emb = ctx.reshape(emb, (bt, seq_len, cfg.freq_dim)).astype(ctx.float32())
        e = self.time_embedding[1](nn.silu(self.time_embedding[0](emb)))
        e0 = self.time_projection(nn.silu(e))
        e0 = ctx.reshape(e0, (bt, seq_len, 6, cfg.dim))
        return e, e0

    def _unpatchify(self, x: Any, grid_sizes: Any) -> Any:
        ctx = self.ctx
        c = self.out_dim
        pt, ph, pw = self.patch_size
        outs = []
        for bi in range(int(x.shape[0])):
            u = x[bi]
            f, h, w = [int(v) for v in grid_sizes[bi].tolist()]
            tok = u[: f * h * w].reshape(f, h, w, pt, ph, pw, c)
            tok = mx.einsum("fhwpqrc->cfphqwr", tok)
            outs.append(tok.reshape(c, f * pt, h * ph, w * pw))
        return ctx.stack(outs, axis=0)

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
        self._get_context_and_cross_kv(txt_embeds)
        if self._compiled_forward is not None:
            return self._compiled_forward(
                latents, timestep, timestep_per_token, seq_len,
            )
        return self._forward_compute(
            latents, timestep, timestep_per_token, seq_len,
        )

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
        patches = []
        grids = []
        for i in range(b):
            inp = self._ncthw_to_conv_input(latents[i])
            pe = self.patch_embedding(inp)
            pe = ctx.squeeze(pe, 0)
            grid = ctx.array([pe.shape[0], pe.shape[1], pe.shape[2]], dtype=ctx.int64())
            flat = ctx.reshape(pe, (-1, pe.shape[-1]))
            patches.append(flat)
            grids.append(grid)
        grid_sizes = ctx.stack(grids)
        if seq_len is None:
            seq_len = max(int(p.shape[0]) for p in patches)
        x = ctx.stack([
            ctx.concat([p, ctx.zeros((seq_len - p.shape[0], p.shape[1]), dtype=p.dtype)], axis=0)
            for p in patches
        ])

        t_in = timestep_per_token if timestep_per_token is not None else timestep
        if t_in is not None:
            if getattr(t_in, "ndim", 0) == 0:
                t_in = mx.reshape(t_in, (1, 1))
            if int(getattr(t_in, "shape", (1,))[0]) == 1 and b > 1:
                t_in = mx.repeat(t_in, b, axis=0)
        e, e0 = self._time_paths(t_in, seq_len)
        freqs = self._freqs

        for blk, cross_kv in zip(self.blocks, cross_kv_list):
            x = blk(x, e0, grid_sizes, freqs, context, cross_kv=cross_kv)

        e_head = mx.expand_dims(self.head_modulation, 0) + mx.expand_dims(e, 2)
        e_chunks = mx.split(e_head, 2, axis=2)
        x = self.head(x * (1 + mx.squeeze(e_chunks[1], axis=2)) + mx.squeeze(e_chunks[0], axis=2))
        return self._unpatchify(x, grid_sizes)

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
