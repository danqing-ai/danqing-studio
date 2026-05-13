"""
CogVideoX Transformer3D — weight layout compatible with HuggingFace diffusers
`CogVideoXTransformer3DModel` / `ZhipuAI/CogVideoX-5b` transformer shards.

Implementation references:
- diffusers `models/transformers/cogvideox_transformer_3d.py`
- diffusers `models/attention_processor.py` (`CogVideoXAttnProcessor2_0`)
"""
from __future__ import annotations

import math
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from backend.engine.common._base import TransformerBase
from backend.engine.common.attention import apply_rope_interleaved_real
from backend.engine.config.model_configs import CogVideoXConfig
from backend.engine.runtime._base import RuntimeContext


def _get_1d_sincos_pos_embed_from_grid_np(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be divisible by 2")
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    return np.concatenate([emb_sin, emb_cos], axis=1)


def _get_2d_sincos_pos_embed_from_grid_np(embed_dim: int, grid: np.ndarray) -> np.ndarray:
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be divisible by 2")
    emb_h = _get_1d_sincos_pos_embed_from_grid_np(embed_dim // 2, grid[0])
    emb_w = _get_1d_sincos_pos_embed_from_grid_np(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


def _get_3d_sincos_pos_embed_np(
    embed_dim: int,
    spatial_size: int | tuple[int, int],
    temporal_size: int,
    spatial_interpolation_scale: float = 1.0,
    temporal_interpolation_scale: float = 1.0,
) -> np.ndarray:
    if embed_dim % 4 != 0:
        raise ValueError("`embed_dim` must be divisible by 4")
    if isinstance(spatial_size, int):
        spatial_size = (spatial_size, spatial_size)

    embed_dim_spatial = 3 * embed_dim // 4
    embed_dim_temporal = embed_dim // 4

    grid_h = np.arange(spatial_size[1], dtype=np.float32) / spatial_interpolation_scale
    grid_w = np.arange(spatial_size[0], dtype=np.float32) / spatial_interpolation_scale
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, spatial_size[1], spatial_size[0]])
    pos_embed_spatial = _get_2d_sincos_pos_embed_from_grid_np(embed_dim_spatial, grid)

    grid_t = np.arange(temporal_size, dtype=np.float32) / temporal_interpolation_scale
    pos_embed_temporal = _get_1d_sincos_pos_embed_from_grid_np(embed_dim_temporal, grid_t)

    pos_embed_spatial = pos_embed_spatial[np.newaxis, :, :]
    pos_embed_spatial = np.repeat(pos_embed_spatial, temporal_size, axis=0)

    pos_embed_temporal = pos_embed_temporal[:, np.newaxis, :]
    pos_embed_temporal = np.repeat(pos_embed_temporal, spatial_size[0] * spatial_size[1], axis=1)

    return np.concatenate([pos_embed_temporal, pos_embed_spatial], axis=-1)


def build_joint_pos_embedding(
    embed_dim: int,
    max_text_seq_length: int,
    post_patch_hw: tuple[int, int],
    temporal_slots: int,
    spatial_interpolation_scale: float,
    temporal_interpolation_scale: float,
) -> np.ndarray:
    post_patch_height, post_patch_width = post_patch_hw
    num_patches = post_patch_height * post_patch_width * temporal_slots

    pos_embedding = _get_3d_sincos_pos_embed_np(
        embed_dim,
        (post_patch_width, post_patch_height),
        temporal_slots,
        spatial_interpolation_scale,
        temporal_interpolation_scale,
    )
    pos_embedding = pos_embedding.reshape(
        temporal_slots * post_patch_height * post_patch_width, embed_dim,
    )

    joint = np.zeros((1, max_text_seq_length + num_patches, embed_dim), dtype=np.float32)
    joint[0, max_text_seq_length:, :] = pos_embedding.astype(np.float32)
    return joint


def _get_timestep_embedding_mx(
    ctx: RuntimeContext,
    timesteps: Any,
    embedding_dim: int,
    *,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1.0,
    scale: float = 1.0,
    max_period: int = 10000,
) -> Any:
    timesteps = ctx.reshape(timesteps, (-1,))
    half_dim = embedding_dim // 2
    denom = max(half_dim - downscale_freq_shift, 1e-8)
    exp_arg = -math.log(max_period) * ctx.arange(half_dim, dtype=ctx.float32()) / denom
    emb = ctx.exp(exp_arg)
    emb = timesteps[:, None].astype(ctx.float32()) * emb[None, :]
    emb = scale * emb
    emb = ctx.concat([ctx.sin(emb), ctx.cos(emb)], axis=-1)
    if flip_sin_to_cos:
        emb = ctx.concat([emb[:, half_dim:], emb[:, :half_dim]], axis=-1)
    if embedding_dim % 2 == 1:
        z = ctx.zeros((emb.shape[0], 1), dtype=emb.dtype)
        emb = ctx.concat([emb, z], axis=-1)
    return emb


class TimestepEmbeddingLinear:
    """Maps to diffusers `TimestepEmbedding`: linear_1 → SiLU → linear_2."""

    def __init__(self, ctx: RuntimeContext, in_channels: int, time_embed_dim: int):
        self.linear_1 = ctx.Linear(in_channels, time_embed_dim, bias=True)
        self.act = nn.SiLU()
        self.linear_2 = ctx.Linear(time_embed_dim, time_embed_dim, bias=True)

    def __call__(self, x: Any) -> Any:
        x = self.linear_1(x)
        x = self.act(x)
        return self.linear_2(x)


class _FFGeluApprox:
    """`FeedForward` stage 0 — checkpoint key `ff.net.0.proj`."""

    def __init__(self, ctx: RuntimeContext, dim_in: int, dim_out: int, bias: bool = True):
        self.proj = ctx.Linear(dim_in, dim_out, bias=bias)

    def __call__(self, x: Any) -> Any:
        return nn.gelu_approx(self.proj(x))


class CogVideoXPatchEmbed:
    """Patch embed + text projection + joint sincos (matches diffusers tensor ops)."""

    def __init__(self, cfg: CogVideoXConfig, ctx: RuntimeContext):
        self.ctx = ctx
        self.cfg = cfg
        ps = cfg.patch_size
        self.proj = ctx.Conv2d(
            cfg.in_channels, cfg.inner_dim,
            kernel_size=(ps, ps), stride=(ps, ps), bias=cfg.patch_bias,
        )
        self.text_proj = ctx.Linear(cfg.text_dim, cfg.inner_dim, bias=True)

    def __call__(self, text_embeds: Any, latents_bcthw: Any) -> Any:
        """latents_bcthw: [B, T, C, H, W] (diffusers layout)."""
        ctx = self.ctx
        cfg = self.cfg
        txt = self.text_proj(text_embeds)
        B, T, C, H, W = latents_bcthw.shape
        x = ctx.permute(latents_bcthw, (0, 1, 3, 4, 2))
        x = ctx.reshape(x, (B * T, H, W, C))
        x = self.proj(x)
        BT, Oh, Ow, Od = x.shape
        x = ctx.reshape(x, (B, T, Oh, Ow, Od))
        x = ctx.reshape(x, (B, T * Oh * Ow, Od))

        embeds = ctx.concat([txt, x], axis=1)

        if cfg.use_rotary_positional_embeddings:
            return embeds

        post_h = H // cfg.patch_size
        post_w = W // cfg.patch_size
        temporal_slots = (cfg.sample_frames - 1) // cfg.temporal_compression_ratio + 1

        pos_np = build_joint_pos_embedding(
            cfg.inner_dim,
            cfg.max_text_seq_length,
            (post_h, post_w),
            temporal_slots,
            cfg.spatial_interpolation_scale,
            cfg.temporal_interpolation_scale,
        )
        pos_mx = mx.array(pos_np.astype("float32"))
        pos_mx = pos_mx.astype(embeds.dtype)
        if pos_mx.shape[1] != embeds.shape[1]:
            raise RuntimeError(
                f"CogVideoX positional embedding length {pos_mx.shape[1]} != sequence {embeds.shape[1]} "
                f"(T={T}, H={H}, W={W}, post=({post_h},{post_w}), temporal_slots={temporal_slots})."
            )
        return embeds + pos_mx


class CogVideoXLayerNormZero:
    def __init__(self, ctx: RuntimeContext, conditioning_dim: int, embedding_dim: int,
                 eps: float = 1e-5, bias: bool = True):
        self.ctx = ctx
        self.silu = nn.SiLU()
        self.linear = ctx.Linear(conditioning_dim, 6 * embedding_dim, bias=bias)
        self.norm = ctx.LayerNorm(embedding_dim, eps=eps, affine=True, bias=True)

    def __call__(
        self, hidden_states: Any, encoder_hidden_states: Any, temb: Any,
    ) -> tuple[Any, Any, Any, Any]:
        ctx = self.ctx
        v = self.linear(self.silu(temb))
        D = v.shape[-1] // 6
        shift, scale, gate, enc_shift, enc_scale, enc_gate = (
            v[..., :D], v[..., D:2 * D], v[..., 2 * D:3 * D],
            v[..., 3 * D:4 * D], v[..., 4 * D:5 * D], v[..., 5 * D:6 * D],
        )
        n_h = self.norm(hidden_states) * (1 + scale)[:, None, :] + shift[:, None, :]
        n_e = self.norm(encoder_hidden_states) * (1 + enc_scale)[:, None, :] + enc_shift[:, None, :]
        return n_h, n_e, gate[:, None, :], enc_gate[:, None, :]


class CogVideoXAttention:
    def __init__(
        self,
        ctx: RuntimeContext,
        dim: int,
        heads: int,
        dim_head: int,
        dropout: float = 0.0,
        bias: bool = True,
        out_bias: bool = True,
    ):
        self.ctx = ctx
        self.inner_dim = dim_head * heads
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        self.to_q = ctx.Linear(dim, self.inner_dim, bias=bias)
        self.to_k = ctx.Linear(dim, self.inner_dim, bias=bias)
        self.to_v = ctx.Linear(dim, self.inner_dim, bias=bias)
        self.norm_q = ctx.LayerNorm(dim_head, eps=1e-6, affine=True, bias=True)
        self.norm_k = ctx.LayerNorm(dim_head, eps=1e-6, affine=True, bias=True)
        self.to_out = ctx.ModuleList([
            ctx.Linear(self.inner_dim, dim, bias=out_bias),
            ctx.Dropout(dropout),
        ])

    def __call__(
        self,
        hidden_states: Any,
        encoder_hidden_states: Any,
        image_rotary_emb: tuple[Any, Any] | None,
    ) -> tuple[Any, Any]:
        ctx = self.ctx
        text_seq_length = int(encoder_hidden_states.shape[1])
        hs = ctx.concat([encoder_hidden_states, hidden_states], axis=1)
        B, Seq, _ = hs.shape
        q = self.to_q(hs)
        k = self.to_k(hs)
        v = self.to_v(hs)

        Hh, Dh = self.heads, self.dim_head
        q = ctx.reshape(q, (B, Seq, Hh, Dh))
        q = ctx.permute(q, (0, 2, 1, 3))
        k = ctx.reshape(k, (B, Seq, Hh, Dh))
        k = ctx.permute(k, (0, 2, 1, 3))
        v = ctx.reshape(v, (B, Seq, Hh, Dh))
        v = ctx.permute(v, (0, 2, 1, 3))

        q = self.norm_q(q)
        k = self.norm_k(k)

        if image_rotary_emb is not None:
            cos, sin = image_rotary_emb
            q_t = q[:, :, text_seq_length:, :]
            k_t = k[:, :, text_seq_length:, :]
            q_t = apply_rope_interleaved_real(ctx, q_t, cos, sin)
            k_t = apply_rope_interleaved_real(ctx, k_t, cos, sin)
            q = ctx.concat([q[:, :, :text_seq_length, :], q_t], axis=2)
            k = ctx.concat([k[:, :, :text_seq_length, :], k_t], axis=2)

        attn_out = ctx.attention(q, k, v, scale=self.scale)
        attn_out = ctx.permute(attn_out, (0, 2, 1, 3))
        attn_out = ctx.reshape(attn_out, (B, Seq, self.inner_dim))
        attn_out = self.to_out[0](attn_out)
        attn_out = self.to_out[1](attn_out)

        enc_out = attn_out[:, :text_seq_length, :]
        hid_out = attn_out[:, text_seq_length:, :]
        return hid_out, enc_out


class CogVideoXFeedForward:
    def __init__(self, ctx: RuntimeContext, dim: int, inner_dim: int,
                 dropout: float = 0.0, bias: bool = True):
        self.net = ctx.ModuleList([
            _FFGeluApprox(ctx, dim, inner_dim, bias=bias),
            ctx.Dropout(dropout),
            ctx.Linear(inner_dim, dim, bias=bias),
        ])

    def __call__(self, x: Any) -> Any:
        for layer in self.net:
            x = layer(x)
        return x


class CogVideoXBlock:
    def __init__(self, cfg: CogVideoXConfig, ctx: RuntimeContext):
        self.ctx = ctx
        inner = cfg.inner_dim
        self.norm1 = CogVideoXLayerNormZero(ctx, cfg.time_embed_dim, inner)
        self.attn1 = CogVideoXAttention(
            ctx, inner, cfg.num_attention_heads, cfg.attention_head_dim,
            dropout=cfg.dropout, bias=cfg.attention_bias, out_bias=cfg.attention_out_bias,
        )
        self.norm2 = CogVideoXLayerNormZero(ctx, cfg.time_embed_dim, inner)
        self.ff = CogVideoXFeedForward(ctx, inner, cfg.ff_inner_dim, dropout=cfg.dropout, bias=cfg.ff_bias)

    def __call__(
        self,
        hidden_states: Any,
        encoder_hidden_states: Any,
        temb: Any,
        image_rotary_emb: tuple[Any, Any] | None,
    ) -> tuple[Any, Any]:
        ctx = self.ctx
        text_seq_length = int(encoder_hidden_states.shape[1])

        n_h, n_e, g_msa, ge_msa = self.norm1(hidden_states, encoder_hidden_states, temb)
        ah, ae = self.attn1(n_h, n_e, image_rotary_emb)
        hidden_states = hidden_states + g_msa * ah
        encoder_hidden_states = encoder_hidden_states + ge_msa * ae

        n_h2, n_e2, g_ff, ge_ff = self.norm2(hidden_states, encoder_hidden_states, temb)
        ff_in = ctx.concat([n_e2, n_h2], axis=1)
        ff_o = self.ff(ff_in)
        hidden_states = hidden_states + g_ff * ff_o[:, text_seq_length:]
        encoder_hidden_states = encoder_hidden_states + ge_ff * ff_o[:, :text_seq_length]
        return hidden_states, encoder_hidden_states


class AdaLayerNormLast:
    """Final AdaLayerNorm (`norm_out`) — chunk_dim=1 branch in diffusers."""

    def __init__(self, ctx: RuntimeContext, embedding_dim: int, output_dim: int,
                 eps: float = 1e-5, affine: bool = True):
        self.inner = embedding_dim
        self.silu = nn.SiLU()
        self.linear = ctx.Linear(embedding_dim, output_dim, bias=True)
        self.norm = ctx.LayerNorm(output_dim // 2, eps=eps, affine=affine, bias=True)

    def __call__(self, x: Any, temb: Any) -> Any:
        t = self.linear(self.silu(temb))
        shift = t[:, : self.inner]
        scale = t[:, self.inner :]
        return self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]


class CogVideoXTransformer3D(TransformerBase):
    """Registry export name `CogVideoXTransformer` — identical checkpoint keys to diffusers."""

    def __init__(self, cfg: CogVideoXConfig, ctx: RuntimeContext, num_frames: int = 13):
        self.config = cfg
        self.ctx = ctx
        nn_ctx = ctx

        self.patch_embed = CogVideoXPatchEmbed(cfg, ctx)
        self.embedding_dropout = nn_ctx.Dropout(cfg.dropout)

        self.time_embedding = TimestepEmbeddingLinear(ctx, cfg.inner_dim, cfg.time_embed_dim)

        self.transformer_blocks = nn_ctx.ModuleList([
            CogVideoXBlock(cfg, ctx) for _ in range(cfg.num_layers)
        ])

        self.norm_final = nn_ctx.LayerNorm(cfg.inner_dim, eps=cfg.norm_eps, affine=True, bias=True)
        self.norm_out = AdaLayerNormLast(ctx, cfg.time_embed_dim, 2 * cfg.inner_dim, eps=cfg.norm_eps)
        out_dim = cfg.patch_size * cfg.patch_size * cfg.out_channels
        self.proj_out = nn_ctx.Linear(cfg.inner_dim, out_dim, bias=True)

        self._init_num_frames = num_frames

    def _time_proj_sin(self, timestep: Any) -> Any:
        cfg = self.config
        return _get_timestep_embedding_mx(
            self.ctx, timestep, cfg.inner_dim,
            flip_sin_to_cos=cfg.flip_sin_to_cos,
            downscale_freq_shift=cfg.freq_shift,
            scale=1.0,
        )

    def forward(
        self,
        latents: Any,
        timestep: Any,
        txt_embeds: Any | None = None,
        sigmas: Any | None = None,
        image_rotary_emb: tuple[Any, Any] | None = None,
        **_: Any,
    ) -> Any:
        ctx = self.ctx
        cfg = self.config
        if txt_embeds is None:
            raise RuntimeError("CogVideoX requires T5 embeddings (`txt_embeds`).")

        # VideoPipeline stores latents as [B, C, T, H, W]; diffusers expects [B, T, C, H, W].
        if latents.ndim != 5:
            raise RuntimeError(f"CogVideoX expects 5D latents, got shape {latents.shape}")
        latents_bcthw = ctx.permute(latents, (0, 2, 1, 3, 4))

        emb = self._time_proj_sin(timestep)
        emb = self.time_embedding(emb)

        hidden_states = self.patch_embed(txt_embeds, latents_bcthw)
        hidden_states = self.embedding_dropout(hidden_states)

        text_seq_length = int(txt_embeds.shape[1])
        encoder_hidden_states = hidden_states[:, :text_seq_length]
        hidden_states = hidden_states[:, text_seq_length:]

        for blk in self.transformer_blocks:
            hidden_states, encoder_hidden_states = blk(
                hidden_states, encoder_hidden_states, emb, image_rotary_emb,
            )

        hidden_states = self.norm_final(hidden_states)
        hidden_states = self.norm_out(hidden_states, emb)
        hidden_states = self.proj_out(hidden_states)

        batch_size = latents_bcthw.shape[0]
        num_frames = latents_bcthw.shape[1]
        _, _, _, height, width = latents_bcthw.shape
        p = cfg.patch_size

        # Unpatchify — diffusers `CogVideoXTransformer3DModel.forward` (patch_size_t is None).
        out = ctx.reshape(
            hidden_states,
            (batch_size, num_frames, height // p, width // p, cfg.out_channels, p, p),
        )
        out = ctx.permute(out, (0, 1, 4, 2, 5, 3, 6))
        out = ctx.reshape(out, (batch_size, num_frames, cfg.out_channels, height, width))
        out = ctx.permute(out, (0, 2, 1, 3, 4))
        return out


CogVideoXTransformer = CogVideoXTransformer3D
