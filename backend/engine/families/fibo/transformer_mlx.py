"""
FIBO Transformer — Reference implementation.

DiT + JSON structured prompt encoding. 50-step default. CFG via batch-dim.
"""
from __future__ import annotations

import mlx.core as mx

from backend.engine.common.attention import SelfAttention
from backend.engine.common.embeddings import PatchEmbed2D, TimestepEmbedding, RoPE2D
from backend.engine.common.norm import RMSNorm
from backend.engine.config.model_configs import FIBOConfig
from backend.engine.runtime._base import RuntimeContext
from backend.engine.common._base import TransformerBase


def _rope_extend_concat_txt(ctx, cos, sin, txt_len: int):
    """Append identity RoPE (cos=1, sin=0) for text tokens after image tokens."""
    if txt_len <= 0:
        return cos, sin
    R = cos.shape[-1]
    b1, b2, _, _ = cos.shape
    pad_cos = ctx.ones((b1, b2, txt_len, R), dtype=cos.dtype)
    pad_sin = ctx.zeros((b1, b2, txt_len, R), dtype=sin.dtype)
    return ctx.concat([cos, pad_cos], axis=2), ctx.concat([sin, pad_sin], axis=2)


class FIBOTransformerBlock:
    def __init__(self, dim: int, num_heads: int, ctx: RuntimeContext):
        nn = ctx
        self.attn = SelfAttention(dim, num_heads, ctx)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * 4)),
            nn.GELU(approximate="tanh"),
            nn.Linear(int(dim * 4), dim),
        )
        self.norm1 = RMSNorm(dim, eps=1e-6, ctx=ctx)
        self.norm2 = RMSNorm(dim, eps=1e-6, ctx=ctx)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim),
        )

    def forward(self, x, c, rope_cos, rope_sin):
        v = self.adaLN_modulation(c)
        D = v.shape[-1] // 6
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            v[..., :D], v[..., D:2*D], v[..., 2*D:3*D],
            v[..., 3*D:4*D], v[..., 4*D:5*D], v[..., 5*D:6*D],
        )
        xn = RMSNorm._apply_norm(x, self.norm1.weight, self.norm1.eps)
        x = x + gate_msa[:, None, :] * self.attn.forward(
            xn * (1 + scale_msa[:, None, :]) + shift_msa[:, None, :], rope_cos, rope_sin,
        )
        xn = RMSNorm._apply_norm(x, self.norm2.weight, self.norm2.eps)
        x = x + gate_mlp[:, None, :] * self.mlp(
            xn * (1 + scale_mlp[:, None, :]) + shift_mlp[:, None, :],
        )
        return x


class FIBOTransformer(TransformerBase):
    """FIBO Transformer — 单流 DiT，结构化 prompt。"""

    def __init__(self, config: FIBOConfig, ctx: RuntimeContext):
        self.config = config
        self.ctx = ctx
        nn = ctx
        dim = config.hidden_dim
        num_heads = config.num_heads

        self.patch_embed = PatchEmbed2D(config.in_channels, dim, patch_size=1, ctx=ctx)
        self.txt_in = nn.Linear(config.text_dim, dim)
        self.time_in = TimestepEmbedding(dim, ctx)
        self.rope = RoPE2D(config.rope_dim, ctx)

        self.blocks = [
            FIBOTransformerBlock(dim, num_heads, ctx)
            for _ in range(config.num_layers)
        ]

        self.final_norm = RMSNorm(dim, eps=1e-6, ctx=ctx)
        self.proj_out = nn.Linear(dim, config.out_channels)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, latents, timestep, txt_embeds=None, sigmas=None, **conditioning):
        ctx = self.ctx
        config = self.config
        B = latents.shape[0]

        img = self.patch_embed(latents)
        img_seq_len = img.shape[1]

        if txt_embeds is not None:
            txt = self.txt_in(txt_embeds)
            x = ctx.concat([img, txt], axis=1)
            txt_len = txt.shape[1]
        else:
            x = img
            txt_len = 0

        timestep_embed_value = conditioning.get("timestep_embed_value")
        if timestep_embed_value is not None:
            t_val = float(timestep_embed_value)
        elif sigmas is not None:
            t_idx = int(timestep)
            n = int(sigmas.shape[0]) if hasattr(sigmas, "shape") else len(sigmas)
            sigma_t = sigmas[t_idx] if t_idx < n else sigmas[-1] if n > 0 else 1.0
            t_val = float(mx.reshape(mx.array(sigma_t), (-1,))[0]) * 1000.0
        else:
            tv = timestep
            if isinstance(tv, mx.array):
                t_val = float(mx.reshape(tv, (-1,))[0])
            else:
                t_val = float(tv)
            if t_val <= 1.0 + 1e-5:
                t_val *= 1000.0

        t_batch = mx.full((B,), t_val, dtype=mx.float32)
        c = self.time_in(t_batch)

        H = W = int(img_seq_len ** 0.5)
        rope_cos, rope_sin = self.rope(H, W)
        rope_cos, rope_sin = _rope_extend_concat_txt(ctx, rope_cos, rope_sin, txt_len)

        for blk in self.blocks:
            x = blk.forward(x, c, rope_cos, rope_sin)

        x = RMSNorm._apply_norm(x[:, :img_seq_len], self.final_norm.weight, self.final_norm.eps)
        x = self.proj_out(x)
        H = W = int(img_seq_len ** 0.5)
        x = ctx.reshape(x, (B, H, W, config.out_channels))
        x = ctx.permute(x, (0, 3, 1, 2))
        return x
