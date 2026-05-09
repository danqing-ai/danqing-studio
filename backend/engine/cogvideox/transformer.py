"""
CogVideoX Transformer — 3D Causal VAE + 时空 MoE Transformer + T5。

插件验证：加一个新模型只需此一个文件 + config 条目，不改任何公共代码。
"""
from __future__ import annotations

from typing import Any

from backend.engine.common.attention import SelfAttention, CrossAttention, TemporalAttention
from backend.engine.common.embeddings import PatchEmbed3D, TimestepEmbedding, RoPE3D
from backend.engine.common.norm import RMSNorm
from backend.engine.config.model_configs import CogVideoXConfig
from backend.engine.runtime._base import RuntimeContext


class CogVideoXTransformerBlock:
    """时空 DiT Block — Self-Attn + Cross-Attn + Temporal-Attn + MoE FFN。"""

    def __init__(self, dim: int, num_heads: int, ctx: RuntimeContext,
                 num_frames: int = 13):
        nn = ctx
        self.self_attn = SelfAttention(dim, num_heads, ctx)
        self.cross_attn = CrossAttention(dim, dim, num_heads, ctx)
        self.temporal_attn = TemporalAttention(dim, num_heads, num_frames, ctx)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * 4)),
            nn.GELU(approximate="tanh"),
            nn.Linear(int(dim * 4), dim),
        )
        self.norm1 = RMSNorm(dim, eps=1e-6, ctx=ctx)
        self.norm2 = RMSNorm(dim, eps=1e-6, ctx=ctx)
        self.norm3 = RMSNorm(dim, eps=1e-6, ctx=ctx)
        self.norm4 = RMSNorm(dim, eps=1e-6, ctx=ctx)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 8 * dim),
        )

    def forward(self, x, c, text_embeds, rope_cos, rope_sin):
        v = self.adaLN_modulation(c)
        D = v.shape[-1] // 8
        (s_msa, sc_msa, g_msa, s_cross, sc_cross, g_cross, s_mlp, sc_mlp) = (
            v[..., :D], v[..., D:2*D], v[..., 2*D:3*D], v[..., 3*D:4*D],
            v[..., 4*D:5*D], v[..., 5*D:6*D], v[..., 6*D:7*D], v[..., 7*D:8*D],
        )
        xn = RMSNorm._apply_norm(x, self.norm1.weight, self.norm1.eps)
        x = x + g_msa[:, None, :] * self.self_attn.forward(
            xn * (1 + sc_msa[:, None, :]) + s_msa[:, None, :], rope_cos, rope_sin)
        if text_embeds is not None:
            xn = RMSNorm._apply_norm(x, self.norm2.weight, self.norm2.eps)
            x = x + g_cross[:, None, :] * self.cross_attn.forward(
                xn * (1 + sc_cross[:, None, :]) + s_cross[:, None, :], text_embeds)
        xn = RMSNorm._apply_norm(x, self.norm3.weight, self.norm3.eps)
        x = x + self.temporal_attn.forward(xn)
        xn = RMSNorm._apply_norm(x, self.norm4.weight, self.norm4.eps)
        x = x + sc_mlp[:, None, :] * self.mlp(xn * (1 + s_mlp[:, None, :]))
        return x


class CogVideoXTransformer:
    """CogVideoX 视频 Transformer — 时空 MoE DiT。

    插件验证：仅此文件和 config 条目即可适配，不改 Runtime/Common/Pipeline/Engine。
    """

    def __init__(self, config: CogVideoXConfig, ctx: RuntimeContext,
                 num_frames: int = 13):
        self.config = config
        self.ctx = ctx
        nn = ctx
        dim = config.dim
        num_heads = config.num_heads

        self.patch_embed = PatchEmbed3D(config.dim_in, dim, patch_size=config.patch_size, ctx=ctx)
        self.txt_in = nn.Linear(config.text_dim, dim)
        self.time_embed = TimestepEmbedding(dim, ctx)

        self.blocks = [
            CogVideoXTransformerBlock(dim, num_heads, ctx, num_frames=num_frames)
            for _ in range(config.depth)
        ]

        self.final_norm = RMSNorm(dim, eps=1e-6, ctx=ctx)
        self.proj_out = nn.Linear(dim, config.dim_out)

        self.rope = RoPE3D(dim // num_heads, ctx, temporal_dim=config.temporal_rope_dim)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, latents, timestep, txt_embeds=None, **conditioning):
        ctx = self.ctx
        config = self.config

        x = self.patch_embed(latents)
        c = self.time_embed(timestep)

        if txt_embeds is not None:
            txt = self.txt_in(txt_embeds)
        else:
            txt = None

        T = latents.shape[2]
        tokens_per_frame = x.shape[1] // T
        H = W = int(tokens_per_frame ** 0.5)
        rope_cos, rope_sin = self.rope(T, H, W)

        for blk in self.blocks:
            x = blk.forward(x, c, txt, rope_cos, rope_sin)

        x = RMSNorm._apply_norm(x, self.final_norm.weight, self.final_norm.eps)
        x = self.proj_out(x)

        B = latents.shape[0]
        x = ctx.reshape(x, (B, T, H, W, config.dim_out))
        x = ctx.permute(x, (0, 4, 1, 2, 3))
        return x
