"""
Wan Video Transformer — 参考公开 Wan 视频 DiT 实现。

双模型（高/低噪声）时空 DiT + T5 文本编码器。
"""
from __future__ import annotations

from typing import Any

from backend.engine.common.attention import SelfAttention, CrossAttention, TemporalAttention
from backend.engine.common._base import TransformerBase
from backend.engine.common.embeddings import PatchEmbed3D, TimestepEmbedding, RoPE3D
from backend.engine.common.norm import RMSNorm
from backend.engine.config.model_configs import WanConfig
from backend.engine.runtime._base import RuntimeContext


class WanTransformerBlock:
    """Wan Transformer Block — 自注意力 + 交叉注意力 + (可选) 时序注意力。"""

    def __init__(self, dim: int, num_heads: int, ctx: RuntimeContext,
                 has_temporal: bool = False, num_frames: int = 1):
        nn = ctx
        self.has_temporal = has_temporal
        self.self_attn = SelfAttention(dim, num_heads, ctx)
        self.cross_attn = CrossAttention(dim, dim, num_heads, ctx)
        if has_temporal:
            self.temporal_attn = TemporalAttention(dim, num_heads, num_frames, ctx)

        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * 4)),
            nn.GELU(approximate="tanh"),
            nn.Linear(int(dim * 4), dim),
        )
        self.norm1 = RMSNorm(dim, eps=1e-6, ctx=ctx)
        self.norm2 = RMSNorm(dim, eps=1e-6, ctx=ctx)
        if has_temporal:
            self.norm_temporal = RMSNorm(dim, eps=1e-6, ctx=ctx)
        self.norm3 = RMSNorm(dim, eps=1e-6, ctx=ctx)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim if not has_temporal else 9 * dim),
        )

    def forward(self, x, c, text_embeds, rope_cos, rope_sin):
        num_splits = 3 if not self.has_temporal else 4
        v = self.adaLN_modulation(c)
        D = v.shape[-1] // (num_splits * 2 + num_splits)

        shift_msa, scale_msa, gate_msa, shift_cross, scale_cross, gate_cross = (
            v[..., :D], v[..., D:2*D], v[..., 2*D:3*D],
            v[..., 3*D:4*D], v[..., 4*D:5*D], v[..., 5*D:6*D],
        )

        xn = RMSNorm._apply_norm(x, self.norm1.weight, self.norm1.eps)
        x = x + gate_msa[:, None, :] * self.self_attn.forward(
            xn * (1 + scale_msa[:, None, :]) + shift_msa[:, None, :], rope_cos, rope_sin,
        )

        if text_embeds is not None:
            xn = RMSNorm._apply_norm(x, self.norm2.weight, self.norm2.eps)
            x = x + gate_cross[:, None, :] * self.cross_attn.forward(
                xn * (1 + scale_cross[:, None, :]) + shift_cross[:, None, :], text_embeds,
            )

        if self.has_temporal:
            shift_temp = v[..., 6*D:7*D]
            scale_temp = v[..., 7*D:8*D]
            gate_temp = v[..., 8*D:9*D]
            xn = RMSNorm._apply_norm(x, self.norm_temporal.weight, self.norm_temporal.eps)
            x = x + gate_temp[:, None, :] * self.temporal_attn.forward(
                xn * (1 + scale_temp[:, None, :]) + shift_temp[:, None, :],
            )

        xn = RMSNorm._apply_norm(x, self.norm3.weight, self.norm3.eps)
        x = x + self.mlp(xn)
        return x


class WanTransformer(TransformerBase):
    """Wan Video Transformer — 双模型时空 DiT。

    双模型架构：high_noise 模型 (粗去噪) + low_noise 模型 (细去噪)。
    双模型共享相同架构，仅在权重上有差异。
    """

    def __init__(self, config: WanConfig, ctx: RuntimeContext,
                 num_frames: int = 81):
        self.config = config
        self.ctx = ctx
        nn = ctx
        dim = config.dim
        num_heads = config.num_heads

        self.patch_embed = PatchEmbed3D(
            config.dim_in, dim, patch_size=config.patch_size, ctx=ctx,
        )
        self.time_embed = TimestepEmbedding(dim, ctx)

        self.blocks = []
        for i in range(config.depth):
            has_temporal = (i + 1) % config.temporal_attn_every == 0
            self.blocks.append(
                WanTransformerBlock(dim, num_heads, ctx,
                                    has_temporal=has_temporal, num_frames=num_frames)
            )

        self.final_norm = RMSNorm(dim, eps=1e-6, ctx=ctx)
        self.proj_out = nn.Linear(dim, config.dim_out)

        self.rope = RoPE3D(dim // num_heads, ctx,
                          temporal_dim=config.temporal_rope_dim)

    def forward(self, latents, timestep,
                txt_embeds=None, image_embeds=None, **conditioning):
        ctx = self.ctx
        config = self.config

        x = self.patch_embed(latents)

        c = self.time_embed(timestep)

        if image_embeds is not None:
            x = x + image_embeds

        T = latents.shape[2] if len(latents.shape) >= 5 else 1
        tokens_per_frame = x.shape[1] // T
        H = W = int(tokens_per_frame ** 0.5)
        rope_cos, rope_sin = self.rope(T, H, W)

        for blk in self.blocks:
            x = blk.forward(x, c, txt_embeds, rope_cos, rope_sin)

        x = RMSNorm._apply_norm(x, self.final_norm.weight, self.final_norm.eps)
        x = self.proj_out(x)

        B = latents.shape[0]
        x = ctx.reshape(x, (B, T, H, W, config.dim_out))
        x = ctx.permute(x, (0, 4, 1, 2, 3))
        return x
