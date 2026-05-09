"""
LTX Video Transformer — 参考 mlx-video 项目实现。

单流时空 DiT + T5 文本编码器。
支持 distilled / dev / dev_two_stage 管线模式。
"""
from __future__ import annotations

from typing import Any

from backend.engine.common.attention import SelfAttention, CrossAttention, TemporalAttention
from backend.engine.common.embeddings import PatchEmbed3D, TimestepEmbedding, RoPE3D
from backend.engine.common.norm import RMSNorm
from backend.engine.config.model_configs import LTXConfig
from backend.engine.runtime._base import RuntimeContext


class LTXBlock:
    """LTX Transformer Block — 自注意力 + 交叉注意力 + (可选) 时序注意力。"""

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
        ctx = x.ctx if hasattr(x, 'ctx') else None

        if self.has_temporal:
            shift_msa, scale_msa, gate_msa, shift_cross, scale_cross, gate_cross, shift_temp, scale_temp, gate_temp = (
                self.adaLN_modulation(c).chunk(9, dim=-1)
            )
        else:
            shift_msa, scale_msa, gate_msa, shift_cross, scale_cross, gate_cross = (
                self.adaLN_modulation(c).chunk(6, dim=-1)
            )

        # Self-attention
        x_norm = RMSNorm._apply_norm(x, self.norm1.weight, self.norm1.eps)
        x_mod = x_norm * (1 + scale_msa[:, None, :]) + shift_msa[:, None, :]
        x = x + gate_msa[:, None, :] * self.self_attn.forward(x_mod, rope_cos, rope_sin)

        # Cross-attention (text)
        if text_embeds is not None:
            x_norm = RMSNorm._apply_norm(x, self.norm2.weight, self.norm2.eps)
            x_mod = x_norm * (1 + scale_cross[:, None, :]) + shift_cross[:, None, :]
            x = x + gate_cross[:, None, :] * self.cross_attn.forward(x_mod, text_embeds)

        # Temporal attention
        if self.has_temporal:
            x_norm = RMSNorm._apply_norm(x, self.norm_temporal.weight, self.norm_temporal.eps)
            x_mod = x_norm * (1 + scale_temp[:, None, :]) + shift_temp[:, None, :]
            x = x + gate_temp[:, None, :] * self.temporal_attn.forward(x_mod)

        # MLP
        x_norm = RMSNorm._apply_norm(x, self.norm3.weight, self.norm3.eps)
        x = x + self.mlp(x_norm)

        return x


class LTXTransformer:
    """LTX Video Transformer — 单流时空 DiT。

    流程:
    1. VAE latent [B, C, T, H, W] → PatchEmbed3D → [B, T*H*W, dim]
    2. T5 text → CrossAttention
    3. Timestep → AdaLN 调制每一层
    4. 每 N 层插入 TemporalAttention
    """

    def __init__(self, config: LTXConfig, ctx: RuntimeContext,
                 num_frames: int = 33):
        self.config = config
        self.ctx = ctx
        nn = ctx
        dim = config.dim
        num_heads = config.num_heads

        self.patch_embed = PatchEmbed3D(
            config.dim_in, dim,
            patch_size=(config.temporal_patch_size, config.patch_size, config.patch_size),
            ctx=ctx,
        )
        self.time_embed = TimestepEmbedding(dim, ctx)

        self.blocks = []
        for i in range(config.depth):
            has_temporal = (i + 1) % config.temporal_attn_every == 0
            self.blocks.append(
                LTXBlock(dim, num_heads, ctx,
                         has_temporal=has_temporal, num_frames=num_frames)
            )

        self.final_norm = RMSNorm(dim, eps=1e-6, ctx=ctx)
        self.proj_out = nn.Linear(dim, config.dim_out)

        self.rope = RoPE3D(dim // num_heads, ctx,
                          temporal_dim=config.temporal_rope_dim)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, latents, timestep,
                txt_embeds=None, image_embeds=None, **conditioning):
        ctx = self.ctx
        config = self.config
        B = latents.shape[0]

        # 1. Patch Embedding
        x = self.patch_embed(latents)  # [B, T*H*W, dim]

        # 2. 时间嵌入
        c = self.time_embed(timestep)

        # 3. 图像条件 (I2V)
        if image_embeds is not None:
            x = x + image_embeds

        # 4. RoPE 3D
        T = latents.shape[2] if len(latents.shape) >= 5 else 1
        tokens_per_frame = x.shape[1] // T
        H = W = int(tokens_per_frame ** 0.5)
        rope_cos, rope_sin = self.rope(T, H, W)

        # 5. Transformer blocks
        for blk in self.blocks:
            x = blk.forward(x, c, txt_embeds, rope_cos, rope_sin)

        # 6. 输出投影
        x = RMSNorm._apply_norm(x, self.final_norm.weight, self.final_norm.eps)
        x = self.proj_out(x)

        # 7. Patches → latent
        x = ctx.reshape(x, (B, T, H, W, config.dim_out))
        x = ctx.permute(x, (0, 4, 1, 2, 3))  # [B, C, T, H, W]
        return x
