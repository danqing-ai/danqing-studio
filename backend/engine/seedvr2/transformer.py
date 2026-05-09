"""
SeedVR2 Transformer — 参考 mflux 项目 SeedVR2 实现。

无条件 DiT — 单步超分。低分辨率 latent 条件注入。
"""
from __future__ import annotations

from typing import Any

from backend.engine.common.attention import SelfAttention
from backend.engine.common.embeddings import PatchEmbed2D, TimestepEmbedding, RoPE2D
from backend.engine.common.norm import RMSNorm
from backend.engine.config.model_configs import SeedVR2Config
from backend.engine.runtime._base import RuntimeContext
from backend.engine.common._base import TransformerBase


class SeedVR2TransformerBlock:
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


class SeedVR2Transformer(TransformerBase):
    """SeedVR2 无条件 DiT — 超分专用 (无文本编码器)。"""

    def __init__(self, config: SeedVR2Config, ctx: RuntimeContext):
        self.config = config
        self.ctx = ctx
        nn = ctx
        dim = config.hidden_dim
        num_heads = config.num_heads

        self.patch_embed = PatchEmbed2D(config.in_channels, dim, patch_size=1, ctx=ctx)
        # 低分辨率条件嵌入 (channel-concat → double in_channels before patch)
        self.time_in = TimestepEmbedding(dim, ctx)
        self.rope = RoPE2D(config.rope_dim, ctx)

        self.blocks = [
            SeedVR2TransformerBlock(dim, num_heads, ctx)
            for _ in range(config.num_layers)
        ]

        self.final_norm = RMSNorm(dim, eps=1e-6, ctx=ctx)
        self.proj_out = nn.Linear(dim, config.out_channels)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, latents, timestep,
                txt_embeds=None, lowres_latents=None, **conditioning):
        ctx = self.ctx
        config = self.config
        B = latents.shape[0]

        # 条件注入：低分辨率 latent 与 noise latent channel-concat
        if lowres_latents is not None:
            latents = ctx.concat([latents, lowres_latents], axis=1)

        img = self.patch_embed(latents)
        img_seq_len = img.shape[1]

        c = self.time_in(timestep)

        H = W = int(img_seq_len ** 0.5)
        rope_cos, rope_sin = self.rope(H, W)

        x = img
        for blk in self.blocks:
            x = blk.forward(x, c, rope_cos, rope_sin)

        x = RMSNorm._apply_norm(x, self.final_norm.weight, self.final_norm.eps)
        x = self.proj_out(x)
        H = W = int(img_seq_len ** 0.5)
        x = ctx.reshape(x, (B, H, W, config.out_channels))
        x = ctx.permute(x, (0, 3, 1, 2))
        return x
