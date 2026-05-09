"""
Flux.1 Transformer (MM-DiT) — 参考 mflux 项目实现。

Flux.1 系列 (schnell / dev / fill / depth / kontext / redux / controlnet) 共用此架构。
差异由 config 参数控制 (supports_guidance, supports_mask 等)。
"""
from __future__ import annotations

from typing import Any, Optional

from backend.engine.common.attention import SelfAttention
from backend.engine.common.embeddings import PatchEmbed2D, TimestepEmbedding, RoPE2D
from backend.engine.common.norm import RMSNorm
from backend.engine.config.model_configs import Flux1Config
from backend.engine.runtime._base import RuntimeContext


class Flux1SingleBlock:
    """Flux.1 单流 block — 文本自注意力 + MLP。"""

    def __init__(self, dim: int, num_heads: int, ctx: RuntimeContext, idx: int):
        nn = ctx
        self.idx = idx
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

    def forward(self, x, c, rope_cos, rope_sin, txt_seq_len):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        # 切片到文本部分 (single blocks only process text)
        x = x[:, :txt_seq_len]
        # Attention with AdaLN
        x_norm = RMSNorm._apply_norm(x, self.norm1.weight, self.norm1.eps)
        attn_out = self.attn.forward(
            x_norm * (1 + scale_msa[:, None, :]) + shift_msa[:, None, :],
            rope_cos, rope_sin,
        )
        x = x + gate_msa[:, None, :] * attn_out
        # MLP with AdaLN
        x_norm = RMSNorm._apply_norm(x, self.norm2.weight, self.norm2.eps)
        mlp_out = self.mlp(x_norm * (1 + scale_mlp[:, None, :]) + shift_mlp[:, None, :])
        x = x + gate_mlp[:, None, :] * mlp_out
        return x


class Flux1JointBlock:
    """Flux.1 双流 block (MM-DiT) — 图像+文本联合注意力 + MLP。"""

    def __init__(self, dim: int, num_heads: int, ctx: RuntimeContext, idx: int):
        nn = ctx
        self.idx = idx
        self.img_attn = SelfAttention(dim, num_heads, ctx)
        self.txt_attn = SelfAttention(dim, num_heads, ctx)
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

    def forward(self, x, c, rope_cos, rope_sin, img_seq_len):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x_norm = RMSNorm._apply_norm(x, self.norm1.weight, self.norm1.eps)
        x_mod = x_norm * (1 + scale_msa[:, None, :]) + shift_msa[:, None, :]
        # 分别处理图像和文本部分
        img_x = x_mod[:, :img_seq_len]
        txt_x = x_mod[:, img_seq_len:]
        img_attn = self.img_attn.forward(img_x, rope_cos, rope_sin)
        txt_attn = self.txt_attn.forward(txt_x, rope_cos[:, :img_seq_len] if rope_cos is not None else None, rope_sin[:, :img_seq_len] if rope_sin is not None else None)
        attn_out = self.ctx.concat([img_attn, txt_attn], axis=1)
        x = x + gate_msa[:, None, :] * attn_out
        # MLP
        x_norm = RMSNorm._apply_norm(x, self.norm2.weight, self.norm2.eps)
        mlp_out = self.mlp(x_norm * (1 + scale_mlp[:, None, :]) + shift_mlp[:, None, :])
        x = x + gate_mlp[:, None, :] * mlp_out
        return x


class Flux1Transformer:
    """Flux.1 Transformer — MM-DiT 双流架构。

    处理:
    - txt_ids: T5 文本嵌入 → Linear → [B, txt_seq, dim]
    - clip_ids: CLIP 文本嵌入 → Linear → [B, clip_seq, dim]
    - img: VAE latent → PatchEmbed2D → [B, img_seq, dim]
    - timestep: t → TimestepEmbedding → [B, dim]
    - pooled: CLIP pooled → TimestepEmbedding → [B, dim]

    双流 (joint blocks): 图像 token 和文本 token 在同一个序列，
    但分开做 self-attention。
    """

    def __init__(self, config: Flux1Config, ctx: RuntimeContext):
        self.config = config
        self.ctx = ctx
        nn = ctx
        dim = config.hidden_dim
        num_heads = config.num_heads

        self.patch_embed = PatchEmbed2D(config.in_channels, dim,
                                        patch_size=1, ctx=ctx)
        self.txt_in = nn.Linear(config.text_dim, dim)
        self.clip_in = nn.Linear(config.clip_dim, dim) if config.clip_dim else None
        self.time_in = TimestepEmbedding(dim, ctx)
        self.vector_in = TimestepEmbedding(dim, ctx)
        self.rope = RoPE2D(config.rope_dim, ctx)

        self.single_blocks = [
            Flux1SingleBlock(dim, num_heads, ctx, i)
            for i in range(config.num_single_layers)
        ]
        self.joint_blocks = [
            Flux1JointBlock(dim, num_heads, ctx, i)
            for i in range(config.num_joint_layers)
        ]

        self.final_norm = RMSNorm(dim, eps=1e-6, ctx=ctx)
        self.proj_out = nn.Linear(dim, config.out_channels)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, latents, timestep,
                txt_embeds=None, clip_embeds=None,
                pooled_embeds=None, **conditioning):
        ctx = self.ctx
        config = self.config
        B = latents.shape[0]

        # 1. 图像编码
        img = self.patch_embed(latents)  # [B, img_seq, dim]
        img_seq_len = img.shape[1]

        # 2. 文本编码
        if txt_embeds is not None and self.txt_in is not None:
            txt = self.txt_in(txt_embeds)  # [B, txt_seq, dim]
        else:
            txt = ctx.zeros((B, 0, config.hidden_dim), dtype=ctx.float32())

        if clip_embeds is not None and self.clip_in is not None:
            clip_txt = self.clip_in(clip_embeds)
            txt = ctx.concat([txt, clip_txt], axis=1)
        txt_seq_len = txt.shape[1]

        # 3. 拼接序列
        x = ctx.concat([img, txt], axis=1)  # [B, img_seq + txt_seq, dim]
        total_seq = img_seq_len + txt_seq_len

        # 4. 时间嵌入 + 池化向量
        c = self.time_in(timestep)
        if pooled_embeds is not None:
            c = c + self.vector_in(pooled_embeds)

        # 5. RoPE (图像分辨率)
        H = W = int(img_seq_len ** 0.5)
        rope_cos, rope_sin = self.rope(H, W)

        # 6. Single blocks → 只处理文本
        for blk in self.single_blocks:
            x = blk.forward(x, c, rope_cos, rope_sin, total_seq)

        # 7. Joint blocks → 图像+文本分开做 attn
        for blk in self.joint_blocks:
            x = blk.forward(x, c, rope_cos, rope_sin, img_seq_len)

        # 8. 输出投影
        x = RMSNorm._apply_norm(x[:, :img_seq_len], self.final_norm.weight, self.final_norm.eps)
        x = self.proj_out(x)

        # 9. Patches → image
        H = W = int(img_seq_len ** 0.5)
        x = ctx.reshape(x, (B, H, W, config.out_channels))
        x = ctx.permute(x, (0, 3, 1, 2))
        return x
