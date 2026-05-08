"""
注意力模块 — 所有模型共用。

参考 mflux 项目的 Flux1/Flux2 Attention 实现和 mlx-video 的时空注意力。
"""
from __future__ import annotations

from typing import Any, Optional


class SelfAttention:
    """标准自注意力 (QKV 投影 + RoPE + Scaled Dot-Product)。

    用于: Flux1 / Flux2 / Qwen / FIBO / Z-Image 图像模型的空间注意力。
    """

    def __init__(self, dim: int, num_heads: int, ctx: Any,
                 qk_norm: bool = True, qkv_bias: bool = True):
        self.ctx = ctx
        nn = ctx
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        if qk_norm:
            self.q_norm = nn.RMSNorm(self.head_dim)
            self.k_norm = nn.RMSNorm(self.head_dim)
        else:
            self.q_norm = None
            self.k_norm = None
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, rope_cos=None, rope_sin=None,
                mask=None) -> Any:
        ctx = self.ctx
        B, N, C = x.shape[0], x.shape[1], self.dim

        qkv = self.qkv(x)
        qkv = ctx.reshape(qkv, (B, N, 3, self.num_heads, self.head_dim))
        qkv = ctx.permute(qkv, (2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if rope_cos is not None and rope_sin is not None:
            q = _apply_rope(ctx, q, rope_cos, rope_sin)
            k = _apply_rope(ctx, k, rope_cos, rope_sin)

        attn_out = ctx.attention(q, k, v, scale=self.scale, mask=mask)
        attn_out = ctx.permute(attn_out, (0, 2, 1, 3))
        attn_out = ctx.reshape(attn_out, (B, N, C))
        return self.proj(attn_out)


class CrossAttention:
    """交叉注意力 (Q 来自 latents, KV 来自 text_embeds)。

    用于: Flux1 MM-DiT 的 cross-attn / LTX / Wan 的 text cross-attn。
    """

    def __init__(self, query_dim: int, ctx_dim: int, num_heads: int, ctx: Any,
                 out_dim: Optional[int] = None, qk_norm: bool = True,
                 qkv_bias: bool = True):
        self.ctx = ctx
        nn = ctx
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        self.scale = self.head_dim ** -0.5
        out_dim = out_dim or query_dim

        self.q = nn.Linear(query_dim, query_dim, bias=qkv_bias)
        self.k = nn.Linear(ctx_dim, query_dim, bias=qkv_bias)
        self.v = nn.Linear(ctx_dim, query_dim, bias=qkv_bias)
        if qk_norm:
            self.q_norm = nn.RMSNorm(self.head_dim)
            self.k_norm = nn.RMSNorm(self.head_dim)
        else:
            self.q_norm = None
            self.k_norm = None
        self.proj = nn.Linear(query_dim, out_dim)

    def forward(self, x, context, mask=None) -> Any:
        ctx = self.ctx
        B, N, C = x.shape[0], x.shape[1], self.q.in_features

        q = ctx.reshape(self.q(x), (B, N, self.num_heads, self.head_dim))
        q = ctx.permute(q, (0, 2, 1, 3))

        ctx_len = context.shape[1]
        k = ctx.reshape(self.k(context), (B, ctx_len, self.num_heads, self.head_dim))
        k = ctx.permute(k, (0, 2, 1, 3))
        v = ctx.reshape(self.v(context), (B, ctx_len, self.num_heads, self.head_dim))
        v = ctx.permute(v, (0, 2, 1, 3))

        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        attn_out = ctx.attention(q, k, v, scale=self.scale, mask=mask)
        attn_out = ctx.permute(attn_out, (0, 2, 1, 3))
        attn_out = ctx.reshape(attn_out, (B, N, self.q.in_features))
        return self.proj(attn_out)


class TemporalAttention:
    """时序自注意力 — 沿帧维度做 Self-Attention。

    用于: LTX / Wan 的时序混合层。输入 [B*F, HW, C]，reshape+permute 后做沿帧的 attn。
    """

    def __init__(self, dim: int, num_heads: int, num_frames: int, ctx: Any,
                 qk_norm: bool = True, qkv_bias: bool = True):
        self.ctx = ctx
        nn = ctx
        self.dim = dim
        self.num_heads = num_heads
        self.num_frames = num_frames
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        if qk_norm:
            self.q_norm = nn.RMSNorm(self.head_dim)
            self.k_norm = nn.RMSNorm(self.head_dim)
        else:
            self.q_norm = None
            self.k_norm = None
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, rope_cos=None, rope_sin=None) -> Any:
        ctx = self.ctx
        # x: [B*F, H*W, C] → [B, F, H*W, C]
        BF, HW, C = x.shape[0], x.shape[1], self.dim
        F = self.num_frames
        B = BF // F
        H = W = int(HW ** 0.5)  # Square image assumption; flexible

        x_reshaped = ctx.reshape(x, (B, F, HW, C))
        x_t = ctx.permute(x_reshaped, (0, 2, 1, 3))  # [B, HW, F, C]
        x_t = ctx.reshape(x_t, (B * HW, F, C))

        qkv = self.qkv(x_t)
        qkv = ctx.reshape(qkv, (B * HW, F, 3, self.num_heads, self.head_dim))
        qkv = ctx.permute(qkv, (2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if rope_cos is not None and rope_sin is not None:
            q = _apply_rope(ctx, q, rope_cos, rope_sin)
            k = _apply_rope(ctx, k, rope_cos, rope_sin)

        attn_out = ctx.attention(q, k, v, scale=self.scale)
        attn_out = ctx.permute(attn_out, (0, 2, 1, 3))
        attn_out = ctx.reshape(attn_out, (B * HW, F, C))
        proj_out = self.proj(attn_out)
        proj_out = ctx.reshape(proj_out, (B, HW, F, C))
        proj_out = ctx.permute(proj_out, (0, 2, 1, 3))  # [B, F, HW, C]
        proj_out = ctx.reshape(proj_out, (BF, HW, C))
        return proj_out


def _apply_rope(ctx, x, cos, sin):
    """将 RoPE 应用于 Q/K 张量。

    x: [B, num_heads, seq, head_dim]
    cos/sin: [1, 1, seq, rope_dim]
    """
    rope_dim = cos.shape[-1]
    x_rope = x[..., :rope_dim]
    x_pass = x[..., rope_dim:]
    x_rotated = x_rope * cos + _rotate_half(ctx, x_rope) * sin
    return ctx.concat([x_rotated, x_pass], axis=-1)


def _rotate_half(ctx, x):
    """Rotate half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return ctx.concat([-x2, x1], axis=-1)
