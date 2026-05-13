"""
Z-Image Transformer — Reference implementation.

Architecture: ZImageAttention + FeedForward (SiLU-gated) + AdaLN modulation
      + RopeEmbedder (complex freq) + TimestepEmbedder
      + noise_refiner / context_refiner / main layers
      + FinalLayer (AdaLN + Linear output)

Parameters from ZImageConfig dataclass.
"""
from __future__ import annotations

import math
from typing import Any

from backend.engine.config.model_configs import ZImageConfig
from backend.engine.runtime._base import RuntimeContext
from backend.engine.common._base import TransformerBase


def _coerce_timestep_index(t: Any, ctx: RuntimeContext) -> int | None:
    """Return integer timestep index if ``t`` is index-like; else ``None``."""
    if isinstance(t, bool):
        return None
    if isinstance(t, int):
        return t
    try:
        import numpy as np
        if isinstance(t, np.integer):
            return int(t)
    except ImportError:
        pass
    if ctx.is_tensor(t) and ctx.is_integer_dtype_tensor(t):
        return int(t.item())
    return None


# =========================================================================
# RopeEmbedder — 复数频率 RoPE
# =========================================================================

class RopeEmbedder:
    """Precompute RoPE frequencies (complex form cos/sin stacked).

    Reference RopeEmbedder: axes_dims=[32,48,48], theta=256.0.
    """

    def __init__(self, config_or_dims: ZImageConfig | list[int] = None,
                 ctx: RuntimeContext = None,
                 theta: float = 256.0,
                 axes_dims: list[int] | None = None,
                 axes_lens: list[int] | None = None):
        if isinstance(config_or_dims, ZImageConfig):
            self.theta = config_or_dims.rope_theta
            axes_dims = [config_or_dims.rope_dim, 48, 48]
            # Reference default: axes_lens=[1024, 512, 512]
            axes_lens = [1024, 512, 512]
        else:
            self.theta = theta
        axes_dims = axes_dims or [32, 48, 48]
        axes_lens = axes_lens or [1024, 512, 512]
        self.ctx = ctx
        self.axes_dims = axes_dims
        self.freqs_cis = self._precompute(axes_dims, axes_lens, self.theta)

    def _precompute(self, axes_dims, axes_lens, theta):
        freqs_cis = []
        for d, e in zip(axes_dims, axes_lens):
            freqs = 1.0 / (theta ** (self.ctx.arange(0, d, 2, dtype=self.ctx.float32()) / d))
            ts = self.ctx.arange(0, e, dtype=self.ctx.float32())
            freqs = self.ctx.einsum("i,j->ij", ts, freqs)
            cos_f = self.ctx.cos(freqs)
            sin_f = self.ctx.sin(freqs)
            freqs_cis_i = self.ctx.stack([cos_f, sin_f], axis=-1)
            freqs_cis.append(freqs_cis_i)
        return freqs_cis

    def forward(self, ids):
        """ids: [N, 3] int32 → freqs_cis [N, total_dim] float32 (cos/sin stacked)。"""
        result = []
        ctx = self.ctx
        for i in range(len(self.axes_dims)):
            index = ctx.cast(ids[:, i], ctx.int32())
            result.append(self.freqs_cis[i][index])
        return self.ctx.concat(result, axis=1)


# =========================================================================
# TimestepEmbedder — 正弦时间步嵌入
# =========================================================================

class TimestepEmbedder:
    """Sinusoidal timestep embedding + MLP. Reference implementation using native functions for numerical consistency."""

    def __init__(self, out_size: int, mid_size: int = 1024,
                 frequency_embedding_size: int = 256, ctx: RuntimeContext = None):
        nn = ctx
        self.ctx = ctx
        self.frequency_embedding_size = frequency_embedding_size
        self.linear1 = nn.Linear(frequency_embedding_size, mid_size, bias=True)
        self.linear2 = nn.Linear(mid_size, out_size, bias=True)

    def forward(self, t):
        import math

        ctx = self.ctx
        half = self.frequency_embedding_size // 2
        dt = ctx.float32()
        freqs = ctx.exp(ctx.mul(ctx.arange(0, half, dtype=dt), (-math.log(10000.0) / half)))
        t_col = ctx.reshape(ctx.cast(t, dt), (-1, 1))
        args = ctx.mul(t_col, ctx.reshape(freqs, (1, -1)))
        emb = ctx.concat([ctx.cos(args), ctx.sin(args)], axis=-1)
        if self.frequency_embedding_size % 2:
            emb = ctx.concat([emb, ctx.zeros_like(emb[:, :1])], axis=-1)
        x = self.linear1(emb)
        x = ctx.silu(x)
        x = self.linear2(x)
        return x


# =========================================================================
# FeedForward — SiLU-gated MLP (SwiGLU variant)
# =========================================================================

class FeedForward:
    """SwiGLU-style FFN: w2(silu(w1(x)) * w3(x)).

    Reference FeedForward: hidden_dim = int(dim / 3 * 8)
    """

    def __init__(self, dim: int, hidden_dim: int, ctx: RuntimeContext):
        nn = ctx
        self.ctx = ctx
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        ctx = self.ctx
        gate = ctx.silu(self.w1(x))
        proj = self.w3(x)
        return self.w2(ctx.mul(gate, proj))


# =========================================================================
# ZImageAttention — QKV 投影 + QK Norm + RoPE + SDPA
# =========================================================================

class ZImageAttention:
    """Z-Image self-attention, separate QK projection + QK Norm + complex RoPE.

    Reference ZImageAttention: dim=3840, n_heads=30.
    """

    def __init__(self, dim: int, n_heads: int, ctx: RuntimeContext,
                 qk_norm: bool = True, eps: float = 1e-5):
        nn = ctx
        self.ctx = ctx
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=False)

        if qk_norm:
            self.norm_q = nn.RMSNorm(self.head_dim, eps=eps)
            self.norm_k = nn.RMSNorm(self.head_dim, eps=eps)
        else:
            self.norm_q = None
            self.norm_k = None

    def forward(self, hidden_states, attention_mask=None, freqs_cis=None):
        ctx = self.ctx
        B, S, _ = hidden_states.shape

        q = self.to_q(hidden_states)
        k = self.to_k(hidden_states)
        v = self.to_v(hidden_states)
        q = ctx.reshape(q, (B, S, self.n_heads, self.head_dim))
        k = ctx.reshape(k, (B, S, self.n_heads, self.head_dim))
        v = ctx.reshape(v, (B, S, self.n_heads, self.head_dim))

        if self.norm_q is not None:
            q = self.norm_q(q)
            k = self.norm_k(k)

        if freqs_cis is not None:
            q = self._apply_rotary(q, freqs_cis)
            k = self._apply_rotary(k, freqs_cis)

        q = ctx.permute(q, (0, 2, 1, 3))
        k = ctx.permute(k, (0, 2, 1, 3))
        v = ctx.permute(v, (0, 2, 1, 3))

        mask = None
        if attention_mask is not None:
            mask = ctx.where(attention_mask[:, None, None, :],
                             ctx.full((1, 1, 1, 1), 0.0, dtype=ctx.float32()),
                             ctx.full((1, 1, 1, 1), float("-inf"), dtype=ctx.float32()))

        out = ctx.attention(q, k, v, scale=self.scale, mask=mask)
        out = ctx.permute(out, (0, 2, 1, 3))
        out = ctx.reshape(out, (B, S, self.dim))
        out = self.to_out(out)
        return out

    def _apply_rotary(self, x, freqs_cis):
        """复数 RoPE: x [B,S,H,D] → 最后一维拆分为 (D/2, 2) → 复数乘法。
        
        freqs_cis: [S, D//2, 2] — 来自 RopeEmbedder（三轴拼接后）。
        """
        ctx = self.ctx
        B, S, H, D = x.shape
        half = D // 2
        x = ctx.reshape(x, (B, S, H, half, 2))
        freqs_cis = ctx.reshape(freqs_cis, (1, S, 1, half, 2))
        x_real, x_imag = x[..., 0], x[..., 1]
        c_real = freqs_cis[..., 0]
        c_imag = freqs_cis[..., 1]
        out_real = x_real * c_real - x_imag * c_imag
        out_imag = x_real * c_imag + x_imag * c_real
        out = ctx.stack([out_real, out_imag], axis=-1)
        return ctx.reshape(out, (B, S, H, D))


# =========================================================================
# ZImageContextBlock — 无 AdaLN，仅用于 context 精炼
# =========================================================================

class ZImageContextBlock:
    """Caption 精炼块：无时间调制，无 gate。"""

    def __init__(self, dim: int, n_heads: int, ctx: RuntimeContext,
                 norm_eps: float = 1e-5, qk_norm: bool = True):
        nn = ctx
        self.ctx = ctx
        self.attention = ZImageAttention(dim, n_heads, ctx, qk_norm=qk_norm, eps=1e-5)
        self.feed_forward = FeedForward(dim, int(dim / 3 * 8), ctx)
        self.attn_norm1 = nn.RMSNorm(dim, eps=norm_eps)
        self.attn_norm2 = nn.RMSNorm(dim, eps=norm_eps)
        self.ffn_norm1 = nn.RMSNorm(dim, eps=norm_eps)
        self.ffn_norm2 = nn.RMSNorm(dim, eps=norm_eps)

    def forward(self, x, attn_mask, freqs_cis):
        # Self-attention
        normed = self.attn_norm1(x)
        attn_out = self.attention.forward(normed, attention_mask=attn_mask, freqs_cis=freqs_cis)
        x = x + self.attn_norm2(attn_out)
        # FFN
        normed = self.ffn_norm1(x)
        ffn_out = self.feed_forward.forward(normed)
        x = x + self.ffn_norm2(ffn_out)
        return x


# =========================================================================
# ZImageTransformerBlock — AdaLN 调制 + Attention + FFN
# =========================================================================

class ZImageTransformerBlock:
    """Main Transformer block: AdaLN time modulation (4 params: scale/gate for attn+ffn).

    Reference ZImageTransformerBlock.
    """

    def __init__(self, dim: int, n_heads: int, ctx: RuntimeContext,
                 norm_eps: float = 1e-5, qk_norm: bool = True):
        nn = ctx
        self.ctx = ctx
        self.attention = ZImageAttention(dim, n_heads, ctx, qk_norm=qk_norm, eps=1e-5)
        self.feed_forward = FeedForward(dim, int(dim / 3 * 8), ctx)
        self.attn_norm1 = nn.RMSNorm(dim, eps=norm_eps)
        self.attn_norm2 = nn.RMSNorm(dim, eps=norm_eps)
        self.ffn_norm1 = nn.RMSNorm(dim, eps=norm_eps)
        self.ffn_norm2 = nn.RMSNorm(dim, eps=norm_eps)
        self.adaLN_modulation = [nn.Linear(min(dim, 256), 4 * dim, bias=True)]

    def forward(self, x, attn_mask, freqs_cis, t_emb):
        ctx = self.ctx
        # AdaLN modulation: [B, 4*dim] → 4 params
        modulation = ctx.reshape(self.adaLN_modulation[0](t_emb), (-1, 1, 4 * self.attention.dim))
        scale_msa, gate_msa, scale_mlp, gate_mlp = self._split4(modulation)
        scale_msa = 1.0 + scale_msa
        scale_mlp = 1.0 + scale_mlp
        gate_msa = self._tanh(gate_msa)
        gate_mlp = self._tanh(gate_mlp)

        # Attention with modulation
        normed = self.attn_norm1(x)
        attn_out = self.attention.forward(normed * scale_msa, attention_mask=attn_mask, freqs_cis=freqs_cis)
        x = x + gate_msa * self.attn_norm2(attn_out)

        # FFN with modulation
        normed = self.ffn_norm1(x)
        ffn_out = self.feed_forward.forward(normed * scale_mlp)
        x = x + gate_mlp * self.ffn_norm2(ffn_out)
        return x

    def _split4(self, x):
        """沿最后一维等分 4 份。"""
        D = x.shape[-1] // 4
        return x[..., :D], x[..., D:2*D], x[..., 2*D:3*D], x[..., 3*D:]

    def _tanh(self, x):
        return self.ctx.tanh(x)


# =========================================================================
# FinalLayer — AdaLN + Linear output
# =========================================================================

class FinalLayer:
    """Output layer: LayerNorm + AdaLN scale + Linear → patch embeddings.

    Reference FinalLayer.
    """

    def __init__(self, hidden_size: int, out_channels: int, ctx: RuntimeContext):
        nn = ctx
        self.ctx = ctx
        self.norm = nn.LayerNorm(hidden_size, eps=1e-6, affine=False)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = [nn.Linear(min(hidden_size, 256), hidden_size, bias=True)]

    def forward(self, x, c):
        ctx = self.ctx
        scale = 1.0 + self.adaLN_modulation[0](ctx.silu(c))
        scale = ctx.reshape(scale, (-1, 1, x.shape[-1]))
        return self.linear(self.norm(x) * scale)


# =========================================================================
# ZImageTransformer — 主模型
# =========================================================================

class ZImageTransformer(TransformerBase):
    """Z-Image / Z-Image-Turbo Transformer。

    VAE latent [C=16, F=1, H, W] → patchify → embed → noise_refiner
    → caption embed → context_refiner → unify → main layers → FinalLayer → unpatchify。
    """

    def __init__(self, config: ZImageConfig, ctx: RuntimeContext):
        self.config = config
        self.ctx = ctx
        nn = ctx
        dim = config.dim
        n_heads = config.num_heads
        norm_eps = config.norm_eps
        qk_norm = config.qk_norm

        self.in_channels = config.in_channels
        self.out_channels = config.in_channels
        self.patch_size = config.patch_size
        self.f_patch_size = 1
        self.dim = dim
        self.n_heads = n_heads
        self.t_scale = config.t_scale

        embed_dim = self.f_patch_size * self.patch_size * self.patch_size * self.in_channels
        self.x_embedder = nn.Linear(embed_dim, dim, bias=True)
        self.final_layer = FinalLayer(dim, embed_dim, ctx)

        self.t_embedder = TimestepEmbedder(out_size=min(dim, 256), mid_size=1024, ctx=ctx)
        self.cap_norm = nn.RMSNorm(config.cap_feat_dim, eps=norm_eps)
        self.cap_embedder = nn.Linear(config.cap_feat_dim, dim, bias=True)
        self.x_pad_token = ctx.zeros((1, dim))
        self.cap_pad_token = ctx.zeros((1, dim))

        self.noise_refiner = [
            ZImageTransformerBlock(dim, n_heads, ctx, norm_eps, qk_norm)
            for _ in range(config.num_refiner_layers)
        ]
        self.context_refiner = [
            ZImageContextBlock(dim, n_heads, ctx, norm_eps, qk_norm)
            for _ in range(config.num_refiner_layers)
        ]
        self.layers = [
            ZImageTransformerBlock(dim, n_heads, ctx, norm_eps, qk_norm)
            for _ in range(config.num_layers)
        ]
        self.rope = RopeEmbedder(config_or_dims=config, ctx=ctx)

        self._param_map: dict[str, Any] = {}
        self._build_param_map()
        # 手动注册非 nn.Module 张量
        self._param_map["x_pad_token"] = self.x_pad_token
        self._param_map["cap_pad_token"] = self.cap_pad_token

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def _build_param_map(self):
        """递归构建 参数路径 → MLX 参数张量 映射。"""
        self._param_map.clear()
        _collect_nn_params(self, "", self._param_map)

    def parameters(self):
        return list(self._param_map.items())

    def forward(self, latents, timestep,
                txt_embeds=None, sigmas=None, **conditioning):
        """前向传播。

        Args:
            latents: [B, C, H, W] 或 [B, C, F, H, W] — VAE latent (C=16)
            timestep: [1] 或标量或整数索引
            txt_embeds: [seq_len, cap_feat_dim] — Qwen3 编码输出
            sigmas: [steps+1] — 调度器 sigma 序列（当 timestep 为整数索引时必需）

        Returns:
            与输入 latents 同形状 — 预测速度场 / 噪声
        """
        ctx = self.ctx

        # 记录输入格式以便恢复
        input_shape = latents.shape
        input_ndim = latents.ndim

        # 使用 cap_feats 作为文本条件（兼容 pipeline 统一接口）
        cap_feats = txt_embeds
        if cap_feats is None:
            cap_feats = conditioning.get("cap_feats")
        if cap_feats is None:
            raise ValueError("ZImageTransformer requires txt_embeds (Qwen3 cap_feats)")
        # 去掉可能的 batch 维度 (1, seq, dim) → (seq, dim)
        if cap_feats.ndim == 3 and cap_feats.shape[0] == 1:
            cap_feats = cap_feats[0]

        # 去掉 batch 维度（模型内部处理 batch=1）
        if latents.ndim == 5 and latents.shape[0] == 1:
            latents = latents[0]  # [C, F, H, W]
        elif latents.ndim == 4 and latents.shape[0] == 1:
            latents = latents[0]  # [C, H, W]
        # 确保有帧维度: [C, F=1, H, W]
        if latents.ndim == 3:
            latents = ctx.reshape(latents, (latents.shape[0], 1, latents.shape[1], latents.shape[2]))

        # Time embedding — reference ZImageTransformer.__call__
        # Integer index timestep needs sigma lookup; scalar is treated as sigma value
        t = timestep
        idx = _coerce_timestep_index(t, ctx)
        if idx is not None:
            if sigmas is None:
                raise ValueError("ZImageTransformer requires sigmas when timestep is an integer index")
            sigma_t = ctx.reshape(sigmas[idx], (1,))
            t = ctx.ones_like(sigma_t) + ctx.mul(sigma_t, -1.0)
        else:
            if not ctx.is_tensor(t):
                t = ctx.array(t, dtype=ctx.float32())
            if hasattr(t, "ndim") and t.ndim == 0:
                t = ctx.reshape(t, (1,))
            # float timestep is already (1 - sigma), do not transform again
        t_emb = self.t_embedder.forward(ctx.mul(ctx.cast(t, ctx.float32()), self.t_scale))

        # Patchify
        x_emb, cap_emb, x_size, x_pos_ids, cap_pos_ids, x_pad_mask, cap_pad_mask = self._patchify(
            image=latents, cap_feats=cap_feats,
        )

        # Image embedding
        x_emb = self.x_embedder(x_emb)
        x_emb = ctx.where(ctx.reshape(x_pad_mask, (-1, 1)), self.x_pad_token, x_emb)
        x_freqs_cis = self.rope.forward(x_pos_ids)
        x_attn_mask = ctx.ones((1, x_emb.shape[0]), dtype=ctx.float32()) > 0
        x_emb = ctx.reshape(x_emb, (1, x_emb.shape[0], x_emb.shape[1]))

        # Noise refiner
        for layer in self.noise_refiner:
            x_emb = layer.forward(x_emb, x_attn_mask, x_freqs_cis, t_emb)

        # Caption embedding
        cap_emb = self.cap_norm(cap_emb)
        cap_emb = self.cap_embedder(cap_emb)
        cap_emb = ctx.where(ctx.reshape(cap_pad_mask, (-1, 1)), self.cap_pad_token, cap_emb)
        cap_freqs_cis = self.rope.forward(cap_pos_ids)
        cap_attn_mask = ctx.ones((1, cap_emb.shape[0]), dtype=ctx.float32()) > 0
        cap_emb = ctx.reshape(cap_emb, (1, cap_emb.shape[0], cap_emb.shape[1]))

        # Context refiner
        for layer in self.context_refiner:
            cap_emb = layer.forward(cap_emb, cap_attn_mask, cap_freqs_cis)

        # Unify
        x_len = x_emb.shape[1]
        unified = ctx.concat([x_emb, cap_emb], axis=1)
        unified_freqs = ctx.concat([x_freqs_cis, cap_freqs_cis], axis=0)
        unified_mask = ctx.ones((1, unified.shape[1]), dtype=ctx.float32()) > 0

        # Main layers
        for layer in self.layers:
            unified = layer.forward(unified, unified_mask, unified_freqs, t_emb)

        # Final layer + unpatchify
        unified = self.final_layer.forward(unified, t_emb)
        output = self._unpatchify(unified[0, :x_len], x_size)

        # 恢复输入格式
        if input_ndim == 4:
            # [C, F, H, W] → [B, C, H, W]（去掉帧维度）
            output = output[:, 0, :, :]  # [C, H, W]
            output = ctx.reshape(output, (1, output.shape[0], output.shape[1], output.shape[2]))
        elif input_ndim == 5:
            output = ctx.reshape(output, input_shape)
        else:
            output = ctx.reshape(output, (1,) + output.shape)
        return -output  # Z-Image 输出取负（速度预测方向）

    # ------------------------------------------------------------------
    # Patchify / Unpatchify
    # ------------------------------------------------------------------

    def _patchify(self, image, cap_feats):
        """图像 → patches + padding；caption → padding to 32 的倍数。"""
        ctx = self.ctx
        pH = pW = self.patch_size
        pF = self.f_patch_size

        # Caption padding
        cap_ori_len = cap_feats.shape[0]
        cap_pad_len = (-cap_ori_len) % 32
        cap_pos_ids = self._coord_grid((cap_ori_len + cap_pad_len, 1, 1), (1, 0, 0))
        cap_pos_ids = ctx.reshape(cap_pos_ids, (-1, 3))
        cap_pad_mask = ctx.concat([
            ctx.zeros((cap_ori_len,), dtype=ctx.float32()) > 0,
            ctx.ones((cap_pad_len,), dtype=ctx.float32()) > 0,
        ], axis=0) if cap_pad_len > 0 else ctx.zeros((cap_ori_len,), dtype=ctx.float32()) > 0

        if cap_pad_len > 0:
            cap_padded = ctx.concat([cap_feats, ctx.repeat(cap_feats[-1:], cap_pad_len, axis=0)], axis=0)
        else:
            cap_padded = cap_feats

        # Image patchification
        C, F, H, W = image.shape
        image_size = (F, H, W)
        F_tok, H_tok, W_tok = F // pF, H // pH, W // pW

        # Reshape to patches: [C, F/pF, pF, H/pH, pH, W/pW, pW]
        img = ctx.reshape(image, (C, F_tok, pF, H_tok, pH, W_tok, pW))
        img = ctx.permute(img, (1, 3, 5, 2, 4, 6, 0))
        img = ctx.reshape(img, (F_tok * H_tok * W_tok, pF * pH * pW * C))

        # Image padding
        img_ori_len = img.shape[0]
        img_pad_len = (-img_ori_len) % 32
        img_pos_ids = self._coord_grid((F_tok, H_tok, W_tok),
                                       (cap_ori_len + cap_pad_len + 1, 0, 0))
        img_pos_ids = ctx.reshape(img_pos_ids, (-1, 3))

        if img_pad_len > 0:
            img_pos_ids = ctx.concat([img_pos_ids, ctx.zeros((img_pad_len, 3), dtype=ctx.int32())], axis=0)
            img = ctx.concat([img, ctx.repeat(img[-1:], img_pad_len, axis=0)], axis=0)

        img_pad_mask = ctx.concat([
            ctx.zeros((img_ori_len,), dtype=ctx.float32()) > 0,
            ctx.ones((img_pad_len,), dtype=ctx.float32()) > 0,
        ], axis=0) if img_pad_len > 0 else ctx.zeros((img_ori_len,), dtype=ctx.float32()) > 0

        return img, cap_padded, image_size, img_pos_ids, cap_pos_ids, img_pad_mask, cap_pad_mask

    def _unpatchify(self, x, size):
        """Patches → image: [F_tok*H_tok*W_tok, out_channels] → [out_channels, F, H, W]。"""
        ctx = self.ctx
        pH = pW = self.patch_size
        pF = self.f_patch_size
        F, H, W = size
        ori_len = (F // pF) * (H // pH) * (W // pW)
        x = x[:ori_len]
        x = ctx.reshape(x, (F // pF, H // pH, W // pW, pF, pH, pW, self.out_channels))
        x = ctx.permute(x, (6, 0, 3, 1, 4, 2, 5))
        return ctx.reshape(x, (self.out_channels, F, H, W))

    def _coord_grid(self, size, start=None):
        """创建坐标网格 [(z0,y0,x0), (z0,y0,x1), ...]。"""
        ctx = self.ctx
        start = start or tuple(0 for _ in size)
        axes = [ctx.arange(x0, x0 + span, dtype=ctx.int32()) for x0, span in zip(start, size)]
        grids = ctx.meshgrid(*axes)
        return ctx.stack(grids, axis=-1)


def _collect_nn_params(obj, prefix: str, result: dict):
    """递归收集子模块 ``parameters()`` / 属性树中的叶子参数到 ``result``。"""
    # 检查当前对象是否有 parameters()
    if hasattr(obj, 'parameters') and callable(obj.parameters):
        try:
            params = obj.parameters()
            if isinstance(params, dict):
                for pname, ptensor in params.items():
                    full_key = f"{prefix}.{pname}" if prefix else pname
                    result[full_key] = ptensor
                return  # parameters() 已返回所有叶子参数
        except Exception:
            pass

    # 遍历属性
    for attr_name in sorted(dir(obj)):
        if attr_name.startswith('_') or attr_name in ('ctx', 'config', 'freqs_cis', '_param_map'):
            continue
        try:
            attr = getattr(obj, attr_name)
        except Exception:
            continue
        if attr is None or isinstance(attr, (int, float, str, bool, type)):
            continue

        new_prefix = f"{prefix}.{attr_name}" if prefix else attr_name

        if hasattr(attr, 'parameters') and callable(attr.parameters):
            _collect_nn_params(attr, new_prefix, result)
        elif isinstance(attr, (list, tuple)):
            for i, item in enumerate(attr):
                _collect_nn_params(item, f"{new_prefix}.{i}", result)
        elif hasattr(attr, '__dict__') and not isinstance(attr, type):
            _collect_nn_params(attr, new_prefix, result)
