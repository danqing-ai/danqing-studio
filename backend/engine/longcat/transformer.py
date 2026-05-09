"""
LongCat-Image Transformer — MM-DiT + Qwen2.5-VL conditioning。

权重键名 100% 匹配 diffusers LongCatImageTransformer2DModel (0.35.1)。

架构要点：
- VAE latent: 16 channels → 2×2 patchify → 64-dim tokens
- 3D MRoPE: axes_dims=[16, 56, 56], theta=10000
- Joint blocks: 10 × MM-DiT (img+txt dual-stream with AdaLN + RoPE)
- Single blocks: 20 × self-attention with AdaLN + RoPE
- Time embed: 2-layer MLP (256 → 3072 → 3072)
- Text encoder: Qwen2.5-VL (3584-dim output)
"""
from __future__ import annotations

from typing import Any

import mlx.core as mx

from backend.engine.config.model_configs import LongCatConfig
from backend.engine.runtime._base import RuntimeContext
from backend.engine.common._base import TransformerBase
from backend.engine.common.attention import _apply_rope


# ------------------------------------------------------------------
# 3D MRoPE (axes_dims=[16, 56, 56], theta=10000)
# ------------------------------------------------------------------

class LongCatRoPE:
    """3D Multimodal Rotary Position Embedding.
    
    每个 token 的 3D 位置 ID: [modality, height, width]。
    - Text tokens:  modality=0, x=y=position
    - Image tokens: modality=1, x=row, y=col
    
    Theta=10000 (标准 RoPE theta)。
    axes_dims=[16, 56, 56], sum=128 = head_dim。
    """
    def __init__(self, ctx: RuntimeContext, theta: float = 10000.0):
        self.ctx = ctx
        self.theta = theta
        self.axes_dims = [16, 56, 56]

    def forward(self, ids):
        """ids: [N, 3] int32 position IDs per token.
        Returns: (cos, sin) each [1, 1, N, R] where R = sum(axes_dims)//2 = 64
        """
        ctx = self.ctx
        cos_list, sin_list = [], []
        for i, dim in enumerate(self.axes_dims):
            pos = ids[:, i].astype(ctx.float32())
            half = dim // 2
            dtype_f32 = ctx.float32()
            freqs = 1.0 / (self.theta ** (ctx.arange(0, dim, 2, dtype=dtype_f32) / dim))
            args = pos[:, None] * freqs[None, :]
            cos_list.append(ctx.cos(args))
            sin_list.append(ctx.sin(args))
        cos = ctx.concat(cos_list, axis=-1)
        sin = ctx.concat(sin_list, axis=-1)
        cos = cos.reshape(1, 1, -1, cos.shape[-1])
        sin = sin.reshape(1, 1, -1, sin.shape[-1])
        return cos, sin


# ------------------------------------------------------------------
# Time Embedding (匹配 diffusers 键名)
# ------------------------------------------------------------------

class LongCatTimestepEmbedder:
    """正弦时间步嵌入 → 2层 MLP。
    
    权重键: time_embed.timestep_embedder.linear_{1,2}
    """
    def __init__(self, dim: int, ctx: RuntimeContext, frequency_embedding_size: int = 256):
        self.ctx = ctx
        nn = ctx
        self.frequency_embedding_size = frequency_embedding_size
        self.linear_1 = nn.Linear(frequency_embedding_size, dim, bias=True)
        self.linear_2 = nn.Linear(dim, dim, bias=True)

    def forward(self, timesteps):
        ctx = self.ctx
        if not isinstance(timesteps, mx.array):
            timesteps = mx.array([timesteps], dtype=mx.float32)
        elif timesteps.ndim == 0:
            timesteps = mx.reshape(timesteps, (1,))

        half = self.frequency_embedding_size // 2
        freqs = ctx.exp(
            -ctx.log(ctx.full((half,), 10000.0))
            * ctx.arange(0, half, 1, dtype=mx.float32) / half
        )
        args = ctx.reshape(timesteps, (-1, 1)) * ctx.reshape(freqs, (1, -1))
        embedding = ctx.concat([ctx.cos(args), ctx.sin(args)], axis=-1)
        x = self.linear_1(embedding)
        x = ctx.silu(x)
        return self.linear_2(x)


# ------------------------------------------------------------------
# Normalization (elementwise_affine=False, 无权重)
# ------------------------------------------------------------------

class _LayerNormNoParams:
    def __init__(self, dim: int, ctx: RuntimeContext, eps: float = 1e-6):
        self.dim = dim
        self.eps = eps
        self.ctx = ctx

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        mean = mx.mean(x, axis=-1, keepdims=True)
        var = mx.mean((x - mean) ** 2, axis=-1, keepdims=True)
        return (x - mean) / mx.sqrt(var + self.eps)


# ------------------------------------------------------------------
# Joint Attention (MM-DiT dual-stream + RoPE)
# ------------------------------------------------------------------

class LongCatJointAttention:
    """双流注意力: img QKV + txt add_QKV, 各自输出, 带 RoPE。"""
    def __init__(self, dim: int, heads: int, ctx: RuntimeContext):
        nn = ctx; self.ctx = ctx; self.heads = heads
        self.dim_head = dim // heads; self.scale = self.dim_head ** -0.5
        self.dim = dim

        self.to_q = nn.Linear(dim, dim, bias=True)
        self.to_k = nn.Linear(dim, dim, bias=True)
        self.to_v = nn.Linear(dim, dim, bias=True)
        self.to_out = nn.Linear(dim, dim, bias=True)

        self.add_q_proj = nn.Linear(dim, dim, bias=True)
        self.add_k_proj = nn.Linear(dim, dim, bias=True)
        self.add_v_proj = nn.Linear(dim, dim, bias=True)
        self.to_add_out = nn.Linear(dim, dim, bias=True)

        self.norm_q = nn.RMSNorm(self.dim_head, eps=1e-6)
        self.norm_k = nn.RMSNorm(self.dim_head, eps=1e-6)
        self.norm_added_q = nn.RMSNorm(self.dim_head, eps=1e-6)
        self.norm_added_k = nn.RMSNorm(self.dim_head, eps=1e-6)

    def forward(self, hidden_states, encoder_hidden_states,
                img_cos=None, img_sin=None, txt_cos=None, txt_sin=None):
        ctx = self.ctx
        B, S_img, _ = hidden_states.shape
        S_txt = encoder_hidden_states.shape[1]

        q = ctx.reshape(self.to_q(hidden_states), (B, S_img, self.heads, self.dim_head))
        k = ctx.reshape(self.to_k(hidden_states), (B, S_img, self.heads, self.dim_head))
        v = ctx.reshape(self.to_v(hidden_states), (B, S_img, self.heads, self.dim_head))
        q = self.norm_q(q); k = self.norm_k(k)

        q_txt = ctx.reshape(self.add_q_proj(encoder_hidden_states), (B, S_txt, self.heads, self.dim_head))
        k_txt = ctx.reshape(self.add_k_proj(encoder_hidden_states), (B, S_txt, self.heads, self.dim_head))
        v_txt = ctx.reshape(self.add_v_proj(encoder_hidden_states), (B, S_txt, self.heads, self.dim_head))
        q_txt = self.norm_added_q(q_txt); k_txt = self.norm_added_k(k_txt)

        q = ctx.permute(q, (0, 2, 1, 3))
        k = ctx.permute(k, (0, 2, 1, 3))
        v = ctx.permute(v, (0, 2, 1, 3))
        q_txt = ctx.permute(q_txt, (0, 2, 1, 3))
        k_txt = ctx.permute(k_txt, (0, 2, 1, 3))
        v_txt = ctx.permute(v_txt, (0, 2, 1, 3))

        if img_cos is not None:
            q = _apply_rope(ctx, q, img_cos, img_sin)
            k = _apply_rope(ctx, k, img_cos, img_sin)
        if txt_cos is not None:
            q_txt = _apply_rope(ctx, q_txt, txt_cos, txt_sin)
            k_txt = _apply_rope(ctx, k_txt, txt_cos, txt_sin)

        # Joint attention: Concatenate txt+img along sequence dimension
        # (Diffusers FluxAttnProcessor: cat([encoder_query, query], dim=1))
        q_joint = ctx.concat([q_txt, q], axis=2)
        k_joint = ctx.concat([k_txt, k], axis=2)
        v_joint = ctx.concat([v_txt, v], axis=2)

        attn_out = ctx.attention(q_joint, k_joint, v_joint, scale=self.scale)
        attn_out = ctx.permute(attn_out, (0, 2, 1, 3))

        # Split: first S_txt = text, rest = image
        txt_out = attn_out[:, :S_txt, :, :].reshape(B, S_txt, self.dim)
        img_out = attn_out[:, S_txt:, :, :].reshape(B, S_img, self.dim)

        return self.to_out(img_out), self.to_add_out(txt_out)


# ------------------------------------------------------------------
# Single Attention (concatenated img+txt + RoPE)
# ------------------------------------------------------------------

class LongCatSingleAttention:
    """单流注意力: 拼接 img+txt, 带 RoPE。"""
    def __init__(self, dim: int, heads: int, ctx: RuntimeContext):
        nn = ctx; self.ctx = ctx; self.heads = heads
        self.dim_head = dim // heads; self.scale = self.dim_head ** -0.5
        self.dim = dim

        self.to_q = nn.Linear(dim, dim, bias=True)
        self.to_k = nn.Linear(dim, dim, bias=True)
        self.to_v = nn.Linear(dim, dim, bias=True)
        self.norm_q = nn.RMSNorm(self.dim_head, eps=1e-6)
        self.norm_k = nn.RMSNorm(self.dim_head, eps=1e-6)

    def forward(self, hidden_states, cos=None, sin=None):
        ctx = self.ctx
        B, S, _ = hidden_states.shape

        q = ctx.reshape(self.to_q(hidden_states), (B, S, self.heads, self.dim_head))
        k = ctx.reshape(self.to_k(hidden_states), (B, S, self.heads, self.dim_head))
        v = ctx.reshape(self.to_v(hidden_states), (B, S, self.heads, self.dim_head))
        q = self.norm_q(q); k = self.norm_k(k)

        q = ctx.permute(q, (0, 2, 1, 3))
        k = ctx.permute(k, (0, 2, 1, 3))
        v = ctx.permute(v, (0, 2, 1, 3))

        if cos is not None:
            q = _apply_rope(ctx, q, cos, sin)
            k = _apply_rope(ctx, k, cos, sin)

        out = ctx.attention(q, k, v, scale=self.scale)
        out = ctx.permute(out, (0, 2, 1, 3))
        out = ctx.reshape(out, (B, S, self.dim))
        return out


# ------------------------------------------------------------------
# Feed Forward (GELU MLP)
# ------------------------------------------------------------------

class LongCatFeedForward:
    """GELU MLP: 线性 → GELU → 线性。
    
    权重键: net.0.proj / net.2 (diffusers 命名)。
    """
    def __init__(self, dim: int, ctx: RuntimeContext, mult: int = 4):
        nn = ctx; self.ctx = ctx
        hidden_dim = int(dim * mult)
        self.net_0_proj = nn.Linear(dim, hidden_dim, bias=True)
        self.net_2 = nn.Linear(hidden_dim, dim, bias=True)

    def forward(self, x):
        ctx = self.ctx
        x = self.net_0_proj(x)
        x = ctx.gelu(x)
        return self.net_2(x)


# ------------------------------------------------------------------
# AdaLN Modulation
# ------------------------------------------------------------------

class _AdaLNModulation:
    """AdaLN 调制线性层。权重键: norm1.linear / norm1_context.linear。"""
    def __init__(self, dim: int, num_params: int, ctx: RuntimeContext):
        self.linear = ctx.Linear(dim, dim * num_params, bias=True)

    def __call__(self, c):
        return self.linear(c)

    def forward(self, c):
        return self.linear(c)


# ------------------------------------------------------------------
# Joint Transformer Block
# ------------------------------------------------------------------

class LongCatJointBlock:
    """MM-DiT Joint Block — img+txt 双流，共享时间调制。"""
    def __init__(self, dim: int, heads: int, ctx: RuntimeContext):
        nn = ctx; self.ctx = ctx; self.dim = dim

        self.norm1 = _AdaLNModulation(dim, 6, ctx)
        self.norm1_context = _AdaLNModulation(dim, 6, ctx)

        self.attn = LongCatJointAttention(dim, heads, ctx)
        self.ff = LongCatFeedForward(dim, ctx, mult=4)
        self.ff_context = LongCatFeedForward(dim, ctx, mult=4)

        self.norm2 = _LayerNormNoParams(dim, ctx, eps=1e-6)
        self.norm2_context = _LayerNormNoParams(dim, ctx, eps=1e-6)

    def forward(self, hidden_states, encoder_hidden_states, img_mod, txt_mod,
                img_cos=None, img_sin=None, txt_cos=None, txt_sin=None):
        B = hidden_states.shape[0]

        (s_msa_i, sc_msa_i, g_msa_i, s_mlp_i, sc_mlp_i, g_mlp_i) = img_mod
        (s_msa_t, sc_msa_t, g_msa_t, s_mlp_t, sc_mlp_t, g_mlp_t) = txt_mod

        n_img = self.norm2(hidden_states)
        n_img = (1 + sc_msa_i[:, None, :]) * n_img + s_msa_i[:, None, :]
        n_txt = self.norm2_context(encoder_hidden_states)
        n_txt = (1 + sc_msa_t[:, None, :]) * n_txt + s_msa_t[:, None, :]

        img_out, txt_out = self.attn.forward(n_img, n_txt, img_cos, img_sin, txt_cos, txt_sin)
        hidden_states = hidden_states + g_msa_i[:, None, :] * img_out
        encoder_hidden_states = encoder_hidden_states + g_msa_t[:, None, :] * txt_out

        n_img = self.norm2(hidden_states)
        n_img = (1 + sc_mlp_i[:, None, :]) * n_img + s_mlp_i[:, None, :]
        n_txt = self.norm2_context(encoder_hidden_states)
        n_txt = (1 + sc_mlp_t[:, None, :]) * n_txt + s_mlp_t[:, None, :]

        hidden_states = hidden_states + g_mlp_i[:, None, :] * self.ff.forward(n_img)
        encoder_hidden_states = encoder_hidden_states + g_mlp_t[:, None, :] * self.ff_context.forward(n_txt)

        return encoder_hidden_states, hidden_states


# ------------------------------------------------------------------
# Single Transformer Block
# ------------------------------------------------------------------

class LongCatSingleBlock:
    """Single-stream block: 自注意力 + MLP，img+txt 拼接后处理。"""
    def __init__(self, dim: int, heads: int, ctx: RuntimeContext):
        nn = ctx; self.ctx = ctx; self.dim = dim

        self.norm = _AdaLNModulation(dim, 3, ctx)
        self.attn = LongCatSingleAttention(dim, heads, ctx)
        self.proj_mlp = nn.Linear(dim, int(dim * 4), bias=True)
        self.proj_out = nn.Linear(int(dim * 4) + dim, dim, bias=True)
        self.norm2 = _LayerNormNoParams(dim, ctx, eps=1e-6)

    def forward(self, x, mod_params, cos=None, sin=None):
        ctx = self.ctx
        shift, scale, gate = mod_params

        n = self.norm2(x)
        n = (1 + scale[:, None, :]) * n + shift[:, None, :]
        attn_out = self.attn.forward(n, cos, sin)

        mlp_out = self.proj_mlp(n)
        combined = ctx.concat([attn_out, mlp_out], axis=-1)
        out = self.proj_out(combined)

        return x + gate[:, None, :] * out


# ------------------------------------------------------------------
# Output Layer
# ------------------------------------------------------------------

class LongCatAdaLNOutput:
    """Final AdaLN: scale + shift，然后投影。"""
    def __init__(self, dim: int, ctx: RuntimeContext):
        self.ctx = ctx
        self.linear = ctx.Linear(dim, dim * 2, bias=True)

    def forward(self, x, c):
        ctx = self.ctx
        v = self.linear(c)
        D = v.shape[-1] // 2
        scale = v[..., :D]
        shift = v[..., D:]
        return x * (1 + scale[:, None, :]) + shift[:, None, :]


# ------------------------------------------------------------------
# LongCat Transformer
# ------------------------------------------------------------------

class LongCatTransformer(TransformerBase):
    """LongCat-Image MM-DiT — 权重键 100% 匹配 diffusers。
    
    注意：接收的 latents 应为已 pack 的 [B, H//2 * W//2, 64]，
    这是 diffusers LongCatImagePipeline 的标准输入格式。
    """

    def __init__(self, config: LongCatConfig, ctx: RuntimeContext):
        self.config = config; self.ctx = ctx; nn = ctx
        dim = config.hidden_dim
        heads = config.num_heads
        n_joint = config.num_joint_layers
        n_single = config.num_single_layers

        self.x_embedder = nn.Linear(64, dim, bias=True)
        self.context_embedder = nn.Linear(config.text_dim, dim, bias=True)
        self.time_embed = LongCatTimestepEmbedder(dim, ctx)
        self.rope = LongCatRoPE(ctx)

        self.transformer_blocks = [
            LongCatJointBlock(dim, heads, ctx) for _ in range(n_joint)
        ]
        self.single_transformer_blocks = [
            LongCatSingleBlock(dim, heads, ctx) for _ in range(n_single)
        ]

        self.norm_out = LongCatAdaLNOutput(dim, ctx)
        self.proj_out = nn.Linear(dim, 64, bias=True)

        self._build_param_map()

    def forward(self, latents, timestep, txt_embeds=None, sigmas=None, **cond):
        ctx = self.ctx
        B = latents.shape[0]
        H, W = latents.shape[2], latents.shape[3]

        if sigmas is not None:
            t_idx = int(timestep)
            sigma_t = sigmas[t_idx] if t_idx < len(sigmas) else sigmas[-1] if len(sigmas) > 0 else 1.0
            t_val = float(sigma_t) * 1000.0
        else:
            t_val = float(timestep) * 1000.0

        x = self._patchify(latents)
        hidden_states = self.x_embedder(x)

        if txt_embeds is not None:
            encoder_hidden_states = self.context_embedder(txt_embeds)
            txt_len = txt_embeds.shape[1]
        else:
            encoder_hidden_states = ctx.zeros((B, 256, self.config.hidden_dim))
            txt_len = 256

        c = self.time_embed.forward(t_val)

        img_ids, txt_ids = self._gen_pos_ids(H // 2, W // 2, txt_len)
        txt_cos, txt_sin = self.rope.forward(txt_ids)
        img_cos, img_sin = self.rope.forward(img_ids)

        for block in self.transformer_blocks:
            img_mod = self._split_modulation(block.norm1(c), 6)
            txt_mod = self._split_modulation(block.norm1_context(c), 6)
            encoder_hidden_states, hidden_states = block.forward(
                hidden_states, encoder_hidden_states, img_mod, txt_mod,
                img_cos=img_cos, img_sin=img_sin,
                txt_cos=txt_cos, txt_sin=txt_sin)

        x = ctx.concat([encoder_hidden_states, hidden_states], axis=1)
        all_ids = ctx.concat([txt_ids, img_ids], axis=0)
        all_cos, all_sin = self.rope.forward(all_ids)
        for block in self.single_transformer_blocks:
            mod = self._split_modulation(block.norm(c), 3)
            x = block.forward(x, mod, cos=all_cos, sin=all_sin)

        encoder_hidden_states = x[:, :txt_len]
        hidden_states = x[:, txt_len:]

        hidden_states = self.norm_out.forward(hidden_states, c)
        hidden_states = self.proj_out(hidden_states)

        output = self._unpatchify(hidden_states, H, W)
        return output

    def _patchify(self, latents):
        ctx = self.ctx
        B, C, H, W = latents.shape
        ps = 2
        x = ctx.reshape(latents, (B, C, H // ps, ps, W // ps, ps))
        x = ctx.permute(x, (0, 2, 4, 1, 3, 5))
        x = ctx.reshape(x, (B, (H // ps) * (W // ps), C * ps * ps))
        return x

    def _unpatchify(self, x, H, W):
        ctx = self.ctx
        B = x.shape[0]
        ps = 2
        C = 16
        x = ctx.reshape(x, (B, H // ps, W // ps, C, ps, ps))
        x = ctx.permute(x, (0, 3, 1, 4, 2, 5))
        x = ctx.reshape(x, (B, C, H, W))
        return x

    def _gen_pos_ids(self, h2, w2, txt_len):
        ctx = self.ctx
        img_h = mx.arange(0, h2, 1, dtype=mx.int32)
        img_w = mx.arange(0, w2, 1, dtype=mx.int32)
        h_grid = mx.reshape(mx.broadcast_to(img_h[:, None], (h2, w2)), (-1,))
        w_grid = mx.reshape(mx.broadcast_to(img_w[None, :], (h2, w2)), (-1,))
        ones = mx.ones(h2 * w2, dtype=mx.int32)
        img_ids = mx.stack([ones, h_grid, w_grid], axis=1)
        txt_pos = mx.arange(0, txt_len, 1, dtype=mx.int32)
        zeros = mx.zeros(txt_len, dtype=mx.int32)
        txt_ids = mx.stack([zeros, txt_pos, txt_pos], axis=1)
        return img_ids, txt_ids

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _split_modulation(self, modulation, num_params):
        """Split [B, num_params*dim] into num_params × [B, dim]。"""
        ctx = self.ctx
        B = modulation.shape[0]
        dim = self.config.hidden_dim
        modulation = modulation.reshape(B, num_params, dim)
        return [modulation[:, i, :] for i in range(num_params)]
