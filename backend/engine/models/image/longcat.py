"""
LongCat-Image Transformer — MM-DiT + Qwen2.5-VL conditioning。

权重键名 100% 匹配 diffusers LongCatImageTransformer2DModel (0.35.1)。

架构要点：
- VAE latent: 16 channels → 2×2 patchify → 64-dim tokens
- Joint blocks: 10 × MM-DiT (img+txt dual-stream with AdaLN)
- Single blocks: 20 × self-attention with AdaLN
- Time embed: 2-layer MLP (256 → 3072 → 3072)
- Text encoder: Qwen2.5-VL (3584-dim output)
- Scheduler: FlowMatchEulerDiscrete (shift=3.0, dynamic)
"""
from __future__ import annotations

from typing import Any

from backend.engine.config.model_configs import LongCatConfig
from backend.engine.runtime._base import RuntimeContext
from backend.engine.models._base import TransformerBase


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
        # 匹配 diffusers 键名: timestep_embedder.linear_1 / linear_2
        self.linear_1 = nn.Linear(frequency_embedding_size, dim, bias=True)
        self.linear_2 = nn.Linear(dim, dim, bias=True)

    def forward(self, timesteps):
        ctx = self.ctx
        import mlx.core as mx
        
        # Ensure timesteps is an MLX array
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
# AdaLN (elementwise_affine=False, 无权重)
# ------------------------------------------------------------------

class _LayerNormNoParams:
    """LayerNorm without learnable params (elementwise_affine=False)。"""
    def __init__(self, dim: int, ctx: RuntimeContext, eps: float = 1e-6):
        self.dim = dim
        self.eps = eps
        self.ctx = ctx

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        import mlx.core as mx
        mean = mx.mean(x, axis=-1, keepdims=True)
        var = mx.mean((x - mean) ** 2, axis=-1, keepdims=True)
        return (x - mean) / mx.sqrt(var + self.eps)


class _RMSNormNoParams:
    """RMSNorm without learnable params。"""
    def __init__(self, dim: int, ctx: RuntimeContext, eps: float = 1e-6):
        self.eps = eps
        self.ctx = ctx

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        import mlx.core as mx
        var = mx.mean(x ** 2, axis=-1, keepdims=True)
        return x / mx.sqrt(var + self.eps)


# ------------------------------------------------------------------
# Joint Transformer Block (MM-DiT)
# ------------------------------------------------------------------

class LongCatJointAttention:
    """双流注意力: img QKV + txt add_QKV, 各自输出。"""
    def __init__(self, dim: int, heads: int, ctx: RuntimeContext):
        nn = ctx; self.ctx = ctx; self.heads = heads
        self.dim_head = dim // heads; self.scale = self.dim_head ** -0.5
        self.dim = dim

        # Image stream
        self.to_q = nn.Linear(dim, dim, bias=True)
        self.to_k = nn.Linear(dim, dim, bias=True)
        self.to_v = nn.Linear(dim, dim, bias=True)
        self.norm_q = nn.RMSNorm(self.dim_head, eps=1e-6)
        self.norm_k = nn.RMSNorm(self.dim_head, eps=1e-6)
        self.to_out = nn.Linear(dim, dim, bias=True)

        # Text stream
        self.add_q_proj = nn.Linear(dim, dim, bias=True)
        self.add_k_proj = nn.Linear(dim, dim, bias=True)
        self.add_v_proj = nn.Linear(dim, dim, bias=True)
        self.to_add_out = nn.Linear(dim, dim, bias=True)

        # QK Norm (RMSNorm with params)
        self.norm_q = nn.RMSNorm(self.dim_head, eps=1e-6)
        self.norm_k = nn.RMSNorm(self.dim_head, eps=1e-6)
        self.norm_added_q = nn.RMSNorm(self.dim_head, eps=1e-6)
        self.norm_added_k = nn.RMSNorm(self.dim_head, eps=1e-6)

    def forward(self, hidden_states, encoder_hidden_states):
        ctx = self.ctx
        B, S_img, _ = hidden_states.shape
        S_txt = encoder_hidden_states.shape[1]

        # Image QKV
        q = ctx.reshape(self.to_q(hidden_states), (B, S_img, self.heads, self.dim_head))
        k = ctx.reshape(self.to_k(hidden_states), (B, S_img, self.heads, self.dim_head))
        v = ctx.reshape(self.to_v(hidden_states), (B, S_img, self.heads, self.dim_head))
        q = self.norm_q(q); k = self.norm_k(k)

        # Text QKV
        q_txt = ctx.reshape(self.add_q_proj(encoder_hidden_states), (B, S_txt, self.heads, self.dim_head))
        k_txt = ctx.reshape(self.add_k_proj(encoder_hidden_states), (B, S_txt, self.heads, self.dim_head))
        v_txt = ctx.reshape(self.add_v_proj(encoder_hidden_states), (B, S_txt, self.heads, self.dim_head))
        q_txt = self.norm_added_q(q_txt); k_txt = self.norm_added_k(k_txt)

        # Permute for attention
        q = ctx.permute(q, (0, 2, 1, 3)); k = ctx.permute(k, (0, 2, 1, 3)); v = ctx.permute(v, (0, 2, 1, 3))
        q_txt = ctx.permute(q_txt, (0, 2, 1, 3)); k_txt = ctx.permute(k_txt, (0, 2, 1, 3)); v_txt = ctx.permute(v_txt, (0, 2, 1, 3))

        # Self-attention (img attends to img, txt attends to txt)
        img_out = ctx.attention(q, k, v, scale=self.scale)
        img_out = ctx.permute(img_out, (0, 2, 1, 3))
        img_out = ctx.reshape(img_out, (B, S_img, self.dim))

        txt_out = ctx.attention(q_txt, k_txt, v_txt, scale=self.scale)
        txt_out = ctx.permute(txt_out, (0, 2, 1, 3))
        txt_out = ctx.reshape(txt_out, (B, S_txt, self.dim))

        return self.to_out(img_out), self.to_add_out(txt_out)


class LongCatSingleAttention:
    """单流注意力: 只有 img QKV，用于 single blocks。"""
    def __init__(self, dim: int, heads: int, ctx: RuntimeContext):
        nn = ctx; self.ctx = ctx; self.heads = heads
        self.dim_head = dim // heads; self.scale = self.dim_head ** -0.5
        self.dim = dim

        # Only image stream (no to_out projection in single blocks)
        self.to_q = nn.Linear(dim, dim, bias=True)
        self.to_k = nn.Linear(dim, dim, bias=True)
        self.to_v = nn.Linear(dim, dim, bias=True)
        self.norm_q = nn.RMSNorm(self.dim_head, eps=1e-6)
        self.norm_k = nn.RMSNorm(self.dim_head, eps=1e-6)

    def forward(self, hidden_states, _encoder_hidden_states=None):
        """Single attention ignores encoder_hidden_states (for API compatibility)."""
        ctx = self.ctx
        B, S, _ = hidden_states.shape

        q = ctx.reshape(self.to_q(hidden_states), (B, S, self.heads, self.dim_head))
        k = ctx.reshape(self.to_k(hidden_states), (B, S, self.heads, self.dim_head))
        v = ctx.reshape(self.to_v(hidden_states), (B, S, self.heads, self.dim_head))
        q = self.norm_q(q); k = self.norm_k(k)

        q = ctx.permute(q, (0, 2, 1, 3))
        k = ctx.permute(k, (0, 2, 1, 3))
        v = ctx.permute(v, (0, 2, 1, 3))

        out = ctx.attention(q, k, v, scale=self.scale)
        out = ctx.permute(out, (0, 2, 1, 3))
        out = ctx.reshape(out, (B, S, self.dim))

        return out


class LongCatFeedForward:
    """GELU MLP: 线性 → GELU → 线性。
    
    权重键: net.0.proj / net.2 (diffusers 命名)。
    diffusers 的 'proj' 是第一个 Linear 的包装属性名。
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


class _AdaLNModulation:
    """AdaLN 调制线性层。权重键: norm1.linear / norm1_context.linear。"""
    def __init__(self, dim: int, num_params: int, ctx: RuntimeContext):
        self.linear = ctx.Linear(dim, dim * num_params, bias=True)

    def __call__(self, c):
        return self.linear(c)

    def forward(self, c):
        return self.linear(c)


class LongCatJointBlock:
    """MM-DiT Joint Block — img+txt 双流，共享时间调制。"""
    def __init__(self, dim: int, heads: int, ctx: RuntimeContext):
        nn = ctx; self.ctx = ctx; self.dim = dim

        # AdaLN: 6*dim = 2 sets × (shift + scale + gate)
        # 权重键: norm1.linear / norm1_context.linear
        self.norm1 = _AdaLNModulation(dim, 6, ctx)
        self.norm1_context = _AdaLNModulation(dim, 6, ctx)

        self.attn = LongCatJointAttention(dim, heads, ctx)
        self.ff = LongCatFeedForward(dim, ctx, mult=4)
        self.ff_context = LongCatFeedForward(dim, ctx, mult=4)

        # 无参数的 LayerNorm (elementwise_affine=False)
        self.norm2 = _LayerNormNoParams(dim, ctx, eps=1e-6)
        self.norm2_context = _LayerNormNoParams(dim, ctx, eps=1e-6)

    def forward(self, hidden_states, encoder_hidden_states, img_mod, txt_mod):
        B = hidden_states.shape[0]

        # img_mod / txt_mod: (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)
        # 每个都是 [B, dim]，需要 broadcast 到 [B, 1, dim]
        (s_msa_i, sc_msa_i, g_msa_i, s_mlp_i, sc_mlp_i, g_mlp_i) = img_mod
        (s_msa_t, sc_msa_t, g_msa_t, s_mlp_t, sc_mlp_t, g_mlp_t) = txt_mod

        # --- Attention path ---
        n_img = self.norm2(hidden_states)
        n_img = (1 + sc_msa_i[:, None, :]) * n_img + s_msa_i[:, None, :]
        n_txt = self.norm2_context(encoder_hidden_states)
        n_txt = (1 + sc_msa_t[:, None, :]) * n_txt + s_msa_t[:, None, :]

        img_out, txt_out = self.attn.forward(n_img, n_txt)
        hidden_states = hidden_states + g_msa_i[:, None, :] * img_out
        encoder_hidden_states = encoder_hidden_states + g_msa_t[:, None, :] * txt_out

        # --- FFN path ---
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

        # AdaLN: 3*dim = shift + scale + gate
        # 权重键: norm.linear
        self.norm = _AdaLNModulation(dim, 3, ctx)

        # Self-attention (single-stream, only img QKV)
        self.attn = LongCatSingleAttention(dim, heads, ctx)

        # MLP
        self.proj_mlp = nn.Linear(dim, int(dim * 4), bias=True)
        self.proj_out = nn.Linear(int(dim * 4) + dim, dim, bias=True)

        # No-param norm
        self.norm2 = _LayerNormNoParams(dim, ctx, eps=1e-6)

    def forward(self, x, mod_params):
        ctx = self.ctx
        shift, scale, gate = mod_params

        # Attention
        n = self.norm2(x)
        n = (1 + scale[:, None, :]) * n + shift[:, None, :]
        attn_out = self.attn.forward(n)  # single-stream self-attention

        # MLP
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
        v = self.linear(c)  # [B, 2*dim]
        D = v.shape[-1] // 2
        scale = v[..., :D]
        shift = v[..., D:]
        return x * (1 + scale[:, None, :]) + shift[:, None, :]


# ------------------------------------------------------------------
# LongCat Transformer
# ------------------------------------------------------------------

class LongCatTransformer(TransformerBase):
    """LongCat-Image MM-DiT — 权重键 100% 匹配 diffusers。"""

    def __init__(self, config: LongCatConfig, ctx: RuntimeContext):
        self.config = config; self.ctx = ctx; nn = ctx
        dim = config.hidden_dim  # 3072
        heads = config.num_heads  # 24
        n_joint = config.num_joint_layers  # 10
        n_single = config.num_single_layers  # 20

        # Patchify: VAE latent [B, 16, H, W] → unfold 2×2 → [B, 64, H//2, W//2]
        # x_embedder 是 Linear(64, 3072)，patchify 在 forward 中手动完成
        self.x_embedder = nn.Linear(64, dim, bias=True)
        self.context_embedder = nn.Linear(config.text_dim, dim, bias=True)
        self.time_embed = LongCatTimestepEmbedder(dim, ctx)

        self.transformer_blocks = [
            LongCatJointBlock(dim, heads, ctx) for _ in range(n_joint)
        ]
        self.single_transformer_blocks = [
            LongCatSingleBlock(dim, heads, ctx) for _ in range(n_single)
        ]

        self.norm_out = LongCatAdaLNOutput(dim, ctx)
        self.proj_out = nn.Linear(dim, 64, bias=True)

        self._build_param_map()

    def _patchify(self, latents):
        """2×2 patchify: [B, 16, H, W] → [B, H//2 * W//2, 64]。"""
        ctx = self.ctx
        B, C, H, W = latents.shape
        ps = 2
        # unfold: [B, C, H, W] → [B, C, H//ps, ps, W//ps, ps]
        x = ctx.reshape(latents, (B, C, H // ps, ps, W // ps, ps))
        # permute: [B, H//ps, W//ps, C, ps, ps]
        x = ctx.permute(x, (0, 2, 4, 1, 3, 5))
        # reshape: [B, H//ps * W//ps, C * ps * ps]
        x = ctx.reshape(x, (B, (H // ps) * (W // ps), C * ps * ps))
        return x

    def _unpatchify(self, x, H, W):
        """[B, H//2 * W//2, 64] → [B, 16, H, W]。"""
        ctx = self.ctx
        B = x.shape[0]
        ps = 2
        C = 16
        x = ctx.reshape(x, (B, H // ps, W // ps, C, ps, ps))
        x = ctx.permute(x, (0, 3, 1, 4, 2, 5))
        x = ctx.reshape(x, (B, C, H, W))
        return x

    def forward(self, latents, timestep, txt_embeds=None, **cond):
        ctx = self.ctx
        B = latents.shape[0]

        # TODO(LongCat): 数值对齐问题——当前实现功能完整但输出质量差（棋盘格噪声）
        # 详见 docs/LONGCAT_DEBUG.md。需对比 diffusers 参考实现修复。
        #
        # Convert integer timestep index to actual timestep value
        # FlowMatchEulerScheduler returns indices [0, 1, 2, ...] but LongCat expects actual time values
        # LongCat uses shift=3.0 in its scheduler config, which corresponds to mu≈3.0
        # This compresses timesteps toward the high end: [1000, 983, 952, 870] for 4 steps
        if isinstance(timestep, int) and 0 <= timestep <= 50:
            idx = timestep
            num_steps = max(idx + 1, 4)
            # Use mu=3.0 to match LongCat's scheduler config (shift=3.0)
            import mlx.core as mx
            sigma = 1.0 - (idx / num_steps) * (1.0 - 1.0 / num_steps)
            # Apply time shift with mu=3.0
            mu = 3.0
            sigma = mx.exp(mu) / (mx.exp(mu) + ((1.0 / sigma - 1.0) ** 1.0))
            timestep = float(sigma * 1000.0)

        # Patchify
        H, W = latents.shape[2], latents.shape[3]
        x = self._patchify(latents)
        hidden_states = self.x_embedder(x)

        # Text embeds
        if txt_embeds is not None:
            encoder_hidden_states = self.context_embedder(txt_embeds)
        else:
            encoder_hidden_states = ctx.zeros((B, 256, self.config.hidden_dim))

        # Time embedding
        c = self.time_embed.forward(timestep)

        # Joint blocks
        for block in self.transformer_blocks:
            img_mod = self._split_modulation(block.norm1(c), 6)
            txt_mod = self._split_modulation(block.norm1_context(c), 6)
            encoder_hidden_states, hidden_states = block.forward(
                hidden_states, encoder_hidden_states, img_mod, txt_mod)

        # Single blocks: concatenate img + txt
        x = ctx.concat([hidden_states, encoder_hidden_states], axis=1)
        for block in self.single_transformer_blocks:
            mod = self._split_modulation(block.norm(c), 3)
            x = block.forward(x, mod)

        # Split back
        img_seq_len = hidden_states.shape[1]
        hidden_states = x[:, :img_seq_len]

        # Output
        hidden_states = self.norm_out.forward(hidden_states, c)
        hidden_states = self.proj_out(hidden_states)

        # Unpatchify
        output = self._unpatchify(hidden_states, H, W)
        return output

    def _split_modulation(self, modulation, num_params):
        """Split [B, num_params*dim] into num_params × [B, dim]。"""
        ctx = self.ctx
        B = modulation.shape[0]
        dim = self.config.hidden_dim
        modulation = modulation.reshape(B, num_params, dim)
        return [modulation[:, i, :] for i in range(num_params)]
