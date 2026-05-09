"""
Flux.2 Klein Transformer — 忠实迁移 mflux Flux2Transformer。

架构: MM-DiT 双流 + AdaLayerNormContinuous 调制 + Qwen3 文本编码器。
in_channels=128, inner_dim=4096 (9B) / 3072 (4B)
"""
from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn

from backend.engine.runtime._base import RuntimeContext
from backend.engine.common._base import TransformerBase


def _apply_rope_bshd(x, cos, sin):
    """与 mflux AttentionUtils.apply_rope_bshd 一致。"""
    out_dtype = x.dtype
    cos = cos.astype(mx.float32)
    sin = sin.astype(mx.float32)
    cos = cos.reshape(1, 1, *cos.shape)
    sin = sin.reshape(1, 1, *sin.shape)
    x_f = x.astype(mx.float32)
    x2 = x_f.reshape(*x_f.shape[:-1], -1, 2)
    real = x2[..., 0]
    imag = x2[..., 1]
    rotated = mx.stack([
        real * cos + (-imag) * sin,
        imag * cos + real * sin,
    ], axis=-1)
    rotated = rotated.reshape(*x_f.shape)
    return rotated.astype(out_dtype)


class Flux2Modulation:
    """AdaLayerNorm 调制: scale + shift + gate, 支持多组并行。"""

    def __init__(self, dim: int, mod_param_sets: int, ctx: RuntimeContext):
        self.ctx = ctx
        nn = ctx
        self.mod_param_sets = mod_param_sets
        self.linear = nn.Linear(dim, dim * mod_param_sets * 3, bias=False)

    def forward(self, c):
        import mlx.nn as nn
        ctx = self.ctx
        out = nn.silu(c)
        out = self.linear(out)
        # Match mflux: if 2D, expand_dims to 3D [B, 1, dim*3*mod_param_sets]
        if out.ndim == 2:
            out = mx.expand_dims(out, axis=1)
        # Split into mod_param_sets groups of 3
        mod_params = mx.split(out, 3 * self.mod_param_sets, axis=-1)
        return tuple(mod_params[3 * i : 3 * (i + 1)] for i in range(self.mod_param_sets))


class Flux2TimestepEmbeddings:
    """时间步 + guidance 嵌入 — 与 mflux Flux2TimestepGuidanceEmbeddings 一致。"""

    def __init__(self, in_channels: int, embedding_dim: int, ctx: RuntimeContext, guidance_embeds: bool = False):
        nn = ctx
        self.ctx = ctx
        self.guidance_embeds = guidance_embeds
        self.freq_dim = in_channels // 2
        self.linear_1 = nn.Linear(in_channels, embedding_dim, bias=False)
        self.linear_2 = nn.Linear(embedding_dim, embedding_dim, bias=False)
        if guidance_embeds:
            self.guidance_linear_1 = nn.Linear(in_channels, embedding_dim, bias=False)
            self.guidance_linear_2 = nn.Linear(embedding_dim, embedding_dim, bias=False)
        else:
            self.guidance_linear_1 = None
            self.guidance_linear_2 = None

    def forward(self, t, guidance=None):
        import mlx.core as mx
        import mlx.nn as mxnn
        import math
        t = t.astype(mx.float32)
        half = self.freq_dim
        freqs = mx.exp(-math.log(10000.0) * mx.arange(0, half, dtype=mx.float32) / half)
        args = t[:, None] * freqs[None]
        emb = mx.concatenate([mx.sin(args), mx.cos(args)], axis=-1)
        # flip_sin_to_cos: [sin, cos] → [cos, sin]
        emb = mx.concatenate([emb[:, half:], emb[:, :half]], axis=-1)
        if self.freq_dim % 2:
            emb = mx.concatenate([emb, mx.zeros((emb.shape[0], 1), dtype=emb.dtype)], axis=-1)
        temb = self.linear_2(mxnn.silu(self.linear_1(emb)))
        if guidance is not None and self.guidance_linear_1 is not None and self.guidance_linear_2 is not None:
            g_args = guidance[:, None].astype(mx.float32) * freqs[None]
            g_emb = mx.concatenate([mx.sin(g_args), mx.cos(g_args)], axis=-1)
            g_emb = mx.concatenate([g_emb[:, half:], g_emb[:, :half]], axis=-1)
            if self.freq_dim % 2:
                g_emb = mx.concatenate([g_emb, mx.zeros((g_emb.shape[0], 1), dtype=g_emb.dtype)], axis=-1)
            temb = temb + self.guidance_linear_2(mxnn.silu(self.guidance_linear_1(g_emb)))
        return temb


class Flux2PosEmbed:
    """RoPE 位置嵌入 — 与 mflux Flux2PosEmbed.__call__ 一致。"""

    def __init__(self, theta: float, axes_dim: list[int], ctx: RuntimeContext):
        self.ctx = ctx
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids):
        import mlx.core as mx
        cos_out = []
        sin_out = []
        pos = ids.astype(mx.float32)
        for i, dim in enumerate(self.axes_dim):
            cos, sin = self._get_1d_rope(dim, pos[..., i])
            cos_out.append(cos)
            sin_out.append(sin)
        freqs_cos = mx.concatenate(cos_out, axis=-1)
        freqs_sin = mx.concatenate(sin_out, axis=-1)
        return freqs_cos, freqs_sin

    @staticmethod
    def _get_1d_rope(dim: int, pos: mx.array):
        scale = mx.arange(0, dim, 2, dtype=mx.float32) / dim
        omega = 1.0 / (2000.0 ** scale)
        pos_expanded = mx.expand_dims(pos, axis=-1)
        omega_expanded = mx.expand_dims(omega, axis=0)
        out = pos_expanded * omega_expanded
        cos_out = mx.cos(out)
        sin_out = mx.sin(out)
        return cos_out, sin_out


class Flux2Attention:
    """双流注意力: img + txt 交叉注意力 (added_kv_proj_dim)。"""

    def __init__(self, dim: int, heads: int, dim_head: int, added_kv_proj_dim: int, ctx: RuntimeContext):
        nn = ctx
        self.ctx = ctx
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(dim, heads * dim_head, bias=False)
        self.to_k = nn.Linear(dim, heads * dim_head, bias=False)
        self.to_v = nn.Linear(dim, heads * dim_head, bias=False)
        self.norm_q = nn.RMSNorm(dim_head, eps=1e-5)
        self.norm_k = nn.RMSNorm(dim_head, eps=1e-5)
        self.to_out = nn.Linear(heads * dim_head, dim, bias=False)

        self.add_q_proj = nn.Linear(dim, heads * dim_head, bias=False)
        self.add_k_proj = nn.Linear(added_kv_proj_dim, heads * dim_head, bias=False)
        self.add_v_proj = nn.Linear(added_kv_proj_dim, heads * dim_head, bias=False)
        self.norm_added_q = nn.RMSNorm(dim_head, eps=1e-5)
        self.norm_added_k = nn.RMSNorm(dim_head, eps=1e-5)
        self.to_add_out = nn.Linear(heads * dim_head, dim, bias=False)

    def forward(self, hidden_states, encoder_hidden_states, image_rotary_emb):
        """与 mflux Flux2Attention.__call__ 一致：联合 attention。"""
        ctx = self.ctx
        B, S_img, _ = hidden_states.shape
        B_txt, S_txt, _ = encoder_hidden_states.shape
        head_dim = self.dim_head

        def process_qkv(h, to_q, to_k, to_v):
            batch = h.shape[0]
            q = to_q(h).reshape(batch, -1, self.heads, head_dim)
            k = to_k(h).reshape(batch, -1, self.heads, head_dim)
            v = to_v(h).reshape(batch, -1, self.heads, head_dim)
            q = ctx.permute(q, (0, 2, 1, 3))
            k = ctx.permute(k, (0, 2, 1, 3))
            v = ctx.permute(v, (0, 2, 1, 3))
            return q, k, v

        def norm_qk(q, k, norm_q, norm_k):
            q = norm_q(q.astype(mx.float32)).astype(q.dtype)
            k = norm_k(k.astype(mx.float32)).astype(k.dtype)
            return q, k

        q_img, k_img, v_img = process_qkv(hidden_states, self.to_q, self.to_k, self.to_v)
        q_img, k_img = norm_qk(q_img, k_img, self.norm_q, self.norm_k)
        q_txt, k_txt, v_txt = process_qkv(encoder_hidden_states, self.add_q_proj, self.add_k_proj, self.add_v_proj)
        q_txt, k_txt = norm_qk(q_txt, k_txt, self.norm_added_q, self.norm_added_k)

        # 拼接 text + image
        q = mx.concatenate([q_txt, q_img], axis=2)
        k = mx.concatenate([k_txt, k_img], axis=2)
        v = mx.concatenate([v_txt, v_img], axis=2)

        # RoPE on joint sequence
        cos, sin = image_rotary_emb
        q = self._rotary(q, cos, sin)
        k = self._rotary(k, cos, sin)

        # Attention
        out = ctx.attention(q, k, v, scale=self.scale)
        out = ctx.permute(out, (0, 2, 1, 3))
        out = out.reshape(B, -1, self.heads * head_dim)

        # 拆分 text 和 image 输出
        txt_out = out[:, :S_txt, :]
        img_out = out[:, S_txt:, :]

        return self.to_out(img_out), self.to_add_out(txt_out)

    def _rotary(self, x, cos, sin):
        return _apply_rope_bshd(x, cos, sin)


class Flux2FeedForward:
    """SwiGLU-style FFN with separate linear_in / linear_out matching diffusers keys."""

    def __init__(self, dim: int, mult: float = 3.0, ctx: RuntimeContext = None):
        nn = ctx
        self.ctx = ctx
        hidden_dim = int(dim * mult)  # 4096 * 3 = 12288
        self.linear_in = nn.Linear(dim, hidden_dim * 2, bias=False)  # mflux uses bias=False
        self.linear_out = nn.Linear(hidden_dim, dim, bias=False)     # mflux uses bias=False

    def forward(self, x):
        gate_up = self.linear_in(x)
        gate, up = gate_up[..., :gate_up.shape[-1] // 2], gate_up[..., gate_up.shape[-1] // 2:]
        return self.linear_out(nn.silu(gate) * up)


class Flux2JointBlock:
    def __init__(self, dim: int, num_heads: int, head_dim: int, mlp_ratio: float, ctx: RuntimeContext):
        nn = ctx
        self.ctx = ctx
        self.norm1 = nn.LayerNorm(dim, eps=1e-6, affine=False)
        self.norm1_context = nn.LayerNorm(dim, eps=1e-6, affine=False)
        self.attn = Flux2Attention(dim, num_heads, head_dim, dim, ctx)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6, affine=False)
        self.ff = Flux2FeedForward(dim, mlp_ratio, ctx)
        self.norm2_context = nn.LayerNorm(dim, eps=1e-6, affine=False)
        self.ff_context = Flux2FeedForward(dim, mlp_ratio, ctx)

    def forward(self, hidden_states, encoder_hidden_states, temb_mod_params_img, temb_mod_params_txt, image_rotary_emb):
        (shift_msa, scale_msa, gate_msa), (shift_mlp, scale_mlp, gate_mlp) = temb_mod_params_img
        (c_shift_msa, c_scale_msa, c_gate_msa), (c_shift_mlp, c_scale_mlp, c_gate_mlp) = temb_mod_params_txt

        n_img = self.norm1(hidden_states)
        n_img = (1 + scale_msa) * n_img + shift_msa
        n_txt = self.norm1_context(encoder_hidden_states)
        n_txt = (1 + c_scale_msa) * n_txt + c_shift_msa

        img_out, txt_out = self.attn.forward(n_img, n_txt, image_rotary_emb)
        hidden_states = hidden_states + gate_msa * img_out
        encoder_hidden_states = encoder_hidden_states + c_gate_msa * txt_out

        n_img = self.norm2(hidden_states)
        n_img = (1 + scale_mlp) * n_img + shift_mlp
        n_txt = self.norm2_context(encoder_hidden_states)
        n_txt = (1 + c_scale_mlp) * n_txt + c_shift_mlp

        hidden_states = hidden_states + gate_mlp * self.ff.forward(n_img)
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp * self.ff_context.forward(n_txt)
        return encoder_hidden_states, hidden_states


class Flux2ParallelSelfAttention:
    """参考 mflux Flux2ParallelSelfAttention — QKV+MLP 联合投影。"""

    def __init__(self, dim: int, heads: int, dim_head: int, mlp_ratio: float, ctx: RuntimeContext):
        nn = ctx
        self.ctx = ctx
        self.heads = heads
        self.dim_head = dim_head
        self.inner_dim = heads * dim_head
        self.mlp_hidden_dim = int(dim * mlp_ratio)
        self.to_qkv_mlp_proj = nn.Linear(dim, self.inner_dim * 3 + self.mlp_hidden_dim * 2, bias=False)
        self.norm_q = nn.RMSNorm(dim_head, eps=1e-5)
        self.norm_k = nn.RMSNorm(dim_head, eps=1e-5)
        self.to_out = nn.Linear(self.inner_dim + self.mlp_hidden_dim, dim, bias=False)

    def forward(self, hidden_states: mx.array, image_rotary_emb):
        ctx = self.ctx
        B, S, _ = hidden_states.shape

        # 联合投影 [B, S, inner_dim*3 + mlp_hidden_dim*2]
        proj = self.to_qkv_mlp_proj(hidden_states)
        qkv, mlp = mx.split(proj, [self.inner_dim * 3], axis=-1)
        q, k, v = mx.split(qkv, 3, axis=-1)

        # Reshape to [B, H, S, D]
        q = ctx.reshape(q, (B, S, self.heads, self.dim_head))
        k = ctx.reshape(k, (B, S, self.heads, self.dim_head))
        v = ctx.reshape(v, (B, S, self.heads, self.dim_head))
        q = ctx.permute(q, (0, 2, 1, 3))
        k = ctx.permute(k, (0, 2, 1, 3))
        v = ctx.permute(v, (0, 2, 1, 3))

        # QK norm in float32 (参考 mflux)
        q = self.norm_q(q.astype(mx.float32)).astype(q.dtype)
        k = self.norm_k(k.astype(mx.float32)).astype(k.dtype)

        # RoPE
        if image_rotary_emb is not None:
            cos, sin = image_rotary_emb
            q = _apply_rope_bshd(q, cos, sin)
            k = _apply_rope_bshd(k, cos, sin)

        # Attention
        scale = self.dim_head ** -0.5
        hidden_states = ctx.attention(q, k, v, scale=scale)
        hidden_states = ctx.permute(hidden_states, (0, 2, 1, 3))
        hidden_states = ctx.reshape(hidden_states, (B, S, self.inner_dim))

        # MLP (SwiGLU)
        mlp_gate, mlp_proj = mx.split(mlp, [self.mlp_hidden_dim], axis=-1)
        mlp_out = nn.silu(mlp_gate) * mlp_proj

        # Concat and project
        hidden_states = ctx.concat([hidden_states, mlp_out], axis=-1)
        hidden_states = self.to_out(hidden_states)
        return hidden_states


class Flux2SingleBlock:
    """Flux2 Single Stream Block — 参考 mflux Flux2SingleTransformerBlock。

    包含 self.attn = Flux2ParallelSelfAttention，参数路径:
      single_blocks.N.attn.to_qkv_mlp_proj.weight
      single_blocks.N.attn.norm_q.weight
      single_blocks.N.attn.norm_k.weight
      single_blocks.N.attn.to_out.weight
    """

    def __init__(self, dim: int, num_heads: int, head_dim: int, mlp_ratio: float, ctx: RuntimeContext):
        nn = ctx
        self.ctx = ctx
        self.norm = nn.LayerNorm(dim, eps=1e-6, affine=False)
        self.attn = Flux2ParallelSelfAttention(dim, num_heads, head_dim, mlp_ratio, ctx)

    def forward(self, hidden_states, temb_mod_params, image_rotary_emb):
        shift, scale, gate = temb_mod_params
        normed = self.norm(hidden_states)
        normed = (1 + scale) * normed + shift

        attn_output = self.attn.forward(normed, image_rotary_emb)
        hidden_states = hidden_states + gate * attn_output
        return hidden_states


class Flux2Transformer(TransformerBase):
    """Flux.2 Klein Transformer — 忠实迁移 mflux 实现。

    in_channels=128, inner_dim=4096 (9B), Qwen3 文本编码器
    """

    def __init__(self, config, ctx: RuntimeContext):
        self.config = config
        self.ctx = ctx
        nn = ctx
        dim = config.num_heads * config.attn_head_dim if hasattr(config, 'attn_head_dim') else config.hidden_dim
        num_heads = config.num_heads if hasattr(config, 'num_heads') else getattr(config, 'num_attention_heads', 32)
        head_dim = getattr(config, 'attn_head_dim', getattr(config, 'attention_head_dim', 128))
        num_layers = config.num_layers if hasattr(config, 'num_layers') else getattr(config, 'num_joint_layers', 8)
        num_single = config.num_single_layers if hasattr(config, 'num_single_layers') else 24
        mlp_ratio = getattr(config, 'mlp_ratio', 3.0)
        joint_dim = getattr(config, 'joint_attn_dim', getattr(config, 'joint_attention_dim', 12288))
        in_ch = getattr(config, 'in_channels', 128)
        self.dtype = getattr(config, 'dtype', mx.bfloat16)

        self.inner_dim = dim
        self.pos_embed = Flux2PosEmbed(2000, [32, 32, 32, 32], ctx)
        self.time_guidance_embed = Flux2TimestepEmbeddings(256, dim, ctx, guidance_embeds=False)

        self.double_stream_modulation_img = Flux2Modulation(dim, 2, ctx)
        self.double_stream_modulation_txt = Flux2Modulation(dim, 2, ctx)
        self.single_stream_modulation = Flux2Modulation(dim, 1, ctx)

        self.x_embedder = nn.Linear(in_ch, dim, bias=False)
        self.context_embedder = nn.Linear(joint_dim, dim, bias=False)

        self.transformer_blocks = [
            Flux2JointBlock(dim, num_heads, head_dim, mlp_ratio, ctx)
            for _ in range(num_layers)
        ]
        self.single_transformer_blocks = [
            Flux2SingleBlock(dim, num_heads, head_dim, mlp_ratio, ctx)
            for _ in range(num_single)
        ]

        from backend.engine.common.norm import AdaLayerNormContinuous
        self.norm_out = AdaLayerNormContinuous(dim, dim, ctx)
        patch_size = getattr(config, 'patch_size', 1)
        self.proj_out = nn.Linear(dim, patch_size * patch_size * getattr(config, 'out_channels', in_ch), bias=False)

        self._build_param_map()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, latents, timestep, txt_embeds=None, sigmas=None, **cond):
        """Flux2 前向 — Pipeline 统一传入 int index，由模型自己处理所有转换。"""
        ctx = self.ctx
        B = latents.shape[0]
        import mlx.core as mx

        # ------------------------------------------------------------------
        # 1. Timestep 转换（Pipeline 传入 int index）
        # ------------------------------------------------------------------
        t_idx = int(timestep)
        if sigmas is not None:
            timestep_val = (sigmas[t_idx] * 1000.0).reshape((1,))
        else:
            # fallback: 直接作为 float 处理
            if not isinstance(timestep, mx.array):
                timestep_val = mx.array(timestep, dtype=mx.float32)
            else:
                timestep_val = timestep
            if timestep_val.ndim == 0:
                timestep_val = mx.full((B,), timestep_val, dtype=mx.float32)
            # Auto-scale: if timestep <= 1.0, scale by 1000 (matching mflux)
            timestep_scale = mx.where(mx.max(timestep_val) <= 1.0, 1000.0, 1.0)
            timestep_val = timestep_val * timestep_scale

        temb = self.time_guidance_embed.forward(timestep_val)
        import mlx.core as mx
        temb = temb.astype(mx.bfloat16)

        # ------------------------------------------------------------------
        # 2. Latent 格式处理 — 自动 pack [B, C, H, W] → [B, H*W, C]
        # ------------------------------------------------------------------
        if latents.ndim == 4:
            latents = ctx.permute(latents, (0, 2, 3, 1))
            latents = ctx.reshape(latents, (B, -1, latents.shape[-1]))
        hidden_states = self.x_embedder(latents)
        encoder_hidden_states = self.context_embedder(txt_embeds) if txt_embeds is not None else ctx.zeros((B, 256, self.inner_dim))

        # ------------------------------------------------------------------
        # 3. Position IDs — 模型自己生成（不依赖 Pipeline）
        # ------------------------------------------------------------------
        H = W = int(hidden_states.shape[1] ** 0.5)
        img_ids = self._make_ids(B, H, W)
        txt_ids = self._make_text_ids(B, encoder_hidden_states.shape[1]) if encoder_hidden_states.shape[1] > 0 else None

        img_ids_2d = img_ids[0] if img_ids.ndim == 3 else img_ids
        txt_ids_2d = txt_ids[0] if txt_ids is not None and txt_ids.ndim == 3 else txt_ids

        img_rotary = self.pos_embed.forward(img_ids_2d) if img_ids_2d is not None else (mx.zeros((1,)), mx.zeros((1,)))
        txt_rotary = self.pos_embed.forward(txt_ids_2d) if txt_ids_2d is not None else (mx.zeros((1,)), mx.zeros((1,)))
        concat_rotary = (
            mx.concatenate([txt_rotary[0], img_rotary[0]], axis=0),
            mx.concatenate([txt_rotary[1], img_rotary[1]], axis=0),
        )

        mod_img = self.double_stream_modulation_img.forward(temb)
        mod_txt = self.double_stream_modulation_txt.forward(temb)

        for block in self.transformer_blocks:
            encoder_hidden_states, hidden_states = block.forward(
                hidden_states, encoder_hidden_states, mod_img, mod_txt, concat_rotary,
            )

        hidden_states = ctx.concat([encoder_hidden_states, hidden_states], axis=1)
        mod_single = self.single_stream_modulation.forward(temb)[0]

        for block in self.single_transformer_blocks:
            hidden_states = block.forward(hidden_states, mod_single, concat_rotary)

        txt_len = encoder_hidden_states.shape[1]
        hidden_states = hidden_states[:, txt_len:, :]
        hidden_states = self.norm_out.forward(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)

        # Unpack: [B, S, C] → [B, C, H, W] (Scheduler 和 VAE 都需要 4D)
        B, S, C = hidden_states.shape
        H = W = int(S ** 0.5)
        hidden_states = hidden_states.reshape(B, H, W, C).transpose(0, 3, 1, 2)
        return hidden_states

    def _make_ids(self, B, H, W):
        """生成位置 IDs — 与 mflux Flux2LatentCreator.prepare_grid_ids 一致。

        顺序: [t=0, h, w, layer=0]，其中 w 先变化（与 latent reshape 一致）。
        """
        ctx = self.ctx
        import mlx.core as mx
        h_ids = mx.arange(H, dtype=mx.int32)
        w_ids = mx.arange(W, dtype=mx.int32)
        h_grid = mx.broadcast_to(mx.expand_dims(h_ids, axis=1), (H, W))
        w_grid = mx.broadcast_to(mx.expand_dims(w_ids, axis=0), (H, W))
        flat_h = h_grid.reshape(-1)
        flat_w = w_grid.reshape(-1)
        t = mx.full(flat_h.shape, 0, dtype=mx.int32)
        layer_ids = mx.zeros_like(flat_h)
        ids = mx.stack([t, flat_h, flat_w, layer_ids], axis=1)
        ids = mx.expand_dims(ids, axis=0)
        return mx.broadcast_to(ids, (B, ids.shape[1], ids.shape[2]))

    def _make_text_ids(self, B, seq_len):
        """生成 text IDs — 与 mflux Flux2PromptEncoder.prepare_text_ids 一致。

        顺序: [t=0, h=0, w=0, token_ids]。
        """
        ctx = self.ctx
        import mlx.core as mx
        t = mx.zeros((seq_len,), dtype=mx.int32)
        h = mx.zeros((seq_len,), dtype=mx.int32)
        w = mx.zeros((seq_len,), dtype=mx.int32)
        token_ids = mx.arange(seq_len, dtype=mx.int32)
        ids = mx.stack([t, h, w, token_ids], axis=1)
        ids = mx.expand_dims(ids, axis=0)
        return mx.broadcast_to(ids, (B, ids.shape[1], ids.shape[2]))

    def load_weights(self, weights, strict=False):
        """加载权重并自动转换为 bfloat16（与 mflux ModelConfig.precision 一致）。"""
        import mlx.core as mx
        loaded, skipped = super().load_weights(weights, strict=strict)
        # Convert all parameters to bfloat16 for numerical consistency with mflux
        for key, param in list(self._param_map.items()):
            if param.dtype != mx.bfloat16:
                new_param = param.astype(mx.bfloat16)
                self._param_map[key] = new_param
                # Update the actual module parameter
                parts = key.split('.')
                obj = self
                for part in parts[:-1]:
                    if part.isdigit():
                        obj = obj[int(part)]
                    else:
                        obj = getattr(obj, part)
                last = parts[-1]
                if hasattr(obj, last):
                    setattr(obj, last, new_param)
                elif hasattr(obj, '_parameters') and last in obj._parameters:
                    obj._parameters[last] = new_param
        return loaded, skipped
