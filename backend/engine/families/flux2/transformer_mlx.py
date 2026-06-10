"""
Flux.2 Klein Transformer — Reference implementation.

Architecture: MM-DiT dual-stream + AdaLayerNormContinuous modulation + Qwen3 text encoder.
in_channels=128, inner_dim=4096 (9B) / 3072 (4B)
"""
from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn

from backend.engine.common.ops.embeddings import sinusoidal_timestep_proj
from backend.engine.common.ops.attention import scaled_dot_product_attention_bhsd_mx
from backend.engine.common.ops.norm import AdaLayerNormContinuous, apply_scale_shift
from backend.engine.runtime._base import RuntimeContext
from backend.engine.common.model.base import TransformerBase


def _apply_rope_bhsd(x, cos, sin):
    """Flux2 joint attention RoPE on ``[B, H, S, D]`` (matches reference apply_rope_bshd)."""
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
        ctx = self.ctx
        out = self.linear(ctx.silu(c))
        if out.ndim == 2:
            out = ctx.expand_dims(out, axis=1)
        mod_params = ctx.split(out, 3 * self.mod_param_sets, axis=-1)
        return tuple(mod_params[3 * i : 3 * (i + 1)] for i in range(self.mod_param_sets))


class Flux2TimestepEmbeddings:
    """Timestep + guidance embedding — matches reference Flux2TimestepGuidanceEmbeddings."""

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
        ctx = self.ctx
        t = t.astype(ctx.float32())
        emb = sinusoidal_timestep_proj(ctx, t, self.freq_dim * 2, flip_sin_to_cos=True)
        temb = self.linear_2(ctx.silu(self.linear_1(emb)))
        if guidance is not None and self.guidance_linear_1 is not None and self.guidance_linear_2 is not None:
            g_emb = sinusoidal_timestep_proj(
                ctx, guidance.astype(ctx.float32()), self.freq_dim * 2, flip_sin_to_cos=True
            )
            temb = temb + self.guidance_linear_2(ctx.silu(self.guidance_linear_1(g_emb)))
        return temb


class Flux2PosEmbed:
    """RoPE position embedding — matches reference Flux2PosEmbed.__call__."""

    def __init__(self, theta: float, axes_dim: list[int], ctx: RuntimeContext):
        self.ctx = ctx
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids):
        ctx = self.ctx
        cos_out = []
        sin_out = []
        pos = ids.astype(ctx.float32())
        for i, dim in enumerate(self.axes_dim):
            cos, sin = self._get_1d_rope(dim, pos[..., i])
            cos_out.append(cos)
            sin_out.append(sin)
        return ctx.concat(cos_out, axis=-1), ctx.concat(sin_out, axis=-1)

    def _get_1d_rope(self, dim: int, pos):
        ctx = self.ctx
        scale = ctx.arange(0, dim, 2, dtype=ctx.float32()) / dim
        omega = 1.0 / (2000.0 ** scale)
        pos_expanded = ctx.expand_dims(pos, axis=-1)
        omega_expanded = ctx.expand_dims(omega, axis=0)
        out = pos_expanded * omega_expanded
        return ctx.cos(out), ctx.sin(out)


class Flux2Attention:
    """双流注意力: img + txt 交叉注意力 (added_kv_proj_dim)。"""

    def __init__(self, dim: int, heads: int, dim_head: int, added_kv_proj_dim: int, ctx: RuntimeContext):
        nn = ctx
        self.ctx = ctx
        self.heads = heads
        self.dim_head = dim_head

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
        """Matches reference Flux2Attention.__call__: joint attention."""
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
            q = norm_q(q.astype(ctx.float32())).astype(q.dtype)
            k = norm_k(k.astype(ctx.float32())).astype(k.dtype)
            return q, k

        q_img, k_img, v_img = process_qkv(hidden_states, self.to_q, self.to_k, self.to_v)
        q_img, k_img = norm_qk(q_img, k_img, self.norm_q, self.norm_k)
        q_txt, k_txt, v_txt = process_qkv(encoder_hidden_states, self.add_q_proj, self.add_k_proj, self.add_v_proj)
        q_txt, k_txt = norm_qk(q_txt, k_txt, self.norm_added_q, self.norm_added_k)

        q = ctx.concat([q_txt, q_img], axis=2)
        k = ctx.concat([k_txt, k_img], axis=2)
        v = ctx.concat([v_txt, v_img], axis=2)

        cos, sin = image_rotary_emb
        q = self._rotary(q, cos, sin)
        k = self._rotary(k, cos, sin)

        attn_scale = 1 / ctx.sqrt(q.shape[-1])
        out = scaled_dot_product_attention_bhsd_mx(mx, q, k, v, scale=float(attn_scale))
        out = ctx.permute(out, (0, 2, 1, 3))
        out = ctx.reshape(out, (B, -1, self.heads * head_dim))

        # 拆分 text 和 image 输出
        txt_out = out[:, :S_txt, :]
        img_out = out[:, S_txt:, :]

        return self.to_out(img_out), self.to_add_out(txt_out)

    def _rotary(self, x, cos, sin):
        return _apply_rope_bhsd(x, cos, sin)


class Flux2FeedForward:
    """SwiGLU-style FFN with separate linear_in / linear_out matching diffusers keys."""

    def __init__(self, dim: int, mult: float = 3.0, ctx: RuntimeContext = None):
        nn = ctx
        self.ctx = ctx
        hidden_dim = int(dim * mult)  # 4096 * 3 = 12288
        self.linear_in = nn.Linear(dim, hidden_dim * 2, bias=False)  # bias=False per reference
        self.linear_out = nn.Linear(hidden_dim, dim, bias=False)     # bias=False per reference

    def forward(self, x):
        ctx = self.ctx
        gate_up = self.linear_in(x)
        gate, up = gate_up[..., :gate_up.shape[-1] // 2], gate_up[..., gate_up.shape[-1] // 2:]
        return self.linear_out(ctx.silu(gate) * up)


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
        n_img = apply_scale_shift(n_img, scale_msa, shift_msa, add_one=True)
        n_txt = self.norm1_context(encoder_hidden_states)
        n_txt = apply_scale_shift(n_txt, c_scale_msa, c_shift_msa, add_one=True)

        img_out, txt_out = self.attn.forward(n_img, n_txt, image_rotary_emb)
        hidden_states = hidden_states + gate_msa * img_out
        encoder_hidden_states = encoder_hidden_states + c_gate_msa * txt_out

        n_img = self.norm2(hidden_states)
        n_img = apply_scale_shift(n_img, scale_mlp, shift_mlp, add_one=True)
        n_txt = self.norm2_context(encoder_hidden_states)
        n_txt = apply_scale_shift(n_txt, c_scale_mlp, c_shift_mlp, add_one=True)

        hidden_states = hidden_states + gate_mlp * self.ff.forward(n_img)
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp * self.ff_context.forward(n_txt)
        return encoder_hidden_states, hidden_states


class Flux2ParallelSelfAttention:
    """Reference Flux2ParallelSelfAttention — QKV+MLP joint projection."""

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
        qkv, mlp = ctx.split(proj, [self.inner_dim * 3], axis=-1)
        q, k, v = ctx.split(qkv, 3, axis=-1)

        # Reshape to [B, H, S, D]
        q = ctx.reshape(q, (B, S, self.heads, self.dim_head))
        k = ctx.reshape(k, (B, S, self.heads, self.dim_head))
        v = ctx.reshape(v, (B, S, self.heads, self.dim_head))
        q = ctx.permute(q, (0, 2, 1, 3))
        k = ctx.permute(k, (0, 2, 1, 3))
        v = ctx.permute(v, (0, 2, 1, 3))

        # QK norm in float32 (reference)
        q = self.norm_q(q.astype(ctx.float32())).astype(ctx.bfloat16())
        k = self.norm_k(k.astype(ctx.float32())).astype(ctx.bfloat16())

        if image_rotary_emb is not None:
            cos, sin = image_rotary_emb
            q = _apply_rope_bhsd(q, cos, sin)
            k = _apply_rope_bhsd(k, cos, sin)

        scale = 1 / ctx.sqrt(q.shape[-1])
        hidden_states = scaled_dot_product_attention_bhsd_mx(mx, q, k, v, scale=float(scale))
        hidden_states = ctx.permute(hidden_states, (0, 2, 1, 3))
        hidden_states = ctx.reshape(hidden_states, (B, S, self.inner_dim))

        mlp_gate, mlp_proj = ctx.split(mlp, [self.mlp_hidden_dim], axis=-1)
        mlp_out = ctx.silu(mlp_gate) * mlp_proj

        # Concat and project
        hidden_states = ctx.concat([hidden_states, mlp_out], axis=-1)
        hidden_states = self.to_out(hidden_states)
        return hidden_states


class Flux2SingleBlock:
    """Flux2 Single Stream Block — reference Flux2SingleTransformerBlock.

    Contains self.attn = Flux2ParallelSelfAttention, weight paths:
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
        normed = apply_scale_shift(normed, scale, shift, add_one=True)

        attn_output = self.attn.forward(normed, image_rotary_emb)
        hidden_states = hidden_states + gate * attn_output
        return hidden_states


class Flux2DiTMLX(TransformerBase):
    """Flux.2 Klein Transformer — Reference implementation.

    in_channels=128, inner_dim=4096 (9B), Qwen3 text encoder
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
        self.dtype = getattr(config, 'dtype', ctx.bfloat16())

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

        self.norm_out = AdaLayerNormContinuous(dim, dim, ctx)
        patch_size = getattr(config, 'patch_size', 1)
        self.proj_out = nn.Linear(dim, patch_size * patch_size * getattr(config, 'out_channels', in_ch), bias=False)

        self._build_param_map()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def sanitize(self, weights: dict[str, Any]) -> dict[str, Any]:
        """Map diffusers Flux.2 Klein weight keys to DanQing engine keys."""
        remapped: dict[str, Any] = {}
        for key, tensor in weights.items():
            new_key = key
            new_key = new_key.replace(".to_out.0.", ".to_out.")
            new_key = new_key.replace(".to_add_out.0.", ".to_add_out.")
            new_key = new_key.replace("time_guidance_embed.timestep_embedder.", "time_guidance_embed.")
            remapped[new_key] = tensor
        return remapped

    def forward(self, latents, timestep, txt_embeds=None, sigmas=None, **cond):
        """Flux2 前向 — Pipeline 统一传入 int index，由模型自己处理所有转换。"""
        ctx = self.ctx
        B = latents.shape[0]

        # ------------------------------------------------------------------
        # 1. Timestep 转换（Pipeline 传入 int index）
        # ------------------------------------------------------------------
        timestep_val = self._resolve_timestep_value(B, timestep, sigmas)
        if hasattr(latents, "dtype"):
            timestep_val = timestep_val.astype(latents.dtype)

        temb = self.time_guidance_embed.forward(timestep_val)
        temb = temb.astype(ctx.bfloat16())

        # ------------------------------------------------------------------
        # 2. Latent 格式处理 — 自动 pack [B, C, H, W] → [B, H*W, C]
        # ------------------------------------------------------------------
        latent_h: int | None = None
        latent_w: int | None = None
        if latents.ndim == 4:
            _, _, lh, lw = latents.shape
            latent_h, latent_w = int(lh), int(lw)
            latents = ctx.permute(latents, (0, 2, 3, 1))
            latents = ctx.reshape(latents, (B, -1, latents.shape[-1]))
        elif latents.ndim == 3:
            seq = int(latents.shape[1])
            lh_kw = cond.get("latent_h")
            lw_kw = cond.get("latent_w")
            if lh_kw is not None and lw_kw is not None:
                latent_h, latent_w = int(lh_kw), int(lw_kw)
                if latent_h * latent_w != seq:
                    raise RuntimeError(
                        "Flux2Transformer: latent_h * latent_w must match sequence length "
                        f"(got latent_h={latent_h}, latent_w={latent_w}, seq_len={seq})."
                    )
            else:
                root = int(seq**0.5)
                if root * root != seq:
                    raise RuntimeError(
                        "Flux2Transformer: flattened latents are non-square; pass latent_h and latent_w "
                        f"in model kwargs (seq_len={seq})."
                    )
                latent_h = latent_w = root
        else:
            raise RuntimeError(
                f"Flux2Transformer: expected latents ndim 3 or 4, got shape={tuple(latents.shape)}."
            )
        hidden_states = self.x_embedder(latents)
        encoder_hidden_states = self.context_embedder(txt_embeds) if txt_embeds is not None else ctx.zeros((B, 256, self.inner_dim))

        # ------------------------------------------------------------------
        # 3. Position IDs — 模型自己生成（不依赖 Pipeline）
        # ------------------------------------------------------------------
        if latent_h is None or latent_w is None:
            raise RuntimeError("Flux2Transformer: could not resolve latent_h/latent_w for position ids.")
        H, W = latent_h, latent_w
        img_ids = self._make_ids(B, H, W)
        txt_ids = self._make_text_ids(B, encoder_hidden_states.shape[1]) if encoder_hidden_states.shape[1] > 0 else None

        img_rotary = self._rotary_from_ids(img_ids)
        txt_rotary = self._rotary_from_ids(txt_ids)
        concat_rotary = (
            ctx.concat([txt_rotary[0], img_rotary[0]], axis=0),
            ctx.concat([txt_rotary[1], img_rotary[1]], axis=0),
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
        if latent_h is None or latent_w is None or S != latent_h * latent_w:
            raise RuntimeError(
                "Flux2Transformer: output spatial length mismatch "
                f"(S={S}, latent_h={latent_h}, latent_w={latent_w})."
            )
        hidden_states = hidden_states.reshape(B, latent_h, latent_w, C).transpose(0, 3, 1, 2)
        return hidden_states

    def _resolve_timestep_value(
        self,
        batch_size: int,
        timestep: Any,
        sigmas: Any | None,
    ) -> Any:
        ctx = self.ctx
        if sigmas is not None:
            t_idx: int | None = None
            if isinstance(timestep, int) and not isinstance(timestep, bool):
                t_idx = int(timestep)
            elif isinstance(timestep, mx.array) and timestep.ndim == 0:
                dt = str(timestep.dtype).lower()
                if "int" in dt and "float" not in dt:
                    t_idx = int(timestep.item())
            if t_idx is not None:
                return (sigmas[t_idx] * 1000.0).reshape((1,))

        timestep_val = timestep if hasattr(timestep, "shape") else ctx.array(timestep, dtype=ctx.float32())
        if timestep_val.ndim == 0:
            timestep_val = ctx.full((batch_size,), float(timestep_val), dtype=ctx.float32())
        timestep_scale = ctx.where(ctx.max(timestep_val) <= 1.0, 1000.0, 1.0)
        return timestep_val * timestep_scale

    def _make_ids(self, B, H, W):
        ctx = self.ctx
        h_ids = ctx.arange(0, H, dtype=ctx.int32())
        w_ids = ctx.arange(0, W, dtype=ctx.int32())
        h_grid = ctx.broadcast_to(ctx.expand_dims(h_ids, axis=1), (H, W))
        w_grid = ctx.broadcast_to(ctx.expand_dims(w_ids, axis=0), (H, W))
        flat_h = h_grid.reshape(-1)
        flat_w = w_grid.reshape(-1)
        t = ctx.full(flat_h.shape, 0, dtype=ctx.int32())
        layer_ids = ctx.zeros_like(flat_h)
        ids = ctx.stack([t, flat_h, flat_w, layer_ids], axis=1)
        return self._broadcast_ids_table(B, ids)

    def _make_text_ids(self, B, seq_len):
        ctx = self.ctx
        t = ctx.zeros((seq_len,), dtype=ctx.int32())
        h = ctx.zeros((seq_len,), dtype=ctx.int32())
        w = ctx.zeros((seq_len,), dtype=ctx.int32())
        token_ids = ctx.arange(0, seq_len, dtype=ctx.int32())
        ids = ctx.stack([t, h, w, token_ids], axis=1)
        return self._broadcast_ids_table(B, ids)

    def _broadcast_ids_table(self, batch_size: int, ids_2d):
        ctx = self.ctx
        ids = ctx.expand_dims(ids_2d, axis=0)
        return ctx.broadcast_to(ids, (batch_size, ids.shape[1], ids.shape[2]))

    @staticmethod
    def _ids_to_2d(ids: mx.array | None) -> mx.array | None:
        if ids is None:
            return None
        return ids[0] if ids.ndim == 3 else ids

    def _rotary_from_ids(self, ids):
        ids_2d = self._ids_to_2d(ids)
        if ids_2d is None:
            ctx = self.ctx
            return ctx.zeros((1,)), ctx.zeros((1,))
        return self.pos_embed.forward(ids_2d)

    def load_weights(
        self,
        weights,
        strict=False,
        ctx=None,
        *,
        bundle_affine_bits=None,
        inference_mode=None,
    ):
        """Load weights using checkpoint/native dtype (reference-aligned)."""
        load_ctx = ctx if ctx is not None else self.ctx
        loaded, skipped = super().load_weights(
            weights,
            strict=strict,
            ctx=load_ctx,
            bundle_affine_bits=bundle_affine_bits,
            inference_mode=inference_mode,
        )
        return loaded, skipped
