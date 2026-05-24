"""
LTX Video Transformer — matches diffusers LTXVideoTransformer3DModel.

Architecture (pytorch diffusers → our flat layer naming):
  proj_in: Linear(128 → 2048)
  time_embed: sinusoidal + Linear(256→2048) + Linear(2048→2048)
  time_embed_out: Linear(2048 → 6*2048)
  caption_proj_in / caption_proj_out: Linear(4096→2048) + GELU + Linear(2048→2048)
  output_modulation: [2, 2048] global scale_shift
  28 × LTXBlock: self-attn + cross-attn + mlp_in/mlp_out + per-block scale_shift_table [6,2048]
  final norm: pure RMS (no affine weight; modulated by output_modulation + t_emb)
  proj_out: Linear(2048 → 128)

No nn.Sequential is used — every learnable layer is a named attribute to ensure
``_collect_params`` collects flat ``name → tensor`` entries.
"""
from __future__ import annotations

from typing import Any

from backend.engine.common._base import TransformerBase
from backend.engine.common.attention import _apply_rope, attention_bhsd_to_blhd
from backend.engine.common.embeddings import PatchEmbed3D, RoPE3D, LTXTimestepEmbeddingMLP, sinusoidal_timestep_proj
from backend.engine.common.norm import apply_rms_norm, apply_scale_shift, unpack_modulation_6table
from backend.engine.config.model_configs import LTXConfig
from backend.engine.runtime._base import RuntimeContext


# ---------------------------------------------------------------------------
# LTX self / cross attention (separate Q/K/V, full-dim QK RMSNorm, out bias)
# ---------------------------------------------------------------------------

class _LTXAttention:
    """Matches diffusers attn1 (self) / attn2 (cross) structure.
    Uses separate Q/K/V projections with full-dim RMSNorm (weight [2048]).
    """

    def __init__(self, query_dim: int, kv_dim: int, num_heads: int, ctx: RuntimeContext,
                 qk_norm: bool = True, qkv_bias: bool = True, out_bias: bool = True):
        self.ctx = ctx
        nn = ctx
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.to_q = nn.Linear(query_dim, query_dim, bias=qkv_bias)
        self.to_k = nn.Linear(kv_dim, query_dim, bias=qkv_bias)
        self.to_v = nn.Linear(kv_dim, query_dim, bias=qkv_bias)
        if qk_norm:
            self.q_norm = nn.RMSNorm(query_dim)
            self.k_norm = nn.RMSNorm(query_dim)
        else:
            self.q_norm = None
            self.k_norm = None
        self.to_out = nn.Linear(query_dim, query_dim, bias=out_bias)

    def forward(self, x, context=None, rope_cos=None, rope_sin=None):
        ctx = self.ctx
        B, N, C = x.shape

        q = self.to_q(x)
        kv_src = context if context is not None else x
        k = self.to_k(kv_src)
        v = self.to_v(kv_src)

        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        kv_len = kv_src.shape[1]
        q = ctx.reshape(q, (B, N, self.num_heads, self.head_dim))
        q = ctx.permute(q, (0, 2, 1, 3))
        k = ctx.reshape(k, (B, kv_len, self.num_heads, self.head_dim))
        k = ctx.permute(k, (0, 2, 1, 3))
        v = ctx.reshape(v, (B, kv_len, self.num_heads, self.head_dim))
        v = ctx.permute(v, (0, 2, 1, 3))

        if rope_cos is not None and rope_sin is not None:
            q = _apply_rope(ctx, q, rope_cos, rope_sin)
            k = _apply_rope(ctx, k, rope_cos, rope_sin)

        attn_out = attention_bhsd_to_blhd(ctx, q, k, v, scale=self.scale)
        attn_out = ctx.reshape(attn_out, (B, N, C))
        return self.to_out(attn_out)

    def __call__(self, x, context=None, rope_cos=None, rope_sin=None):
        return self.forward(x, context, rope_cos, rope_sin)


# ---------------------------------------------------------------------------
# LTX Block
# ---------------------------------------------------------------------------

class LTXBlock:
    """Single LTX transformer block.

    Takes pre-computed block_cond [B, 6, dim] (t_cond + per-block scale_shift_table).
    Uses pure RMS norms (no learned affine weights).
    """

    def __init__(self, dim: int, num_heads: int, ctx: RuntimeContext):
        nn = ctx
        self.ctx = ctx
        self.dim = dim

        self.self_attn = _LTXAttention(dim, dim, num_heads, ctx,
                                       qk_norm=True, qkv_bias=True, out_bias=True)
        self.cross_attn = _LTXAttention(dim, dim, num_heads, ctx,
                                        qk_norm=True, qkv_bias=True, out_bias=True)
        self.mlp_in = nn.Linear(dim, int(dim * 4))
        self.mlp_out = nn.Linear(int(dim * 4), dim)

        # Registered manually in LTXTransformer._build_param_map.
        self.scale_shift_table = ctx.zeros((6, dim))

    def forward(self, x, text_embeds, block_cond, rope_cos, rope_sin):
        """block_cond: [B, 6, dim]"""
        ctx = self.ctx

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = unpack_modulation_6table(
            block_cond
        )

        # self-attention (modulated)
        x_norm = _rms_norm(x, ctx, 1e-6)
        x_mod = apply_scale_shift(x_norm, scale_msa[:, None, :], shift_msa[:, None, :], add_one=True)
        x = x + gate_msa[:, None, :] * self.self_attn(x_mod, rope_cos=rope_cos, rope_sin=rope_sin)

        # cross-attention (unmodulated)
        if text_embeds is not None:
            x_norm = _rms_norm(x, ctx, 1e-6)
            x = x + self.cross_attn(x_norm, context=text_embeds)

        # feed-forward (modulated)
        x_norm = _rms_norm(x, ctx, 1e-6)
        x_mod = apply_scale_shift(x_norm, scale_mlp[:, None, :], shift_mlp[:, None, :], add_one=True)
        x_ff = self.mlp_in(x_mod)
        x_ff = ctx.gelu(x_ff)
        x_ff = self.mlp_out(x_ff)
        x = x + gate_mlp[:, None, :] * x_ff

        return x


# ---------------------------------------------------------------------------
# LTX Transformer
# ---------------------------------------------------------------------------

class LTXTransformer(TransformerBase):
    """LTX Video Transformer — single-stream spatiotemporal DiT + T5.

    Flow:
    1. PatchEmbed3D:  [B, C, T, H, W] → [B, T*H*W, dim]
    2. T5 text → caption_proj_in → GELU → caption_proj_out → [B, 512, dim]
    3. Timestep → time_embed [B, dim] + time_embed_out [B, 6, dim]
    4. 28 × LTXBlock with per-block scale_shift_table
    5. Output: pure RMSNorm + output_modulation → proj_out → [B, C, T, H, W]
    """

    def __init__(self, config: LTXConfig, ctx: RuntimeContext, num_frames: int = 33):
        self.config = config
        self.ctx = ctx
        nn = ctx
        dim = config.dim
        num_heads = config.num_heads

        # Input embedding
        self.patch_embed = PatchEmbed3D(
            config.dim_in, dim,
            patch_size=(config.temporal_patch_size, config.patch_size, config.patch_size),
            ctx=ctx,
        )

        # Time embedding: sinusoidal → MLP (in → out) → out_proj (→ 6*dim)
        self.time_embed = LTXTimestepEmbeddingMLP(dim, ctx)           # [B] → [B, dim]
        self.time_embed_out = nn.Linear(dim, 6 * dim)               # [B, dim] → [B, 6*dim]

        # Text projection: T5 4096 → dim
        self.caption_proj_in = nn.Linear(config.text_dim, dim)      # 4096 → 2048
        self.caption_proj_out = nn.Linear(dim, dim)                 # 2048 → 2048

        # Global output modulation [2, dim] — shift, scale for final norm
        self.output_modulation = ctx.zeros((2, dim))

        # Transformer blocks
        self.blocks = []
        for _ in range(config.depth):
            self.blocks.append(LTXBlock(dim, num_heads, ctx))

        # Final projection
        self.proj_out = nn.Linear(dim, config.dim_out)

        # 3-D RoPE
        self.rope = RoPE3D(config.head_dim, ctx)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, latents, timestep,
                txt_embeds=None, image_embeds=None, **conditioning):
        ctx = self.ctx
        config = self.config
        B = latents.shape[0]
        dim = config.dim

        timestep_embed_value = conditioning.get("timestep_embed_value")
        if timestep_embed_value is not None:
            t_val = float(timestep_embed_value)
            t_batch = ctx.full((B,), t_val, dtype=ctx.float32())
        elif hasattr(timestep, "dtype"):
            t_batch = timestep
            if getattr(t_batch, "ndim", 0) == 0:
                t_batch = ctx.full((B,), float(t_batch), dtype=ctx.float32())
            else:
                t_batch = ctx.reshape(t_batch, (-1,))
                if int(t_batch.shape[0]) == 1 and B > 1:
                    t_batch = ctx.broadcast_to(t_batch, (B,))
        else:
            raise RuntimeError(
                "LTXTransformer received a scalar step index without ``timestep_embed_value``. "
                "VideoPipeline must pass the scheduler's continuous timestep (same contract as ImagePipeline)."
            )

        # 1. Patch embedding: [B, C, T, H, W] → [B, T*H*W, dim]
        x = self.patch_embed(latents)

        # 2. Timestep → time embedding
        t_emb = self.time_embed(t_batch)                     # [B, dim]
        t_cond = self.time_embed_out(t_emb)                   # [B, 6*dim]
        t_cond = ctx.reshape(t_cond, (B, 6, dim))             # [B, 6, dim]

        # 3. Text projection (T5 output → dim, via GELU)
        if txt_embeds is not None:
            txt_embeds = self.caption_proj_in(txt_embeds)
            txt_embeds = ctx.gelu(txt_embeds)
            txt_embeds = self.caption_proj_out(txt_embeds)
            txt_embeds = txt_embeds.astype(x.dtype)

        # 4. Image condition (I2V)
        if image_embeds is not None:
            x = x + image_embeds

        # 5. 3-D RoPE — grid must match PatchEmbed3D output (H and W may differ).
        if latents.ndim >= 5:
            _, _, T_lat, H_lat, W_lat = latents.shape
        else:
            T_lat, H_lat, W_lat = 1, int(latents.shape[-2]), int(latents.shape[-1])
        pt, ph, pw = self.patch_embed.patch_size
        T = max(int(T_lat) // int(pt), 1)
        H = max(int(H_lat) // int(ph), 1)
        W = max(int(W_lat) // int(pw), 1)
        rope_cos, rope_sin = self.rope(T, H, W)

        # 6. Transformer blocks
        for i, blk in enumerate(self.blocks):
            block_cond = t_cond + blk.scale_shift_table[None, :, :]  # [B, 6, dim]
            x = blk.forward(x, txt_embeds, block_cond, rope_cos, rope_sin)

        # 7. Final norm + output modulation
        x_norm = _rms_norm(x, ctx, 1e-6)
        out_cond = t_emb[:, None, :] + self.output_modulation[None, :, :]  # [B, 2, dim]
        final_shift = out_cond[:, 0, :]
        final_scale = out_cond[:, 1, :]
        x = apply_scale_shift(x_norm, final_scale[:, None, :], final_shift[:, None, :], add_one=True)

        # 8. Output projection → reshape to latent
        x = self.proj_out(x)  # [B, T*H*W, dim_out]
        x = ctx.reshape(x, (B, T, H, W, config.dim_out))
        x = ctx.permute(x, (0, 4, 1, 2, 3))  # [B, C, T, H, W]
        return x

    # ------------------------------------------------------------------
    # Parameter registration
    # ------------------------------------------------------------------

    def _build_param_map(self):
        super()._build_param_map()
        for i, blk in enumerate(self.blocks):
            self._param_map[f"blocks.{i}.scale_shift_table"] = blk.scale_shift_table
        self._param_map["output_modulation"] = self.output_modulation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rms_norm(x, ctx, eps: float = 1e-6):
    """RMS normalization without learned affine weight."""
    weight = ctx.ones((int(x.shape[-1]),), dtype=x.dtype)
    return apply_rms_norm(x, weight, eps)
