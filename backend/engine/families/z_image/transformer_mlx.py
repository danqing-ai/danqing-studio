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

from backend.engine.common.ops.dit_runtime import DiTRuntimeSession
from backend.engine.common.ops.lemica import lemica_compute_steps
from backend.engine.common.ops.step_cache import StepCacheSession
from backend.engine.config.model_configs import ZImageConfig
from backend.engine.runtime._base import RuntimeContext
from backend.engine.common.model.base import TransformerBase
from backend.engine.common.ops.attention import attention_blhd, build_padding_attention_bias
from backend.engine.common.ops.embeddings import (
    apply_complex_rope_from_cis_bshd,
    pad_len_to_multiple as _pad_len_to_multiple,
    build_tail_pad_mask as _build_tail_pad_mask,
    pad_tail_with_last as _pad_tail_with_last,
    apply_pad_token as _apply_pad_token,
    sinusoidal_timestep_proj,
)
from backend.engine.common.ops.norm import apply_scale_shift, unpack_modulation_4way
from backend.engine.common.ops.lemica import LEMICA_SCHEDULES  # noqa: F401 — re-export for tests

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
        ctx = self.ctx
        # Z-Image reference timestep embedding uses [cos, sin] concat order.
        emb = sinusoidal_timestep_proj(
            ctx,
            t,
            self.frequency_embedding_size,
            sin_first=False,
            flip_sin_to_cos=False,
        )
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
            q = apply_complex_rope_from_cis_bshd(ctx, q, freqs_cis)
            k = apply_complex_rope_from_cis_bshd(ctx, k, freqs_cis)

        mask = None
        if attention_mask is not None:
            mask = build_padding_attention_bias(
                ctx,
                attention_mask,
                attention_mask.shape[-1],
                ctx.float32(),
                valid_value=1,
                neg_value=float("-inf"),
            )

        out = attention_blhd(ctx, q, k, v, scale=self.scale, mask=mask)
        out = ctx.reshape(out, (B, S, self.dim))
        out = self.to_out(out)
        return out

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
        scale_msa, gate_msa, scale_mlp, gate_mlp = unpack_modulation_4way(modulation)
        scale_msa = 1.0 + scale_msa
        scale_mlp = 1.0 + scale_mlp
        gate_msa = ctx.tanh(gate_msa)
        gate_mlp = ctx.tanh(gate_mlp)

        # Attention with modulation
        normed = self.attn_norm1(x)
        attn_out = self.attention.forward(
            apply_scale_shift(normed, scale_msa, 0.0, add_one=False),
            attention_mask=attn_mask,
            freqs_cis=freqs_cis,
        )
        x = x + gate_msa * self.attn_norm2(attn_out)

        # FFN with modulation
        normed = self.ffn_norm1(x)
        ffn_out = self.feed_forward.forward(apply_scale_shift(normed, scale_mlp, 0.0, add_one=False))
        x = x + gate_mlp * self.ffn_norm2(ffn_out)
        return x


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
# ZImageDiTMLX — 主模型
# =========================================================================

class ZImageDiTMLX(TransformerBase):
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
        spec = getattr(config, "latent_noise_dtype", None)
        self._act_dtype = ctx.bfloat16() if isinstance(spec, str) and spec.lower() in ("bfloat16", "bf16") else ctx.float32()

        self._param_map: dict[str, Any] = {}
        self._build_param_map()
        # 手动注册非 nn.Module 张量
        self._param_map["x_pad_token"] = self.x_pad_token
        self._param_map["cap_pad_token"] = self.cap_pad_token
        self._compiled_forward = None
        self._compiled_cfg_forward = None
        self._control = None
        self._control_context_scale = 1.0
        self._lemica_bool_list: tuple[bool, ...] | None = None
        self._lemica_step_counter = 0
        self._lemica_previous_residual = None
        self._step_cache: StepCacheSession | None = None
        self._use_mlx_compile_run = False

    def after_load_weights(self, bundle_root=None) -> None:
        super().after_load_weights(bundle_root)
        self._refresh_compiled_forward()

    def configure_lemica(self, mode: str, num_steps: int) -> None:
        self.reset_lemica_state()
        self._lemica_bool_list = lemica_compute_steps(mode, num_steps)

    def reset_lemica_state(self) -> None:
        self._lemica_step_counter = 0
        self._lemica_previous_residual = None

    def activate_z_image_control(self, flat_weights: dict[str, Any], *, context_scale: float = 0.75) -> None:
        from backend.engine.families.z_image.control_mlx import ZImageControlRuntime

        if self._control is None:
            self._control = ZImageControlRuntime(self.config, self.ctx)
        self._control.load_control_weights(flat_weights)
        self._control_context_scale = float(context_scale)

    def deactivate_z_image_control(self) -> None:
        self._control = None
        self._control_context_scale = 1.0

    def sanitize(self, weights: dict[str, Any]) -> dict[str, Any]:
        """Map diffusers-format Z-Image weight keys to DanQing engine keys."""
        patch_size = getattr(self.config, "patch_size", 2)
        prefix_key = f"{patch_size}-1"
        remapped: dict[str, Any] = {}
        for key, tensor in weights.items():
            new_key = key
            new_key = new_key.replace(f"all_x_embedder.{prefix_key}", "x_embedder")
            new_key = new_key.replace(f"all_final_layer.{prefix_key}", "final_layer")
            new_key = new_key.replace("t_embedder.mlp.0.", "t_embedder.linear1.")
            new_key = new_key.replace("t_embedder.mlp.2.", "t_embedder.linear2.")
            new_key = new_key.replace("attention_norm1.", "attn_norm1.")
            new_key = new_key.replace("attention_norm2.", "attn_norm2.")
            new_key = new_key.replace(".to_out.0.", ".to_out.")
            new_key = new_key.replace(".adaLN_modulation.1.", ".adaLN_modulation.0.")
            new_key = new_key.replace("cap_embedder.0.", "cap_norm.")
            new_key = new_key.replace("cap_embedder.1.", "cap_embedder.")
            remapped[new_key] = tensor
        return remapped

    def _refresh_compiled_forward(self) -> None:
        self._compiled_forward = None
        self._compiled_cfg_forward = None
        if getattr(self.ctx, "backend", None) != "mlx":
            return
        if not self._use_mlx_compile_run:
            return
        try:
            self._compiled_forward = self.ctx.compile(self._forward_cached_compute)
        except Exception:
            self._compiled_forward = None
        if not getattr(self.config, "use_mlx_cfg_fusion", True):
            return
        try:
            self._compiled_cfg_forward = self.ctx.compile(self._forward_cfg_cached_compute)
        except Exception:
            self._compiled_cfg_forward = None

    def step_callback(self, step_idx: int, latents: Any, noise_pred: Any) -> None:
        del latents, noise_pred
        if self._step_cache is not None:
            self._step_cache.set_step_counter(int(step_idx) + 1)

    def before_denoise(self, latents, timesteps, sigmas, **cond):
        runtime, cond = DiTRuntimeSession.from_before_denoise_cond(
            family="z_image",
            config=self.config,
            entry=None,
            ctx=self.ctx,
            cond=dict(cond),
            timesteps=timesteps,
        )
        self._step_cache = runtime.step_cache
        self._lemica_bool_list = runtime.lemica_bool_list
        if runtime.step_cache is not None and runtime.step_cache.enabled:
            self.reset_lemica_state()
        elif runtime.lemica_bool_list is not None:
            self.reset_lemica_state()
        else:
            self.reset_lemica_state()
            self._lemica_bool_list = None

        self._use_mlx_compile_run = runtime.use_mlx_compile
        if self._step_cache is not None and self._step_cache.enabled:
            # Gate uses mx.eval / host floats — incompatible inside ctx.compile.
            self._use_mlx_compile_run = False
        self._refresh_compiled_forward()

        if getattr(self.ctx, "backend", None) != "mlx":
            return latents, cond
        if not runtime.plan.needs_precompute_cap():
            return latents, cond
        txt_embeds = cond.pop("txt_embeds", None)
        neg_embeds = cond.pop("neg_embeds", None)
        latents_n = self._normalize_latents(latents)
        if txt_embeds is not None:
            cap_feats = self._resolve_cap_feats(txt_embeds, cond)
            cap_cache = self._encode_cap_branch(cap_feats)
            geo_cache = self._encode_geo_cache(latents_n, cap_cache[2])
            self.ctx.eval(cap_cache[0], cap_cache[1], geo_cache[0], geo_cache[1])
            cond["zimage_cap_cache"] = cap_cache
            cond["zimage_geo_cache"] = geo_cache
        if neg_embeds is not None:
            neg_feats = self._resolve_cap_feats(neg_embeds, cond)
            neg_cap_cache = self._encode_cap_branch(neg_feats)
            neg_geo_cache = self._encode_geo_cache(latents_n, neg_cap_cache[2])
            self.ctx.eval(neg_cap_cache[0], neg_cap_cache[1], neg_geo_cache[0], neg_geo_cache[1])
            cond["zimage_neg_cap_cache"] = neg_cap_cache
            cond["zimage_neg_geo_cache"] = neg_geo_cache
        return latents, cond

    def forward_cfg(
        self,
        latents,
        timestep,
        txt_embeds,
        neg_embeds,
        guidance: float,
        sigmas=None,
        *,
        cfg_renorm: bool = False,
        cfg_renorm_min: float = 0.0,
        **conditioning,
    ):
        """Single fused CFG forward (MLX fast path). Returns combined noise prediction."""
        input_shape = latents.shape
        input_ndim = latents.ndim
        latents_n = self._normalize_latents(latents)
        t = self._resolve_timestep(timestep, sigmas)

        cap_cache = conditioning.get("zimage_cap_cache")
        neg_cap_cache = conditioning.get("zimage_neg_cap_cache")
        geo_cache = conditioning.get("zimage_geo_cache")
        neg_geo_cache = conditioning.get("zimage_neg_geo_cache")
        if cap_cache is None:
            cap_cache = self._encode_cap_branch(self._resolve_cap_feats(txt_embeds, conditioning))
        if neg_cap_cache is None:
            neg_cap_cache = self._encode_cap_branch(self._resolve_cap_feats(neg_embeds, conditioning))
        if geo_cache is None:
            geo_cache = self._encode_geo_cache(latents_n, cap_cache[2])
        if neg_geo_cache is None:
            neg_geo_cache = self._encode_geo_cache(latents_n, neg_cap_cache[2])

        cap_emb, cap_freqs, _ = cap_cache
        neg_cap_emb, neg_cap_freqs, _ = neg_cap_cache
        x_freqs, x_pad_mask, x_len, image_size, _ = geo_cache
        nx_freqs, nx_pad_mask, nx_len, n_image_size, _ = neg_geo_cache

        use_cfg_compile = (
            self._compiled_cfg_forward is not None
            and not cfg_renorm
            and x_len == nx_len
            and image_size == n_image_size
        )
        if use_cfg_compile:
            tokens = self._compiled_cfg_forward(
                latents_n, t,
                cap_emb, cap_freqs, x_freqs, x_pad_mask, x_len,
                neg_cap_emb, neg_cap_freqs, nx_freqs, nx_pad_mask, nx_len,
                float(guidance),
            )
            output = self._unpatchify(tokens, image_size)
            return self._reshape_output(-output, input_shape, input_ndim)

        noise_cond = self._forward_from_caches(
            latents_n, t, cap_emb, cap_freqs, x_freqs, x_pad_mask, x_len, image_size,
            teacache_branch="cond",
        )
        noise_uncond = self._forward_from_caches(
            latents_n, t, neg_cap_emb, neg_cap_freqs, nx_freqs, nx_pad_mask, nx_len, n_image_size,
            teacache_branch="uncond",
        )
        noise_pred = self.combine_cfg_noise(noise_cond, noise_uncond, guidance)
        if cfg_renorm:
            noise_pred = self.refine_cfg_noise(
                noise_cond, noise_pred, cfg_renorm_min=cfg_renorm_min,
            )
        return self._reshape_output(-noise_pred, input_shape, input_ndim)

    def combine_cfg_noise(self, noise_cond, noise_uncond, guidance: float):
        """Z-Image reference CFG convention: ``eps_c + g * (eps_c - eps_u)``."""
        return noise_cond + guidance * (noise_cond - noise_uncond)

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
        input_shape = latents.shape
        input_ndim = latents.ndim
        latents_n = self._normalize_latents(latents)
        t = self._resolve_timestep(timestep, sigmas)

        cap_cache = conditioning.get("zimage_cap_cache")
        geo_cache = conditioning.get("zimage_geo_cache")
        control_ctx = conditioning.get("zimage_control_context")
        control_scale = float(conditioning.get("zimage_control_context_scale", self._control_context_scale))
        teacache_branch = str(conditioning.get("_teacache_branch") or "default")
        if cap_cache is not None and geo_cache is not None:
            cap_emb, cap_freqs, _ = cap_cache
            x_freqs, x_pad_mask, x_len, image_size, _ = geo_cache
            output = self._forward_from_caches(
                latents_n, t, cap_emb, cap_freqs, x_freqs, x_pad_mask, x_len, image_size,
                use_compile=self._use_mlx_compile_run,
                control_context=control_ctx,
                control_context_scale=control_scale,
                teacache_branch=teacache_branch,
            )
        else:
            cap_feats = self._resolve_cap_feats(txt_embeds, conditioning)
            output = self._forward_compute(
                latents_n,
                t,
                cap_feats,
                control_context=conditioning.get("zimage_control_context"),
                control_context_scale=float(
                    conditioning.get("zimage_control_context_scale", self._control_context_scale)
                ),
            )

        return self._reshape_output(-output, input_shape, input_ndim)

    def _resolve_cap_feats(self, txt_embeds, conditioning):
        cap_feats = txt_embeds if txt_embeds is not None else conditioning.get("cap_feats")
        if cap_feats is None:
            raise ValueError("ZImageDiTMLX requires txt_embeds (Qwen3 cap_feats)")
        if cap_feats.ndim == 3 and cap_feats.shape[0] == 1:
            cap_feats = cap_feats[0]
        return cap_feats

    def _normalize_latents(self, latents):
        ctx = self.ctx
        if latents.shape[0] == 1 and latents.ndim in (4, 5):
            latents = latents[0]
        if latents.ndim == 3:
            latents = ctx.reshape(latents, (latents.shape[0], 1, latents.shape[1], latents.shape[2]))
        return latents

    def _resolve_timestep(self, timestep, sigmas):
        ctx = self.ctx
        t = timestep
        idx = None
        if not isinstance(t, bool):
            if isinstance(t, int):
                idx = t
            else:
                try:
                    import numpy as np
                    if isinstance(t, np.integer):
                        idx = int(t)
                except ImportError:
                    pass
                if idx is None and ctx.is_tensor(t) and ctx.is_integer_dtype_tensor(t):
                    idx = int(t.item())
        if idx is not None:
            if sigmas is None:
                raise ValueError("ZImageDiTMLX requires sigmas when timestep is an integer index")
            sigma_t = ctx.reshape(sigmas[idx], (1,))
            t = ctx.ones_like(sigma_t) + ctx.mul(sigma_t, -1.0)
        else:
            if not ctx.is_tensor(t):
                t = ctx.array(t, dtype=ctx.float32())
            if hasattr(t, "ndim") and t.ndim == 0:
                t = ctx.reshape(t, (1,))
        return t

    def _reshape_output(self, output, input_shape, input_ndim):
        ctx = self.ctx
        if input_ndim == 4:
            output = output[:, 0, :, :]
            return ctx.reshape(output, (1, output.shape[0], output.shape[1], output.shape[2]))
        if input_ndim == 5:
            return ctx.reshape(output, input_shape)
        return ctx.reshape(output, (1,) + output.shape)

    def _forward_from_caches(
        self,
        latents,
        t,
        cap_emb,
        cap_freqs,
        x_freqs,
        x_pad_mask,
        x_len,
        image_size,
        *,
        use_compile: bool = False,
        control_context=None,
        control_context_scale: float = 1.0,
        teacache_branch: str = "default",
    ):
        if use_compile and self._compiled_forward is not None and control_context is None:
            tokens = self._compiled_forward(latents, t, cap_emb, cap_freqs, x_freqs, x_pad_mask, x_len)
        else:
            tokens = self._forward_cached_compute(
                latents,
                t,
                cap_emb,
                cap_freqs,
                x_freqs,
                x_pad_mask,
                x_len,
                control_context=control_context,
                control_context_scale=control_context_scale,
                teacache_branch=teacache_branch,
            )
        return self._unpatchify(tokens, image_size)

    def _forward_cached_compute(
        self,
        latents,
        t,
        cap_emb,
        cap_freqs,
        x_freqs,
        x_pad_mask,
        x_len,
        *,
        control_context=None,
        control_context_scale: float = 1.0,
        teacache_branch: str = "default",
    ):
        ctx = self.ctx
        act = self._act_dtype
        t_emb = self.t_embedder.forward(ctx.mul(ctx.cast(t, ctx.float32()), self.t_scale))

        pH = pW = self.patch_size
        pF = self.f_patch_size
        C, F, H, W = latents.shape
        F_tok, H_tok, W_tok = F // pF, H // pH, W // pW
        img = ctx.reshape(latents, (C, F_tok, pF, H_tok, pH, W_tok, pW))
        img = ctx.permute(img, (1, 3, 5, 2, 4, 6, 0))
        img = ctx.reshape(img, (F_tok * H_tok * W_tok, pF * pH * pW * C))
        img = _pad_tail_with_last(ctx, img, int(x_pad_mask.shape[0]) - int(img.shape[0]))
        x_emb = self.x_embedder(img)
        x_emb = ctx.cast(x_emb, act)
        x_emb = _apply_pad_token(ctx, x_emb, x_pad_mask, self.x_pad_token)
        x_emb = ctx.reshape(x_emb, (1, x_emb.shape[0], x_emb.shape[1]))

        refiner_hints = None
        refined_control = None
        layer_hints = None
        ctrl = self._control
        c_scale = float(control_context_scale or self._control_context_scale)
        if ctrl is not None and control_context is not None:
            if control_context.ndim == 5:
                control_n = control_context[0]
            elif control_context.ndim == 4:
                control_n = control_context
            else:
                raise RuntimeError(f"zimage_control_context expected 4D or 5D, got {control_context.shape}")
            control_embed = ctrl.embed_control_context(
                control_n,
                t_emb,
                x_pad_token=self.x_pad_token,
                img_pad_mask=x_pad_mask,
                img_freqs=x_freqs,
            )
            refiner_hints, refined_control = ctrl.forward_control_refiner(x_emb, control_embed, x_freqs, t_emb)

        from backend.engine.families.z_image.control_mlx import apply_control_hint

        for layer_idx, layer in enumerate(self.noise_refiner):
            hint = ctrl.hint_for_refiner_layer(layer_idx, refiner_hints) if ctrl else None
            x_emb = layer.forward(x_emb, None, x_freqs, t_emb)
            x_emb = apply_control_hint(x_emb, hint, c_scale, ctx)

        unified = ctx.concat([x_emb, cap_emb], axis=1)
        unified_freqs = ctx.concat([x_freqs, cap_freqs], axis=0)

        if ctrl is not None and refined_control is not None:
            layer_hints = ctrl.forward_control_layers(unified, refined_control, cap_emb, unified_freqs, t_emb)

        unified_before = unified
        step_idx = self._lemica_step_counter
        use_lemica_skip = (
            self._lemica_bool_list is not None
            and step_idx < len(self._lemica_bool_list)
            and not self._lemica_bool_list[step_idx]
            and self._lemica_previous_residual is not None
        )
        use_teacache_skip = False
        cache_branch = str(teacache_branch or "default")
        if self._step_cache is not None and self._step_cache.enabled:
            signal = unified_before
            mod_in = signal[0:1] if int(signal.shape[0]) > 1 else signal
            decision = self._step_cache.gate(mod_in, branch=cache_branch)
            cached_residual = self._step_cache.branch_cached_residual(cache_branch)
            if self._step_cache.should_skip(decision, branch=cache_branch):
                unified = unified_before + cached_residual
                use_teacache_skip = True
            elif decision is not None and decision.should_update_cache:
                self._step_cache.store_branch_mod_input(cache_branch, mod_in)
            elif (
                decision is not None
                and decision.should_compute
                and decision.rel_l1 is not None
            ):
                self._step_cache.store_branch_mod_input(cache_branch, mod_in)

        if use_lemica_skip:
            unified = unified + self._lemica_previous_residual
        elif use_teacache_skip:
            pass
        else:
            for layer_idx, layer in enumerate(self.layers):
                hint = ctrl.hint_for_main_layer(layer_idx, layer_hints) if ctrl else None
                unified = layer.forward(unified, None, unified_freqs, t_emb)
                unified = apply_control_hint(unified, hint, c_scale, ctx)
            if self._lemica_bool_list is not None:
                self._lemica_previous_residual = unified - unified_before
                if not self._use_mlx_compile_run:
                    ctx.eval(self._lemica_previous_residual)
            elif self._step_cache is not None and self._step_cache.enabled:
                residual = unified - unified_before
                self._step_cache.store_branch_residual(cache_branch, residual)
                if not self._use_mlx_compile_run:
                    ctx.eval(residual)
        if self._lemica_bool_list is not None:
            self._lemica_step_counter += 1

        unified = self.final_layer.forward(unified, t_emb)
        return unified[0, :x_len]

    def _encode_cap_branch(self, cap_feats):
        """Precompute caption embed + context refiner (constant across denoise steps)."""
        ctx = self.ctx
        cap_ori_len = cap_feats.shape[0]
        cap_pad_len = _pad_len_to_multiple(cap_ori_len, 32)
        cap_padded_len = cap_ori_len + cap_pad_len

        cap_feats = _pad_tail_with_last(ctx, cap_feats, cap_pad_len)

        cap_pos_ids = self._coord_grid((cap_padded_len, 1, 1), (1, 0, 0))
        cap_pos_ids = ctx.reshape(cap_pos_ids, (-1, 3))
        cap_pad_mask = _build_tail_pad_mask(ctx, int(cap_ori_len), int(cap_pad_len))

        cap_emb = self.cap_norm(cap_feats)
        cap_emb = self.cap_embedder(cap_emb)
        cap_emb = _apply_pad_token(ctx, cap_emb, cap_pad_mask, self.cap_pad_token)
        cap_freqs_cis = self.rope.forward(cap_pos_ids)
        cap_emb = ctx.reshape(cap_emb, (1, cap_emb.shape[0], cap_emb.shape[1]))

        for layer in self.context_refiner:
            cap_emb = layer.forward(cap_emb, None, cap_freqs_cis)

        return cap_emb, cap_freqs_cis, cap_padded_len

    def _encode_geo_cache(self, latents, cap_padded_len: int):
        """Precompute image RoPE / pad metadata (constant across denoise steps)."""
        ctx = self.ctx
        pH = pW = self.patch_size
        pF = self.f_patch_size
        _C, F, H, W = latents.shape
        image_size = (F, H, W)
        F_tok, H_tok, W_tok = F // pF, H // pH, W // pW
        img_ori_len = F_tok * H_tok * W_tok
        img_pad_len = _pad_len_to_multiple(img_ori_len, 32)
        x_len = img_ori_len + img_pad_len

        img_pos_ids = self._coord_grid((F_tok, H_tok, W_tok), (cap_padded_len + 1, 0, 0))
        img_pos_ids = ctx.reshape(img_pos_ids, (-1, 3))
        if img_pad_len > 0:
            img_pos_ids = ctx.concat([img_pos_ids, ctx.zeros((img_pad_len, 3), dtype=ctx.int32())], axis=0)

        x_freqs_cis = self.rope.forward(img_pos_ids)
        x_pad_mask = _build_tail_pad_mask(ctx, int(img_ori_len), int(img_pad_len))

        return x_freqs_cis, x_pad_mask, x_len, image_size, img_ori_len

    def _forward_compute(
        self,
        latents,
        t,
        cap_feats,
        *,
        control_context=None,
        control_context_scale: float = 1.0,
    ):
        cap_emb, cap_freqs, cap_pad_len = self._encode_cap_branch(cap_feats)
        x_freqs, x_pad_mask, x_len, image_size, _ = self._encode_geo_cache(latents, cap_pad_len)
        tokens = self._forward_cached_compute(
            latents,
            t,
            cap_emb,
            cap_freqs,
            x_freqs,
            x_pad_mask,
            x_len,
            control_context=control_context,
            control_context_scale=control_context_scale,
        )
        return self._unpatchify(tokens, image_size)

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
        cap_pad_len = _pad_len_to_multiple(cap_ori_len, 32)
        cap_pos_ids = self._coord_grid((cap_ori_len + cap_pad_len, 1, 1), (1, 0, 0))
        cap_pos_ids = ctx.reshape(cap_pos_ids, (-1, 3))
        cap_pad_mask = _build_tail_pad_mask(ctx, int(cap_ori_len), int(cap_pad_len))
        cap_padded = _pad_tail_with_last(ctx, cap_feats, cap_pad_len)

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
        img_pad_len = _pad_len_to_multiple(img_ori_len, 32)
        img_pos_ids = self._coord_grid((F_tok, H_tok, W_tok),
                                       (cap_ori_len + cap_pad_len + 1, 0, 0))
        img_pos_ids = ctx.reshape(img_pos_ids, (-1, 3))

        if img_pad_len > 0:
            img_pos_ids = ctx.concat([img_pos_ids, ctx.zeros((img_pad_len, 3), dtype=ctx.int32())], axis=0)
        img = _pad_tail_with_last(ctx, img, img_pad_len)

        img_pad_mask = _build_tail_pad_mask(ctx, int(img_ori_len), int(img_pad_len))

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

