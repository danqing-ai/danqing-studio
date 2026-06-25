"""Wan DiT — PyTorch (CUDA) implementation (ported from ``WanModelMLX``)."""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from backend.engine.common.model.base import TransformerBase
from backend.engine.common.ops.attention import scaled_dot_product_attention_bhsd_torch
from backend.engine.common.ops.cfg_batch import (
    TEXT_KEYS_MINIMAL,
    predict_noise_cfg_batched,
)
from backend.engine.common.ops.norm import (
    apply_scale_shift,
    unpack_modulation_2table,
    unpack_modulation_6table,
)
from backend.engine.config.model_configs import WanConfig
from backend.engine.runtime.cuda import CudaContext


# ---------------------------------------------------------------------------
# Factorized 3D RoPE helpers (torch)
# ---------------------------------------------------------------------------

def _factorized_rope_dims(half_d: int) -> tuple[int, int, int]:
    d_t = half_d - 2 * (half_d // 3)
    d_h = half_d // 3
    d_w = half_d // 3
    return d_t, d_h, d_w


def _factorized_rope_params_torch(max_seq_len: int, dim: int, theta: float = 10000.0) -> torch.Tensor:
    """Precompute factorized RoPE frequencies as ``[L, dim//2, 2]`` (cos, sin)."""
    if dim % 2 != 0:
        raise ValueError("rope dim must be even")
    import numpy as np
    freqs = (
        np.arange(max_seq_len, dtype=np.float64)[:, None]
        * (1.0 / np.power(theta, np.arange(0, dim, 2, dtype=np.float64) / dim)[None, :])
    )
    return torch.from_numpy(np.stack([np.cos(freqs), np.sin(freqs)], axis=-1).astype(np.float32))


def _factorized_rope_concat_params_torch(max_seq_len: int, dims: list[int], theta: float = 10000.0) -> torch.Tensor:
    parts = [_factorized_rope_params_torch(max_seq_len, int(d), theta) for d in dims]
    return torch.cat(parts, dim=1)


def _factorized_rope_precompute_cos_sin_torch(
    grid_sizes: list[tuple[int, int, int]],
    freqs: torch.Tensor,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    f, h, w = grid_sizes[0]
    seq_len = f * h * w
    half_d = freqs.shape[1]
    d_t, d_h, d_w = _factorized_rope_dims(half_d)

    freqs_t = freqs[:, :d_t]
    freqs_h = freqs[:, d_t : d_t + d_h]
    freqs_w = freqs[:, d_t + d_h : d_t + d_h + d_w]

    ft = freqs_t[:f].view(f, 1, 1, d_t, 2).expand(f, h, w, d_t, 2)
    fh = freqs_h[:h].view(1, h, 1, d_h, 2).expand(f, h, w, d_h, 2)
    fw = freqs_w[:w].view(1, 1, w, d_w, 2).expand(f, h, w, d_w, 2)

    freqs_i = torch.cat([ft, fh, fw], dim=3).reshape(seq_len, 1, half_d, 2)
    return freqs_i[..., 0].to(dtype=dtype, device=device), freqs_i[..., 1].to(dtype=dtype, device=device)


def _factorized_rope_apply_torch(
    x: torch.Tensor,
    grid_sizes: list[tuple[int, int, int]],
    freqs: torch.Tensor,
    precomputed_cos_sin: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Apply factorized 3-way RoPE on ``[B, L, num_heads, head_dim]``."""
    b, s, _n, d = x.shape
    half_d = d // 2

    if precomputed_cos_sin is not None:
        cos_f, sin_f = precomputed_cos_sin
        f0, h0, w0 = grid_sizes[0]
        seq_len = f0 * h0 * w0
        all_same = all(grid_sizes[i] == grid_sizes[0] for i in range(1, b)) if b > 1 else True

        if all_same:
            x_seq = x[:, :seq_len].reshape(b, seq_len, -1, half_d, 2)
            x_real = x_seq[..., 0]
            x_imag = x_seq[..., 1]
            out_real = x_real * cos_f - x_imag * sin_f
            out_imag = x_real * sin_f + x_imag * cos_f
            x_rotated = torch.stack([out_real, out_imag], dim=-1).reshape(b, seq_len, -1, d)
            if seq_len < s:
                x_rotated = torch.cat([x_rotated, x[:, seq_len:]], dim=1)
            return x_rotated

        outputs = []
        for i in range(b):
            f, h, w = grid_sizes[i]
            sl = f * h * w
            x_i = x[i, :sl].reshape(sl, -1, half_d, 2)
            x_real = x_i[..., 0]
            x_imag = x_i[..., 1]
            out_real = x_real * cos_f - x_imag * sin_f
            out_imag = x_real * sin_f + x_imag * cos_f
            x_rotated = torch.stack([out_real, out_imag], dim=-1).reshape(sl, -1, d)
            if sl < s:
                x_rotated = torch.cat([x_rotated, x[i, sl:]], dim=0)
            outputs.append(x_rotated)
        return torch.stack(outputs)

    d_t, d_h, d_w = _factorized_rope_dims(half_d)
    freqs_t = freqs[:, :d_t]
    freqs_h = freqs[:, d_t : d_t + d_h]
    freqs_w = freqs[:, d_t + d_h : d_t + d_h + d_w]

    outputs = []
    for i in range(b):
        f, h, w = grid_sizes[i]
        seq_len = f * h * w
        x_i = x[i, :seq_len].reshape(seq_len, -1, half_d, 2)

        ft = freqs_t[:f].view(f, 1, 1, d_t, 2).expand(f, h, w, d_t, 2)
        fh = freqs_h[:h].view(1, h, 1, d_h, 2).expand(f, h, w, d_h, 2)
        fw = freqs_w[:w].view(1, 1, w, d_w, 2).expand(f, h, w, d_w, 2)
        freqs_i = torch.cat([ft, fh, fw], dim=3).reshape(seq_len, 1, half_d, 2)
        cos_f = freqs_i[..., 0].to(x.device, x.dtype)
        sin_f = freqs_i[..., 1].to(x.device, x.dtype)

        x_real = x_i[..., 0]
        x_imag = x_i[..., 1]
        out_real = x_real * cos_f - x_imag * sin_f
        out_imag = x_real * sin_f + x_imag * cos_f
        x_rotated = torch.stack([out_real, out_imag], dim=-1).reshape(seq_len, -1, d)
        if seq_len < s:
            x_rotated = torch.cat([x_rotated, x[i, seq_len:]], dim=0)
        outputs.append(x_rotated)

    return torch.stack(outputs)


# ---------------------------------------------------------------------------
# Pad ragged sequences (torch)
# ---------------------------------------------------------------------------

def _pad_ragged_2d_sequences_torch(
    sequences: list[torch.Tensor],
    target_len: int | None = None,
    dtype: torch.dtype | None = None,
    pad_value: float = 0.0,
) -> torch.Tensor:
    if not sequences:
        raise RuntimeError("pad_ragged_2d_sequences requires non-empty sequences")
    max_len = max(int(s.shape[0]) for s in sequences)
    t = target_len if target_len is not None else max_len
    padded: list[torch.Tensor] = []
    for s in sequences:
        cur = int(s.shape[0])
        dim = int(s.shape[1])
        use = s[:t] if cur > t else s
        if cur < t:
            pad = torch.full((t - cur, dim), float(pad_value), dtype=s.dtype, device=s.device)
            use = torch.cat([use, pad], dim=0)
        padded.append(use)
    out = torch.stack(padded, dim=0)
    return out.to(dtype) if dtype is not None else out


# ---------------------------------------------------------------------------
# Attention mask (torch)
# ---------------------------------------------------------------------------

def _build_key_padding_mask_from_lengths_torch(
    lengths: torch.Tensor,
    seq_len: int,
    dtype: torch.dtype,
    device: torch.device,
    neg_value: float = -1e9,
) -> torch.Tensor:
    b = int(lengths.shape[0])
    positions = torch.arange(seq_len, dtype=torch.int32, device=device)
    valid_k = positions.view(1, 1, 1, seq_len) < lengths.view(b, 1, 1, 1)
    shape = (b, 1, seq_len, seq_len)
    neg = torch.full(shape, float(neg_value), dtype=dtype, device=device)
    zeros = torch.zeros(shape, dtype=dtype, device=device)
    return torch.where(valid_k, zeros, neg)


# ---------------------------------------------------------------------------
# Sinusoidal embedding (torch)
# ---------------------------------------------------------------------------

def _sinusoidal_embedding_1d_torch(
    dim: int,
    position: torch.Tensor,
    base: float = 10000.0,
) -> torch.Tensor:
    if dim % 2 != 0:
        raise ValueError("dim must be even")
    half = dim // 2
    pos = position.float()
    freqs = torch.pow(
        torch.tensor(float(base), dtype=torch.float32, device=pos.device),
        -torch.arange(half, dtype=torch.float32, device=pos.device) / half,
    )
    sinusoid = torch.outer(pos.flatten(), freqs)
    return torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=-1)


# ---------------------------------------------------------------------------
# RMSNorm (torch)
# ---------------------------------------------------------------------------

class _RMSNormTorch(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight.float() * norm).to(dtype)


# ---------------------------------------------------------------------------
# Wan LayerNorm (fp32 compute)
# ---------------------------------------------------------------------------

class _WanLayerNormTorch(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = False):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=elementwise_affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        out = self.norm(x.float())
        return out.to(dtype)


# ---------------------------------------------------------------------------
# Attention modules
# ---------------------------------------------------------------------------

class _WanSelfAttentionTorch(nn.Module):
    def __init__(self, dim: int, num_heads: int, qk_norm: bool, eps: float, device: torch.device):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=True)
        self.k = nn.Linear(dim, dim, bias=True)
        self.v = nn.Linear(dim, dim, bias=True)
        self.o = nn.Linear(dim, dim, bias=True)
        self.norm_q = _RMSNormTorch(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = _RMSNormTorch(dim, eps=eps) if qk_norm else nn.Identity()
        self._device = device

    def forward(
        self,
        x: torch.Tensor,
        grid_sizes: list[tuple[int, int, int]],
        freqs: torch.Tensor,
        rope_cos_sin: tuple[torch.Tensor, torch.Tensor] | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b, s, n, d = x.shape[0], x.shape[1], self.num_heads, self.head_dim
        w_dtype = self.q.weight.dtype
        x_w = x.to(w_dtype)
        q = self.norm_q(self.q(x_w)).reshape(b, s, n, d)
        k = self.norm_k(self.k(x_w)).reshape(b, s, n, d)
        v = self.v(x_w).reshape(b, s, n, d)
        q = _factorized_rope_apply_torch(q.float(), grid_sizes, freqs, precomputed_cos_sin=rope_cos_sin).to(w_dtype)
        k = _factorized_rope_apply_torch(k.float(), grid_sizes, freqs, precomputed_cos_sin=rope_cos_sin).to(w_dtype)

        # [B, S, H, D] -> [B, H, S, D]
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        out = scaled_dot_product_attention_bhsd_torch(q, k, v, mask=attn_mask, scale=self.scale)
        out = out.permute(0, 2, 1, 3).reshape(b, s, -1)
        return self.o(out)


class _WanCrossAttentionTorch(_WanSelfAttentionTorch):
    def cross_kv(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Precompute K/V for fixed text context (reused across denoise steps)."""
        b, n, d = context.shape[0], self.num_heads, self.head_dim
        k = self.norm_k(self.k(context)).reshape(b, -1, n, d)
        v = self.v(context).reshape(b, -1, n, d)
        return k, v

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        cross_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        b, n, d = x.shape[0], self.num_heads, self.head_dim
        q = self.norm_q(self.q(x)).reshape(b, -1, n, d)
        if cross_kv is not None:
            k, v = cross_kv
        else:
            if context is None:
                raise RuntimeError("WanCrossAttention requires context or cross_kv")
            k, v = self.cross_kv(context)

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        out = scaled_dot_product_attention_bhsd_torch(q, k, v, scale=self.scale)
        return self.o(out.permute(0, 2, 1, 3).reshape(b, -1, n * d))


# ---------------------------------------------------------------------------
# FFN
# ---------------------------------------------------------------------------

class _WanFFNTorch(nn.Module):
    def __init__(self, dim: int, ffn_dim: int):
        super().__init__()
        self.layer_0 = nn.Linear(dim, ffn_dim, bias=True)
        self.layer_2 = nn.Linear(ffn_dim, dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer_2(F.gelu(self.layer_0(x), approximate="tanh"))


# ---------------------------------------------------------------------------
# Attention block
# ---------------------------------------------------------------------------

class _WanAttentionBlockTorch(nn.Module):
    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        qk_norm: bool,
        cross_attn_norm: bool,
        eps: float,
        device: torch.device,
    ):
        super().__init__()
        self.norm1 = _WanLayerNormTorch(dim, eps)
        self.self_attn = _WanSelfAttentionTorch(dim, num_heads, qk_norm, eps, device)
        self.norm3 = _WanLayerNormTorch(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = _WanCrossAttentionTorch(dim, num_heads, qk_norm, eps, device)
        self.norm2 = _WanLayerNormTorch(dim, eps)
        self.ffn = _WanFFNTorch(dim, ffn_dim)
        self.modulation = nn.Parameter(torch.zeros(1, 6, dim))

    def forward(
        self,
        x: torch.Tensor,
        e: torch.Tensor,
        grid_sizes: list[tuple[int, int, int]],
        freqs: torch.Tensor,
        context: torch.Tensor,
        cross_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        rope_cos_sin: tuple[torch.Tensor, torch.Tensor] | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        mod = self.modulation.float() + e.float()
        e0, e1, e2, e3, e4, e5 = unpack_modulation_6table(mod)
        y = self.self_attn(
            apply_scale_shift(self.norm1(x).float(), e1, e0, add_one=True),
            grid_sizes,
            freqs,
            rope_cos_sin=rope_cos_sin,
            attn_mask=attn_mask,
        )
        x = x + y * e2
        x = x + self.cross_attn(self.norm3(x), context, cross_kv=cross_kv)
        y = self.ffn(
            apply_scale_shift(self.norm2(x).float(), e4, e3, add_one=True)
        )
        return x + y * e5


# ---------------------------------------------------------------------------
# Wan Model (CUDA)
# ---------------------------------------------------------------------------

class WanModelCUDA(TransformerBase):
    """Wan video DiT — ``VideoPipeline`` contract: latents ``[B,C,T,H,W]``."""

    def __init__(self, config: WanConfig, ctx: CudaContext, num_frames: int = 81):
        super().__init__()
        self.config = config
        self.ctx = ctx
        self._num_frames = num_frames
        pt, ph, pw = config.patch_size
        patch_dim = int(config.dim_in) * int(pt) * int(ph) * int(pw)
        device = ctx.device

        self.patch_embedding = nn.Linear(patch_dim, config.dim, bias=True).to(device)
        self._patch_size = config.patch_size
        self.text_embedding = nn.ModuleList([
            nn.Linear(config.text_dim, config.dim, bias=True).to(device),
            nn.Linear(config.dim, config.dim, bias=True).to(device),
        ])
        self.time_embedding = nn.ModuleList([
            nn.Linear(config.freq_dim, config.dim, bias=True).to(device),
            nn.Linear(config.dim, config.dim, bias=True).to(device),
        ])
        self.time_projection = nn.Linear(config.dim, config.dim * 6, bias=True).to(device)
        self.blocks = nn.ModuleList([
            _WanAttentionBlockTorch(
                config.dim, config.ffn_dim, config.num_heads,
                config.qk_norm, config.cross_attn_norm, config.eps, device,
            )
            for _ in range(config.depth)
        ])
        self.head_norm = _WanLayerNormTorch(config.dim, config.eps)
        self.head = nn.Linear(config.dim, ph * pw * pt * config.dim_out, bias=True).to(device)
        self.head_modulation = nn.Parameter(torch.zeros(1, 2, config.dim))
        self.patch_size = self._patch_size
        self.text_len = config.text_len
        self.out_dim = config.dim_out

        d = config.dim // config.num_heads
        self._freqs = _factorized_rope_concat_params_torch(
            1024,
            [d - 4 * (d // 6), 2 * (d // 6), 2 * (d // 6)],
        ).to(device)

        self._rope_cos_sin: tuple[torch.Tensor, torch.Tensor] | None = None
        self._rope_grid_key: tuple[int, int, int] | None = None
        self._i2v_cond: Any | None = None
        self._i2v_mask: Any | None = None
        self._text_cache_key: tuple[int, ...] | None = None
        self._cached_context: torch.Tensor | None = None
        self._cached_cross_kv: list[tuple[torch.Tensor, torch.Tensor]] | None = None
        self._build_param_map()

    def invalidate_text_cache(self) -> None:
        self._text_cache_key = None
        self._cached_context = None
        self._cached_cross_kv = None
        self._rope_cos_sin = None
        self._rope_grid_key = None
        # I2V conditioning is per-run; clear so cached models do not leak into T2V.
        self._i2v_cond = None
        self._i2v_mask = None

    def after_load_weights(self, bundle_root=None) -> None:
        super().after_load_weights(bundle_root)
        self.invalidate_text_cache()

    def sanitize(self, weights: dict) -> dict:
        from backend.engine.families.wan.weights import remap_wan_weights
        return remap_wan_weights(weights)

    def _build_param_map(self) -> None:
        self._param_map = {}
        self._param_map["patch_embedding.weight"] = self.patch_embedding.weight
        self._param_map["patch_embedding.bias"] = self.patch_embedding.bias
        self._param_map["text_embedding.0.weight"] = self.text_embedding[0].weight
        self._param_map["text_embedding.0.bias"] = self.text_embedding[0].bias
        self._param_map["text_embedding.2.weight"] = self.text_embedding[1].weight
        self._param_map["text_embedding.2.bias"] = self.text_embedding[1].bias
        self._param_map["time_embedding.0.weight"] = self.time_embedding[0].weight
        self._param_map["time_embedding.0.bias"] = self.time_embedding[0].bias
        self._param_map["time_embedding.2.weight"] = self.time_embedding[1].weight
        self._param_map["time_embedding.2.bias"] = self.time_embedding[1].bias
        self._param_map["time_projection.1.weight"] = self.time_projection.weight
        self._param_map["time_projection.1.bias"] = self.time_projection.bias
        self._param_map["head.head.weight"] = self.head.weight
        self._param_map["head.head.bias"] = self.head.bias
        self._param_map["head.modulation"] = self.head_modulation
        for i, blk in enumerate(self.blocks):
            prefix = f"blocks.{i}"
            self._param_map[f"{prefix}.modulation"] = blk.modulation
            for part in ("self_attn", "cross_attn"):
                attn = getattr(blk, part)
                for w in ("q", "k", "v", "o"):
                    lin = getattr(attn, w)
                    self._param_map[f"{prefix}.{part}.{w}.weight"] = lin.weight
                    self._param_map[f"{prefix}.{part}.{w}.bias"] = lin.bias
                if hasattr(attn, "norm_q") and hasattr(attn.norm_q, "weight"):
                    self._param_map[f"{prefix}.{part}.norm_q.weight"] = attn.norm_q.weight
                if hasattr(attn, "norm_k") and hasattr(attn.norm_k, "weight"):
                    self._param_map[f"{prefix}.{part}.norm_k.weight"] = attn.norm_k.weight
            if hasattr(blk.norm3, "weight"):
                self._param_map[f"{prefix}.norm3.weight"] = blk.norm3.weight
                self._param_map[f"{prefix}.norm3.bias"] = blk.norm3.bias
            self._param_map[f"{prefix}.ffn.layer_0.weight"] = blk.ffn.layer_0.weight
            self._param_map[f"{prefix}.ffn.layer_0.bias"] = blk.ffn.layer_0.bias
            self._param_map[f"{prefix}.ffn.layer_2.weight"] = blk.ffn.layer_2.weight
            self._param_map[f"{prefix}.ffn.layer_2.bias"] = blk.ffn.layer_2.bias

    def set_i2v_state(self, cond: Any | None, mask: Any | None) -> None:
        self._i2v_cond = cond
        self._i2v_mask = mask

    def reblend_i2v_latents(self, latents: Any) -> Any:
        if self._i2v_cond is None or self._i2v_mask is None:
            return latents
        from .conditioning import prepare_ti2v_i2v_latents
        return prepare_ti2v_i2v_latents(self.ctx, latents, self._i2v_cond, self._i2v_mask)

    def forward(
        self,
        latents: Any,
        timestep: Any,
        txt_embeds: Any | None = None,
        *,
        timestep_per_token: Any | None = None,
        seq_len: int | None = None,
        **_: Any,
    ) -> Any:
        if txt_embeds is None:
            raise RuntimeError("Wan requires T5 embeddings (`txt_embeds`).")
        key = tuple(int(x) for x in txt_embeds.shape) + (id(txt_embeds),)
        if not (
            self._text_cache_key == key
            and self._cached_context is not None
            and self._cached_cross_kv is not None
        ):
            cfg = self.config
            batch = _pad_ragged_2d_sequences_torch(
                [txt_embeds[i] for i in range(int(txt_embeds.shape[0]))],
                target_len=cfg.text_len,
                dtype=txt_embeds.dtype,
                pad_value=0.0,
            )
            context = self.text_embedding[1](F.gelu(self.text_embedding[0](batch), approximate="tanh"))
            self._cached_context = context
            self._cached_cross_kv = [blk.cross_attn.cross_kv(context) for blk in self.blocks]
            self._text_cache_key = key
        return self._forward_compute(latents, timestep, timestep_per_token, seq_len)

    def _forward_compute(
        self,
        latents: Any,
        timestep: Any,
        timestep_per_token: Any | None,
        seq_len: int | None,
    ) -> Any:
        ctx = self.ctx
        context = self._cached_context
        cross_kv_list = self._cached_cross_kv
        if context is None or cross_kv_list is None:
            raise RuntimeError("Wan: text context cache missing; call forward() with txt_embeds first.")
        if latents.ndim != 5:
            raise RuntimeError(f"Wan expects latents [B,C,T,H,W], got {latents.shape}")

        b = int(latents.shape[0])
        per_token = timestep_per_token is not None
        patches = []
        grid_sizes_list: list[tuple[int, int, int]] = []
        seq_lens_list: list[int] = []
        pt, ph, pw = self._patch_size
        for i in range(b):
            sample = latents[i]
            c, f, h, w = (int(sample.shape[j]) for j in range(4))
            if f % pt != 0 or h % ph != 0 or w % pw != 0:
                raise RuntimeError(
                    f"Wan latent shape [C,T,H,W]=[{c},{f},{h},{w}] is not divisible by "
                    f"patch_size={self._patch_size} (need T%{pt}==0, H%{ph}==0, W%{pw}==0). "
                    f"Snap pixel size to multiples of vae_scale×patch "
                    f"(e.g. 480×704 not 480×720 for vae_scale=16)."
                )
            f_out, h_out, w_out = f // pt, h // ph, w // pw
            patch = sample.reshape(c, f_out, pt, h_out, ph, w_out, pw)
            patch = patch.permute(1, 3, 5, 0, 2, 4, 6)
            flat = patch.reshape(f_out * h_out * w_out, -1)
            flat = self.patch_embedding(flat).to(self.patch_embedding.weight.dtype)
            grid = (f_out, h_out, w_out)
            patches.append(flat)
            grid_sizes_list.append(grid)
            seq_lens_list.append(int(flat.shape[0]))
        if seq_len is None:
            seq_len = max(seq_lens_list)
        x = _pad_ragged_2d_sequences_torch(patches, target_len=int(seq_len))

        t_in = timestep_per_token if per_token else timestep
        if t_in is None:
            raise RuntimeError("Wan forward requires timestep or timestep_per_token")
        if getattr(t_in, "ndim", 0) == 0:
            t_in = t_in.reshape(1)
        if int(getattr(t_in, "shape", (1,))[0]) == 1 and b > 1:
            t_in = t_in.repeat(b)

        cfg = self.config
        ndim = getattr(t_in, "ndim", 0)
        if per_token:
            if ndim == 0:
                raise RuntimeError("Wan per-token timesteps require a 2D tensor [B, L]")
            if ndim == 1:
                t_in = t_in.reshape(1, -1)
            bt = int(t_in.shape[0])
            seq_tok = int(t_in.shape[1])
            flat_t = t_in.reshape(-1)
            emb = _sinusoidal_embedding_1d_torch(cfg.freq_dim, flat_t)
            emb = emb.reshape(bt, seq_tok, cfg.freq_dim).float()
            e = self.time_embedding[1](F.silu(self.time_embedding[0](emb)))
            e0 = self.time_projection(F.silu(e))
            e0 = e0.reshape(bt, seq_tok, 6, cfg.dim)
        else:
            if ndim == 0:
                t_b = t_in.reshape(1)
            elif ndim == 1:
                t_b = t_in
            elif ndim == 2 and int(t_in.shape[1]) == 1:
                t_b = t_in.reshape(-1)
            else:
                raise RuntimeError(
                    f"Wan scalar timestep expected [B] or scalar, got shape {getattr(t_in, 'shape', ())}"
                )
            t_b = t_b.float()
            emb = _sinusoidal_embedding_1d_torch(cfg.freq_dim, t_b).float()
            e = self.time_embedding[1](F.silu(self.time_embedding[0](emb)))
            e0 = self.time_projection(F.silu(e))
            e0 = e0.reshape(int(t_b.shape[0]), 1, 6, cfg.dim)

        freqs = self._freqs
        rope_key = grid_sizes_list[0]
        if self._rope_grid_key == rope_key and self._rope_cos_sin is not None:
            rope_cos_sin = self._rope_cos_sin
        else:
            w_dtype = self.patch_embedding.weight.dtype
            rope_cos_sin = _factorized_rope_precompute_cos_sin_torch(
                grid_sizes_list, self._freqs, dtype=w_dtype, device=self.ctx.device,
            )
            self._rope_grid_key = rope_key
            self._rope_cos_sin = rope_cos_sin

        if all(sl >= seq_len for sl in seq_lens_list):
            attn_mask = None
        else:
            lens = torch.tensor(seq_lens_list[:b], dtype=torch.int32, device=self.ctx.device)
            attn_mask = _build_key_padding_mask_from_lengths_torch(
                lens, seq_len, self.patch_embedding.weight.dtype, self.ctx.device,
            )

        for blk, cross_kv in zip(self.blocks, cross_kv_list):
            x = blk(
                x,
                e0,
                grid_sizes_list,
                freqs,
                context,
                cross_kv=cross_kv,
                rope_cos_sin=rope_cos_sin,
                attn_mask=attn_mask,
            )

        if e.ndim == 2:
            e_h = e.unsqueeze(1)
        else:
            e_h = e
        mod = self.head_modulation.float()[None, None, :, :] + e_h.float()[:, :, None, :]
        e_shift, e_scale = unpack_modulation_2table(mod)
        x = self.head_norm(x).float()
        x = self.head(apply_scale_shift(x, e_scale, e_shift, add_one=True))
        c = self.out_dim
        pt, ph, pw = self.patch_size
        outs = []
        for bi in range(int(x.shape[0])):
            u = x[bi]
            f, h, w = grid_sizes_list[bi]
            tok = u[: f * h * w].reshape(f, h, w, pt, ph, pw, c)
            tok = torch.einsum("fhwpqrc->cfphqwr", tok)
            outs.append(tok.reshape(c, f * pt, h * ph, w * pw))
        return torch.stack(outs, dim=0)

    def predict_noise_cfg(
        self,
        latents_in: Any,
        t: Any,
        *,
        guidance: float,
        pos_kwargs: dict[str, Any],
        neg_kwargs: dict[str, Any],
        cfg_renorm: bool = False,
        cfg_renorm_min: float = 0.0,
    ) -> Any:
        return predict_noise_cfg_batched(
            self.forward,
            self.ctx,
            latents_in,
            t,
            guidance=guidance,
            pos_kwargs=pos_kwargs,
            neg_kwargs=neg_kwargs,
            text_keys=TEXT_KEYS_MINIMAL,
            combine_cfg_noise=self.combine_cfg_noise,
            refine_cfg_noise=self.refine_cfg_noise,
            cfg_renorm=cfg_renorm,
            cfg_renorm_min=cfg_renorm_min,
        )
