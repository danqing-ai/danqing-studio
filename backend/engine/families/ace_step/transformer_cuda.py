"""
ACE-Step DiT decoder — PyTorch (CUDA) implementation.

Mirrors ``transformer_mlx.py`` architecture identically, using ``torch.nn``.
Convolution ops apply a channel-format permute because MLX uses NLC (channels-last)
while PyTorch uses NCL (channels-first).  All dense attention/MLP operations stay
in NLC to avoid unnecessary permutations.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from backend.engine.common._base import TransformerBase
from backend.engine.common.attention import (
    build_window_with_padding_bias_torch,
    repeat_kv_heads_torch,
    scaled_dot_product_attention_bhsd_torch,
    rotate_half_torch,
)
from backend.engine.common.embeddings import sinusoidal_timestep_proj
from backend.engine.common.norm import apply_scale_shift, unpack_modulation_6table
from backend.engine.runtime.cuda import CudaContext

logger = logging.getLogger(__name__)

_CUDA_CTX = CudaContext()

# ---------------------------------------------------------------------------
# Rotary helpers
# ---------------------------------------------------------------------------

def _apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    q_embed = (q * cos) + (rotate_half_torch(q) * sin)
    k_embed = (k * cos) + (rotate_half_torch(k) * sin)
    return q_embed, k_embed


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------

class _AceStepRotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_len: int = 32768, base: float = 1_000_000.0):
        super().__init__()
        self.head_dim = head_dim
        self.max_len = max_len
        self.base = base

        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        positions = torch.arange(max_len).float()
        freqs = positions[:, None] * inv_freq[None, :]
        freqs = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("_cos", freqs.cos(), persistent=False)
        self.register_buffer("_sin", freqs.sin(), persistent=False)

    def forward(self, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        cos = self._cos[:seq_len][None, None, :, :]
        sin = self._sin[:seq_len][None, None, :, :]
        return cos, sin


# ---------------------------------------------------------------------------
# Cross-attention KV cache
# ---------------------------------------------------------------------------

class _CrossAttentionCache:
    def __init__(self):
        self._keys: Dict[int, torch.Tensor] = {}
        self._values: Dict[int, torch.Tensor] = {}
        self._updated: set = set()

    def update(self, key: torch.Tensor, value: torch.Tensor, layer_idx: int):
        self._keys[layer_idx] = key
        self._values[layer_idx] = value
        self._updated.add(layer_idx)

    def is_updated(self, layer_idx: int) -> bool:
        return layer_idx in self._updated

    def get(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._keys[layer_idx], self._values[layer_idx]


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

class _AceStepRMSNorm(nn.Module):
    """Pure-PyTorch RMSNorm (compatible with torch < 2.4)."""
    def __init__(self, dims: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dims))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight.float() * norm).to(dtype)


class _SwiGLUMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class _Attention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        rms_norm_eps: float,
        attention_bias: bool,
        layer_idx: int,
        is_cross_attention: bool = False,
        sliding_window: Optional[int] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.num_kv_heads = num_key_value_heads
        self.head_dim = head_dim
        self.n_rep = num_attention_heads // num_key_value_heads
        self.scale = head_dim ** -0.5
        self.layer_idx = layer_idx
        self.is_cross_attention = is_cross_attention
        self.sliding_window = sliding_window

        self.q_proj = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=attention_bias)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=attention_bias)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=attention_bias)
        self.o_proj = nn.Linear(num_attention_heads * head_dim, hidden_size, bias=attention_bias)

        self.q_norm = _AceStepRMSNorm(head_dim, eps=rms_norm_eps)
        self.k_norm = _AceStepRMSNorm(head_dim, eps=rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_cos_sin: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        cache: Optional[_CrossAttentionCache] = None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        B, L, _ = hidden_states.shape

        q = self.q_proj(hidden_states)
        q = self.q_norm(q.reshape(B, L, self.num_heads, self.head_dim))
        q = q.transpose(1, 2)

        if self.is_cross_attention and encoder_hidden_states is not None:
            if cache is not None and cache.is_updated(self.layer_idx):
                k, v = cache.get(self.layer_idx)
            else:
                enc_L = encoder_hidden_states.shape[1]
                k = self.k_proj(encoder_hidden_states)
                k = self.k_norm(k.reshape(B, enc_L, self.num_kv_heads, self.head_dim))
                k = k.transpose(1, 2)
                v = self.v_proj(encoder_hidden_states).reshape(
                    B, enc_L, self.num_kv_heads, self.head_dim,
                ).transpose(1, 2)
                if cache is not None and use_cache:
                    cache.update(k, v, self.layer_idx)
        else:
            k = self.k_proj(hidden_states)
            k = self.k_norm(k.reshape(B, L, self.num_kv_heads, self.head_dim))
            k = k.transpose(1, 2)
            v = self.v_proj(hidden_states).reshape(
                B, L, self.num_kv_heads, self.head_dim,
            ).transpose(1, 2)

            if position_cos_sin is not None:
                cos, sin = position_cos_sin
                q, k = _apply_rotary_pos_emb(q, k, cos, sin)

        k = repeat_kv_heads_torch(k, self.n_rep)
        v = repeat_kv_heads_torch(v, self.n_rep)

        attn_out = scaled_dot_product_attention_bhsd_torch(
            q, k, v, mask=attention_mask, scale=self.scale
        )
        attn_out = attn_out.transpose(1, 2).reshape(B, L, -1)
        return self.o_proj(attn_out)


# ---------------------------------------------------------------------------
# DiT layer  (identical to MLX version, but with torch ops)
# ---------------------------------------------------------------------------

class _DiTLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        rms_norm_eps: float,
        attention_bias: bool,
        layer_idx: int,
        layer_type: str,
        sliding_window: Optional[int] = None,
    ):
        super().__init__()
        self.layer_type = layer_type
        sw = sliding_window if layer_type == "sliding_attention" else None

        self.self_attn_norm = _AceStepRMSNorm(hidden_size, eps=rms_norm_eps)
        self.self_attn = _Attention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            rms_norm_eps=rms_norm_eps,
            attention_bias=attention_bias,
            layer_idx=layer_idx,
            is_cross_attention=False,
            sliding_window=sw,
        )

        self.cross_attn_norm = _AceStepRMSNorm(hidden_size, eps=rms_norm_eps)
        self.cross_attn = _Attention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            rms_norm_eps=rms_norm_eps,
            attention_bias=attention_bias,
            layer_idx=layer_idx,
            is_cross_attention=True,
        )

        self.mlp_norm = _AceStepRMSNorm(hidden_size, eps=rms_norm_eps)
        self.mlp = _SwiGLUMLP(hidden_size, intermediate_size)

        self.scale_shift_table = nn.Parameter(torch.zeros(1, 6, hidden_size))

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_cos_sin: Tuple[torch.Tensor, torch.Tensor],
        temb: torch.Tensor,
        self_attn_mask: Optional[torch.Tensor],
        encoder_hidden_states: Optional[torch.Tensor],
        encoder_attention_mask: Optional[torch.Tensor],
        cache: Optional[_CrossAttentionCache] = None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        modulation = self.scale_shift_table + temb
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = unpack_modulation_6table(
            modulation
        )

        # 1) Self-attention
        normed = self.self_attn_norm(hidden_states)
        normed = apply_scale_shift(normed, scale_msa, shift_msa, add_one=True)
        attn_out = self.self_attn(
            normed,
            position_cos_sin=position_cos_sin,
            attention_mask=self_attn_mask,
        )
        hidden_states = hidden_states + attn_out * gate_msa

        # 2) Cross-attention
        normed = self.cross_attn_norm(hidden_states)
        cross_out = self.cross_attn(
            normed,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=encoder_attention_mask,
            cache=cache,
            use_cache=use_cache,
        )
        hidden_states = hidden_states + cross_out

        # 3) MLP
        normed = self.mlp_norm(hidden_states)
        normed = apply_scale_shift(normed, c_scale_msa, c_shift_msa, add_one=True)
        ff_out = self.mlp(normed)
        hidden_states = hidden_states + ff_out * c_gate_msa

        return hidden_states


# ---------------------------------------------------------------------------
# Timestep embedding
# ---------------------------------------------------------------------------

class _TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding → MLP → (temb, 6-way projection).

    Mirrors ``transformer_mlx._TimestepEmbedding``; weights: ``linear_1``,
    ``linear_2``, ``time_proj`` (no separate ``act1`` — SiLU is functional).
    """

    def __init__(self, in_channels: int = 256, time_embed_dim: int = 2048, scale: float = 1000.0):
        super().__init__()
        self.in_channels = in_channels
        self.scale = scale
        self.linear_1 = nn.Linear(in_channels, time_embed_dim, bias=True)
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim, bias=True)
        self.act2 = nn.SiLU()
        self.time_proj = nn.Linear(time_embed_dim, time_embed_dim * 6, bias=True)

    def forward(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        t_freq = sinusoidal_timestep_proj(
            _CUDA_CTX, t, self.in_channels, sin_first=False, scale=self.scale
        ).to(t.dtype)
        x = t_freq.to(t.dtype)
        temb = self.linear_2(F.silu(self.linear_1(x)))
        proj = self.time_proj(self.act2(temb))
        timestep_proj = proj.reshape(proj.shape[0], 6, -1)
        return temb, timestep_proj


# ---------------------------------------------------------------------------
# Full DiT decoder  (PyTorch mirror of AceStepDiTMLX)
# ---------------------------------------------------------------------------

class AceStepDiTCuda(nn.Module):
    """PyTorch DiT decoder for ACE-Step audio generation.

    Identical architecture to ``AceStepDiTMLX`` but using ``torch.nn``.
    Convolution ops internally permute NLC ↔ NCL for PyTorch compatibility.
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        intermediate_size: int = 6144,
        num_hidden_layers: int = 24,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 8,
        head_dim: int = 128,
        rms_norm_eps: float = 1e-6,
        attention_bias: bool = False,
        in_channels: int = 192,
        audio_acoustic_hidden_dim: int = 64,
        patch_size: int = 2,
        sliding_window: int = 128,
        layer_types: Optional[list] = None,
        rope_theta: float = 1_000_000.0,
        max_position_embeddings: int = 32768,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.patch_size = patch_size

        if layer_types is None:
            layer_types = [
                "sliding_attention" if bool((i + 1) % 2) else "full_attention"
                for i in range(num_hidden_layers)
            ]

        self.rotary_emb = _AceStepRotaryEmbedding(
            head_dim, max_len=max_position_embeddings, base=rope_theta,
        )

        # PyTorch Conv1d uses NCL format; we permute on the fly.
        self.proj_in = nn.Conv1d(
            in_channels=in_channels,
            out_channels=hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
            bias=True,
        )

        self.time_embed = _TimestepEmbedding(time_embed_dim=hidden_size)
        self.time_embed_r = _TimestepEmbedding(time_embed_dim=hidden_size)

        self.condition_embedder = nn.Linear(hidden_size, hidden_size, bias=True)

        self.layers = nn.ModuleList([
            _DiTLayer(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
                head_dim=head_dim,
                rms_norm_eps=rms_norm_eps,
                attention_bias=attention_bias,
                layer_idx=i,
                layer_type=layer_types[i],
                sliding_window=sliding_window,
            )
            for i in range(num_hidden_layers)
        ])

        self.norm_out = _AceStepRMSNorm(hidden_size, eps=rms_norm_eps)
        self.proj_out = nn.ConvTranspose1d(
            in_channels=hidden_size,
            out_channels=audio_acoustic_hidden_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
            bias=True,
        )

        self.scale_shift_table = nn.Parameter(torch.zeros(1, 2, hidden_size))
        self._sliding_window = sliding_window
        self._layer_types = layer_types
        self._sliding_masks: Dict[tuple[int, torch.dtype, str], torch.Tensor] = {}

    def _get_sliding_mask(self, seq_len: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        dev_key = f"{device.type}:{device.index}"
        key = (int(seq_len), dtype, dev_key)
        if key not in self._sliding_masks:
            self._sliding_masks[key] = build_window_with_padding_bias_torch(
                int(seq_len),
                dtype,
                device,
                attention_mask=None,
                sliding_window=self._sliding_window,
                neg_value=-1e9,
            )
        return self._sliding_masks[key]

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        timestep_r: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        context_latents: torch.Tensor,
        cache: Optional[_CrossAttentionCache] = None,
        use_cache: bool = True,
    ) -> Tuple[torch.Tensor, Optional[_CrossAttentionCache]]:
        temb_t, proj_t = self.time_embed(timestep)
        temb_r, proj_r = self.time_embed_r(timestep - timestep_r)
        temb = temb_t + temb_r
        timestep_proj = proj_t + proj_r

        hidden_states = torch.cat([context_latents, hidden_states], dim=-1)

        original_seq_len = hidden_states.shape[1]
        pad_length = 0
        if hidden_states.shape[1] % self.patch_size != 0:
            pad_length = self.patch_size - (hidden_states.shape[1] % self.patch_size)
            padding = torch.zeros(
                hidden_states.shape[0], pad_length, hidden_states.shape[2],
                dtype=hidden_states.dtype, device=hidden_states.device,
            )
            hidden_states = torch.cat([hidden_states, padding], dim=1)

        # Permute NLC → NCL for PyTorch Conv1d
        hidden_states = hidden_states.permute(0, 2, 1)
        hidden_states = self.proj_in(hidden_states)
        hidden_states = hidden_states.permute(0, 2, 1)  # back to NLC

        encoder_hidden_states = self.condition_embedder(encoder_hidden_states)

        seq_len = hidden_states.shape[1]
        dtype = hidden_states.dtype
        device = hidden_states.device

        cos, sin = self.rotary_emb(seq_len)
        cos = cos.to(device=device, dtype=dtype)
        sin = sin.to(device=device, dtype=dtype)

        has_sliding = any(lt == "sliding_attention" for lt in self._layer_types)
        sliding_mask = self._get_sliding_mask(seq_len, dtype, device) if has_sliding else None

        for layer in self.layers:
            self_attn_mask = sliding_mask if layer.layer_type == "sliding_attention" else None
            hidden_states = layer(
                hidden_states,
                position_cos_sin=(cos, sin),
                temb=timestep_proj,
                self_attn_mask=self_attn_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=None,
                cache=cache,
                use_cache=use_cache,
            )

        shift, scale = torch.chunk(
            self.scale_shift_table + temb.unsqueeze(1), 2, dim=1,
        )
        hidden_states = apply_scale_shift(self.norm_out(hidden_states), scale, shift, add_one=True)

        hidden_states = hidden_states.permute(0, 2, 1)  # NLC → NCL for ConvTranspose1d
        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states.permute(0, 2, 1)  # back to NLC

        hidden_states = hidden_states[:, :original_seq_len, :]

        return hidden_states, cache
