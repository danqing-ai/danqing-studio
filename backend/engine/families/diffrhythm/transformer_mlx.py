"""
DiffRhythm 2 CFM + DiT — MLX implementation (ASLP-lab/DiffRhythm2).

Ports ``diffrhythm2/backbones/dit.py``, ``diffrhythm2/cfm.py``, and
``diffrhythm2/cache_utils.py`` to pure MLX (no PyTorch).
"""
from __future__ import annotations

import logging
import math
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from backend.engine.common.attention import (
    rotate_half,
    scaled_dot_product_attention_bhsd_mx,
)
from backend.engine.runtime.mlx import MLXContext

logger = logging.getLogger(__name__)

_MLX_CTX = MLXContext()

# Default DiT hyper-parameters (DiffRhythm2 config.json)
_DEFAULT_DIM = 2048
_DEFAULT_DEPTH = 16
_DEFAULT_HEADS = 16
_DEFAULT_FF_MULT = 4
_DEFAULT_MEL_DIM = 64
_DEFAULT_TEXT_NUM_EMBEDS = 1000
_DEFAULT_BLOCK_SIZE = 10
_DEFAULT_COND_DIM = 512


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bool_mask_to_sdpa(mask: Optional[mx.array]) -> Optional[mx.array]:
    """Convert upstream bool mask (True = attend) to MLX SDPA additive bias."""
    if mask is None:
        return None
    if mask.dtype == mx.bool_:
        neg = mx.full(mask.shape, -1e9, dtype=mx.float32)
        return mx.where(mask, mx.zeros_like(neg), neg)
    return mask


def _pad_sequences_bhsd(sequences: List[mx.array]) -> mx.array:
    """Pad list of ``[H, L_b, D]`` tensors to ``[B, H, L_max, D]``."""
    if not sequences:
        raise RuntimeError("_pad_sequences_bhsd requires non-empty sequences")
    max_len = max(int(s.shape[1]) for s in sequences)
    batch = len(sequences)
    heads = int(sequences[0].shape[0])
    dim = int(sequences[0].shape[2])
    dtype = sequences[0].dtype
    out = mx.zeros((batch, heads, max_len, dim), dtype=dtype)
    for b, seq in enumerate(sequences):
        length = int(seq.shape[1])
        out[b, :, :length, :] = seq
    return out


def _sinusoidal_position_embedding(x: mx.array, dim: int, *, scale: float = 1000.0) -> mx.array:
    """Upstream ``SinusPositionEmbedding`` (scale * t → sin/cos concat)."""
    half_dim = dim // 2
    if half_dim < 1:
        raise RuntimeError(f"sinusoidal embedding dim must be >= 2, got {dim}")
    log_base = math.log(10000.0) / max(half_dim - 1, 1)
    freq = mx.exp(-log_base * mx.arange(half_dim, dtype=mx.float32))
    t = x.astype(mx.float32)
    if t.ndim == 1:
        t = t[:, None]
    if t.ndim == 2:
        t = t[:, :, None]
    emb = scale * t * freq[None, None, :]
    return mx.concatenate([mx.sin(emb), mx.cos(emb)], axis=-1)


# ---------------------------------------------------------------------------
# Block flow-matching KV cache
# ---------------------------------------------------------------------------


class BlockFlowMatchingCacheMLX:
    """MLX port of ``diffrhythm2/cache_utils.BlockFlowMatchingCache``."""

    def __init__(
        self,
        text_lengths: Optional[mx.array] = None,
        block_size: Optional[int] = None,
        num_history_block: Optional[int] = None,
    ) -> None:
        self._seen_tokens = 0
        self.text_key_cache: List[mx.array] = []
        self.text_value_cache: List[mx.array] = []
        self.key_cache: List[mx.array] = []
        self.value_cache: List[mx.array] = []
        self.text_lengths = text_lengths
        self.block_size = block_size
        self.num_history_block = num_history_block
        self.is_cache_text = False
        self.is_storage_cache = False
        if num_history_block is not None and block_size is None:
            raise RuntimeError("num_history_block requires block_size")
        if num_history_block is not None and num_history_block <= 0:
            self.num_history_block = None

    @contextmanager
    def cache_text(self):
        self.is_cache_text = True
        try:
            yield self
        finally:
            self.is_cache_text = False

    @contextmanager
    def cache_context(self):
        self.is_storage_cache = True
        try:
            yield self
        finally:
            self.is_storage_cache = False

    def update(
        self,
        key_states: mx.array,
        value_states: mx.array,
        layer_idx: int,
    ) -> Tuple[mx.array, mx.array]:
        if self.is_cache_text:
            if self.text_lengths is None:
                self.text_lengths = mx.array(
                    [int(key_states.shape[-2])] * int(key_states.shape[0]),
                    dtype=mx.int32,
                )
            self.text_key_cache.append(key_states)
            self.text_value_cache.append(value_states)
            return self.text_key_cache[layer_idx], self.text_value_cache[layer_idx]

        if layer_idx == 0:
            self._seen_tokens += int(key_states.shape[-2])

        if key_states is not None:
            while len(self.key_cache) <= layer_idx:
                self.key_cache.append(None)
                self.value_cache.append(None)

            cached_key = self.key_cache[layer_idx]
            cached_value = self.value_cache[layer_idx]
            if cached_key is not None and int(cached_key.shape[-2]) > 0:
                key_states = mx.concatenate([cached_key, key_states], axis=-2)
                value_states = mx.concatenate([cached_value, value_states], axis=-2)

            if self.num_history_block is not None and self.block_size is not None:
                history_length = self.block_size * (self.num_history_block + 1)
                key_states = key_states[:, :, -history_length:, :]
                value_states = value_states[:, :, -history_length:, :]

            if self.is_storage_cache:
                self.key_cache[layer_idx] = key_states
                self.value_cache[layer_idx] = value_states

        batch = int(key_states.shape[0])
        heads = int(key_states.shape[1])
        head_dim = int(key_states.shape[-1])
        dtype = key_states.dtype

        if self.text_lengths is None:
            text_lens = [0] * batch
        else:
            text_lens = [int(self.text_lengths[b]) for b in range(batch)]

        k_parts: List[mx.array] = []
        v_parts: List[mx.array] = []
        for b in range(batch):
            text_len = text_lens[b]
            if len(self.text_key_cache) > layer_idx:
                text_k = self.text_key_cache[layer_idx][b, :, :text_len, :]
                text_v = self.text_value_cache[layer_idx][b, :, :text_len, :]
            else:
                text_k = mx.zeros((heads, 0, head_dim), dtype=dtype)
                text_v = mx.zeros((heads, 0, head_dim), dtype=dtype)
            k_parts.append(mx.concatenate([text_k, key_states[b]], axis=-2))
            v_parts.append(mx.concatenate([text_v, value_states[b]], axis=-2))

        return _pad_sequences_bhsd(k_parts), _pad_sequences_bhsd(v_parts)


# ---------------------------------------------------------------------------
# DiT building blocks
# ---------------------------------------------------------------------------


class _SinusPositionEmbedding(nn.Module):
    """Stateless sinusoidal position embedding (no learnable params)."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def __call__(self, x: mx.array, scale: float = 1000.0) -> mx.array:
        return _sinusoidal_position_embedding(x, self.dim, scale=scale)


class _TimeMLP(nn.Module):
    """PyTorch ``time_mlp``: Linear(0) → SiLU → Linear(2)."""

    def __init__(self, freq_embed_dim: int, dim: int):
        super().__init__()
        setattr(self, "0", nn.Linear(freq_embed_dim, dim))
        setattr(self, "2", nn.Linear(dim, dim))

    def __call__(self, hidden: mx.array) -> mx.array:
        hidden = getattr(self, "0")(hidden)
        hidden = nn.silu(hidden)
        return getattr(self, "2")(hidden)


class _TimestepEmbedding(nn.Module):
    def __init__(self, dim: int, freq_embed_dim: int = 256):
        super().__init__()
        self.time_embed = _SinusPositionEmbedding(freq_embed_dim)
        self.time_mlp = _TimeMLP(freq_embed_dim, dim)

    def __call__(self, timestep: mx.array) -> mx.array:
        hidden = self.time_embed(timestep)
        return self.time_mlp(hidden)


class _TextEmbedding(nn.Module):
    def __init__(self, text_num_embeds: int, text_dim: int):
        super().__init__()
        self.text_embed = nn.Embedding(text_num_embeds + 1, text_dim)

    def __call__(self, text: mx.array) -> mx.array:
        return self.text_embed(text)


class _InputEmbedding(nn.Module):
    def __init__(self, cond_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(cond_dim, cond_dim)
        self.proj_2 = nn.Linear(cond_dim, out_dim)

    def __call__(self, x: mx.array, style_emb: mx.array, time_emb: mx.array) -> mx.array:
        style = mx.expand_dims(style_emb, axis=1)
        style = mx.broadcast_to(style, (int(x.shape[0]), int(x.shape[1]), int(style_emb.shape[-1])))
        x_orig = x
        x = x + style + time_emb
        x = self.proj(x) + x_orig
        return self.proj_2(x)


class _LatentEmbed(nn.Module):
    def __init__(self, mel_dim: int, cond_dim: int):
        super().__init__()
        setattr(self, "0", nn.Linear(mel_dim, cond_dim))
        setattr(self, "1", nn.Linear(cond_dim, cond_dim))

    def __call__(self, x: mx.array) -> mx.array:
        x = getattr(self, "0")(x)
        return getattr(self, "1")(x)


class _AdaLayerNormZeroFinal(nn.Module):
    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.linear = nn.Linear(cond_dim, dim * 2)
        self.norm = nn.LayerNorm(dim, affine=False, eps=1e-6)

    def __call__(self, x: mx.array, emb: mx.array) -> mx.array:
        emb = self.linear(nn.silu(emb))
        scale, shift = mx.split(emb, 2, axis=-1)
        return self.norm(x) * (1.0 + scale) + shift


class _DiffRhythmRotaryEmbedding(nn.Module):
    """RoPE cos/sin from explicit ``position_ids`` (Llama-style)."""

    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()
        self.head_dim = head_dim
        self.inv_freq = 1.0 / (base ** (mx.arange(0, head_dim, 2, dtype=mx.float32) / head_dim))

    def __call__(self, x: mx.array, position_ids: mx.array) -> Tuple[mx.array, mx.array]:
        pos = position_ids.astype(mx.float32)
        freqs = pos[..., None] * self.inv_freq[None, None, :]
        emb = mx.concatenate([freqs, freqs], axis=-1)
        cos = mx.cos(emb).astype(x.dtype)
        sin = mx.sin(emb).astype(x.dtype)
        return cos, sin


class _LlamaSwiGLUMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class _LlamaNARSelfAttention(nn.Module):
    """Llama self-attention with Q/K RMSNorm, RoPE, and block-FM cache."""

    def __init__(self, hidden_size: int, num_heads: int, head_dim: int, layer_idx: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.layer_idx = layer_idx
        self.scale = head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        self.q_norm = nn.RMSNorm(head_dim, eps=1e-6)
        self.k_norm = nn.RMSNorm(head_dim, eps=1e-6)

    def __call__(
        self,
        hidden_states: mx.array,
        *,
        attention_mask: Optional[mx.array] = None,
        position_embeddings: Optional[Tuple[mx.array, mx.array]] = None,
        past_key_value: Optional[BlockFlowMatchingCacheMLX] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ) -> Tuple[mx.array, Optional[mx.array], Optional[BlockFlowMatchingCacheMLX]]:
        del output_attentions  # upstream optional; not needed for CFM sampling

        batch, seq_len, _ = hidden_states.shape
        q = self.q_proj(hidden_states).reshape(batch, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).reshape(batch, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(hidden_states).reshape(batch, seq_len, self.num_heads, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)

        if position_embeddings is not None:
            cos, sin = position_embeddings
            cos = mx.expand_dims(cos, axis=2)
            sin = mx.expand_dims(sin, axis=2)
            q = (q * cos) + (rotate_half(_MLX_CTX, q) * sin)
            k = (k * cos) + (rotate_half(_MLX_CTX, k) * sin)

        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        if use_cache and past_key_value is not None:
            k, v = past_key_value.update(k, v, self.layer_idx)

        attn_out = scaled_dot_product_attention_bhsd_mx(
            mx,
            q,
            k,
            v,
            scale=self.scale,
            mask=_bool_mask_to_sdpa(attention_mask),
            compute_dtype=mx.float32,
            out_dtype=hidden_states.dtype,
        )
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(batch, seq_len, -1)
        return self.o_proj(attn_out), None, past_key_value


class _LlamaNARDecoderLayer(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, head_dim: int, intermediate_size: int, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.self_attn = _LlamaNARSelfAttention(hidden_size, num_heads, head_dim, layer_idx)
        self.mlp = _LlamaSwiGLUMLP(hidden_size, intermediate_size)
        self.input_layernorm = nn.RMSNorm(hidden_size, eps=1e-6)
        self.post_attention_layernorm = nn.RMSNorm(hidden_size, eps=1e-6)

    def __call__(
        self,
        hidden_states: mx.array,
        *,
        attention_mask: Optional[mx.array] = None,
        position_embeddings: Optional[Tuple[mx.array, mx.array]] = None,
        past_key_value: Optional[BlockFlowMatchingCacheMLX] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> List[Any]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, attn_weights, present_kv = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs: List[Any] = [hidden_states]
        if output_attentions:
            outputs.append(attn_weights)
        if use_cache:
            outputs.append(present_kv)
        return outputs


class _RepaProjector(nn.Module):
    """Optional REPA projector MLP (layers 0/2/4 in upstream Sequential)."""

    def __init__(self, dim: int, repa_dim: int):
        super().__init__()
        setattr(self, "0", nn.Linear(dim, dim * 2))
        setattr(self, "2", nn.Linear(dim * 2, dim * 2))
        setattr(self, "4", nn.Linear(dim * 2, repa_dim))

    def __call__(self, x: mx.array) -> mx.array:
        x = nn.silu(getattr(self, "0")(x))
        x = nn.silu(getattr(self, "2")(x))
        return getattr(self, "4")(x)


# ---------------------------------------------------------------------------
# DiT
# ---------------------------------------------------------------------------


class DiffRhythm2DiTMLX(nn.Module):
    """DiffRhythm 2 DiT — Llama-NAR decoder with text/style conditioning."""

    def __init__(
        self,
        *,
        dim: int = _DEFAULT_DIM,
        depth: int = _DEFAULT_DEPTH,
        heads: int = _DEFAULT_HEADS,
        ff_mult: int = _DEFAULT_FF_MULT,
        mel_dim: int = _DEFAULT_MEL_DIM,
        text_num_embeds: int = _DEFAULT_TEXT_NUM_EMBEDS,
        cond_dim: int = _DEFAULT_COND_DIM,
        repa_depth: int = -1,
        repa_dims: Optional[List[int]] = None,
        block_size: Optional[int] = None,
        num_history_block: Optional[int] = None,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.mel_dim = mel_dim
        self.cond_dim = cond_dim
        self.repa_depth = repa_depth
        self.repa_dims = list(repa_dims or [])
        self.block_size = block_size
        self.num_history_block = num_history_block

        head_dim = dim // heads
        intermediate_size = dim * ff_mult

        self.time_embed = _TimestepEmbedding(cond_dim)
        self.text_embed = _TextEmbedding(text_num_embeds, cond_dim)
        self.input_embed = _InputEmbedding(cond_dim, dim)
        self.latent_embed = _LatentEmbed(mel_dim, cond_dim)
        self.rotary_embed = _DiffRhythmRotaryEmbedding(head_dim)

        self.transformer_blocks = [
            _LlamaNARDecoderLayer(dim, heads, head_dim, intermediate_size, layer_idx=i)
            for i in range(depth)
        ]

        self.norm_out = _AdaLayerNormZeroFinal(dim, cond_dim)
        self.proj_out = nn.Linear(dim, mel_dim, bias=False)

        self.projectors: Optional[List[_RepaProjector]] = None
        if self.repa_depth > 0 and self.repa_dims:
            self.projectors = [_RepaProjector(dim, repa_dim) for repa_dim in self.repa_dims]

    def __call__(
        self,
        x: mx.array,
        time: mx.array,
        position_ids: mx.array,
        style_prompt: mx.array,
        attn_mask: mx.array,
        output_attentions: bool = False,
        use_cache: bool = False,
        past_key_value: Optional[BlockFlowMatchingCacheMLX] = None,
    ) -> Tuple[mx.array, List[Any], Optional[BlockFlowMatchingCacheMLX]]:
        t = self.time_embed(time)
        c = t

        x = self.input_embed(x, style_prompt, c)
        position_embeddings = self.rotary_embed(x, position_ids)

        attn_weights: List[Any] = []
        if not use_cache:
            past_key_value = None

        for i, block in enumerate(self.transformer_blocks):
            block_out = block(
                x,
                attention_mask=attn_mask,
                position_embeddings=position_embeddings,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
            x = block_out[0]
            offset = 1
            if output_attentions:
                attn_weights.append(block_out[offset])
                offset += 1
            if use_cache:
                past_key_value = block_out[offset]

        x = self.norm_out(x, c)
        output = self.proj_out(x)
        return output, attn_weights, past_key_value


# ---------------------------------------------------------------------------
# CFM
# ---------------------------------------------------------------------------


class DiffRhythm2CFMMLX(nn.Module):
    """Conditional flow matching wrapper around :class:`DiffRhythm2DiTMLX`."""

    def __init__(
        self,
        transformer: DiffRhythm2DiTMLX,
        *,
        num_channels: Optional[int] = None,
        block_size: int = _DEFAULT_BLOCK_SIZE,
        num_history_block: Optional[int] = None,
    ):
        super().__init__()
        self.transformer = transformer
        self.num_channels = num_channels if num_channels is not None else transformer.mel_dim
        self.block_size = block_size
        self.num_history_block = num_history_block
        self.transformer.block_size = block_size
        self.transformer.num_history_block = num_history_block

    def sample_block_cache(
        self,
        text: mx.array,
        duration: int,
        style_prompt: mx.array,
        steps: int = 32,
        cfg_strength: float = 1.0,
        seed: Optional[int] = None,
    ) -> mx.array:
        """Block-wise CFM sampling with KV cache (upstream ``sample_block_cache``)."""
        if seed is not None:
            mx.random.seed(int(seed))

        batch = int(text.shape[0])
        num_blocks = duration // self.block_size + (1 if duration % self.block_size > 0 else 0)

        text_emb = self.transformer.text_embed(text)
        cfg_text_emb = self.transformer.text_embed(mx.zeros_like(text))
        text_lens = mx.array([int(text_emb.shape[1])], dtype=mx.int32)

        clean_emb_stream = mx.zeros((batch, 0, self.num_channels), dtype=style_prompt.dtype)
        noisy_lens = mx.array([self.block_size], dtype=mx.int32)

        kv_cache = BlockFlowMatchingCacheMLX(
            text_lengths=text_lens,
            block_size=self.block_size,
            num_history_block=self.num_history_block,
        )
        cfg_kv_cache = BlockFlowMatchingCacheMLX(
            text_lengths=text_lens,
            block_size=self.block_size,
            num_history_block=self.num_history_block,
        )

        cache_time = mx.full((batch, self.block_size), 1.0, dtype=style_prompt.dtype)

        if int(text_emb.shape[1]) != 0:
            text_time = mx.full((batch, int(text_emb.shape[1])), -1.0, dtype=style_prompt.dtype)
            text_position_ids = mx.arange(int(text_emb.shape[1]), dtype=mx.int32)[None, :]
            text_position_ids = mx.broadcast_to(text_position_ids, (batch, int(text_emb.shape[1])))
            text_seq = int(text_emb.shape[1])
            text_attn_mask = mx.ones((batch, 1, text_seq, text_seq), dtype=mx.bool_)

            with kv_cache.cache_text():
                _, _, kv_cache = self.transformer(
                    x=text_emb,
                    time=text_time,
                    attn_mask=text_attn_mask,
                    position_ids=text_position_ids,
                    style_prompt=style_prompt,
                    use_cache=True,
                    past_key_value=kv_cache,
                )
            with cfg_kv_cache.cache_text():
                _, _, cfg_kv_cache = self.transformer(
                    x=cfg_text_emb,
                    time=text_time,
                    attn_mask=text_attn_mask,
                    position_ids=text_position_ids,
                    style_prompt=mx.zeros_like(style_prompt),
                    use_cache=True,
                    past_key_value=cfg_kv_cache,
                )

        end_pos = 0
        for _bid in range(num_blocks):
            clean_lens = mx.array([int(clean_emb_stream.shape[1])], dtype=mx.int32)
            kv_len = int(text_lens[0]) + int(clean_lens[0]) + int(noisy_lens[0])
            query_len = int(noisy_lens[0])
            attn_mask = mx.ones((batch, 1, query_len, kv_len), dtype=mx.bool_)

            total_pos = int(clean_lens[0]) + int(noisy_lens[0])
            position_ids = mx.arange(total_pos, dtype=mx.int32)[None, -query_len:]
            position_ids = mx.broadcast_to(position_ids, (batch, query_len))

            def velocity_fn(t_scalar: mx.array, x_state: mx.array) -> mx.array:
                noisy_embed = self.transformer.latent_embed(x_state)
                if int(t_scalar.ndim) == 0:
                    t_batch = mx.full((batch,), float(t_scalar), dtype=style_prompt.dtype)
                else:
                    t_batch = t_scalar
                time_in = mx.broadcast_to(
                    mx.expand_dims(t_batch, axis=1),
                    (batch, query_len),
                )

                pred, _, _ = self.transformer(
                    x=noisy_embed,
                    time=time_in,
                    attn_mask=attn_mask,
                    position_ids=position_ids,
                    style_prompt=style_prompt,
                    use_cache=True,
                    past_key_value=kv_cache,
                )
                if cfg_strength < 1e-5:
                    return pred

                null_pred, _, _ = self.transformer(
                    x=noisy_embed,
                    time=time_in,
                    attn_mask=attn_mask,
                    position_ids=position_ids,
                    style_prompt=mx.zeros_like(style_prompt),
                    use_cache=True,
                    past_key_value=cfg_kv_cache,
                )
                return pred + (pred - null_pred) * cfg_strength

            noisy_emb = mx.random.normal((batch, self.block_size, self.num_channels)).astype(style_prompt.dtype)
            t_set = mx.linspace(0.0, 1.0, int(steps), dtype=noisy_emb.dtype)
            x_cur = noisy_emb
            for step_idx in range(int(steps) - 1):
                t_i = t_set[step_idx]
                dt = t_set[step_idx + 1] - t_i
                x_cur = x_cur + dt * velocity_fn(t_i, x_cur)
            sampled = x_cur

            cache_embed = self.transformer.latent_embed(sampled)
            with kv_cache.cache_context():
                _, _, kv_cache = self.transformer(
                    x=cache_embed,
                    time=cache_time,
                    attn_mask=attn_mask,
                    position_ids=position_ids,
                    style_prompt=style_prompt,
                    use_cache=True,
                    past_key_value=kv_cache,
                )
            with cfg_kv_cache.cache_context():
                _, _, cfg_kv_cache = self.transformer(
                    x=cache_embed,
                    time=cache_time,
                    attn_mask=attn_mask,
                    position_ids=position_ids,
                    style_prompt=mx.zeros_like(style_prompt),
                    use_cache=True,
                    past_key_value=cfg_kv_cache,
                )

            clean_emb_stream = mx.concatenate([clean_emb_stream, sampled], axis=1)

            pos = -1
            curr_frame = clean_emb_stream[:, pos, :]
            eos = mx.ones_like(curr_frame)
            last_kl = mx.mean(mx.square(curr_frame - eos))
            if float(last_kl) <= 0.05:
                while float(last_kl) <= 0.05 and abs(pos) < int(clean_emb_stream.shape[1]):
                    pos -= 1
                    curr_frame = clean_emb_stream[:, pos, :]
                    last_kl = mx.mean(mx.square(curr_frame - eos))
                end_pos = int(clean_emb_stream.shape[1]) + pos
                break
            end_pos = int(clean_emb_stream.shape[1])

        return clean_emb_stream[:, :end_pos, :]


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------


def parameters_flat(module: nn.Module) -> Dict[str, Any]:
    """Flatten ``nn.Module.parameters()`` to dot-separated keys."""
    flat: Dict[str, Any] = {}
    _collect_flat_params(module, "", flat)
    return flat


def _collect_flat_params(obj: Any, prefix: str, result: Dict[str, Any]) -> None:
    if isinstance(obj, mx.array):
        if prefix:
            result[prefix] = obj
        return

    if isinstance(obj, nn.Module):
        params = obj.parameters()
        if isinstance(params, dict) and params:
            for name, value in params.items():
                key = f"{prefix}.{name}" if prefix else str(name)
                _collect_flat_params(value, key, result)
            return

    if isinstance(obj, dict):
        for name, value in obj.items():
            key = f"{prefix}.{name}" if prefix else str(name)
            _collect_flat_params(value, key, result)
        return

    if isinstance(obj, (list, tuple)):
        for idx, value in enumerate(obj):
            key = f"{prefix}.{idx}" if prefix else str(idx)
            _collect_flat_params(value, key, result)
        return


def _nest_flat_weights(flat: Dict[str, Any]) -> Dict[str, Any]:
    """Convert flat dot keys into nested dict/list structure for ``Module.update``."""
    root: Dict[str, Any] = {}

    def _insert(tree: Dict[str, Any], parts: List[str], value: Any) -> None:
        key = parts[0]
        if len(parts) == 1:
            tree[key] = value
            return
        nxt = parts[1]
        if nxt.isdigit():
            idx = int(nxt)
            lst = tree.setdefault(key, [])
            while len(lst) <= idx:
                lst.append({})
            if len(parts) == 2:
                lst[idx] = value
            else:
                if not isinstance(lst[idx], dict):
                    lst[idx] = {}
                _insert(lst[idx], parts[2:], value)
        else:
            child = tree.setdefault(key, {})
            if not isinstance(child, dict):
                raise RuntimeError(f"Cannot nest weight key under non-dict node: {key}")
            _insert(child, parts[1:], value)

    for key, value in flat.items():
        _insert(root, key.split("."), value)
    return root


def load_cfm_weights(model: DiffRhythm2CFMMLX, weights: List[Tuple[str, Any]]) -> None:
    """Load CFM checkpoint weights into ``model.transformer`` (strip ``transformer.`` prefix)."""
    param_map = parameters_flat(model.transformer)
    updates: Dict[str, Any] = {}
    unknown: List[str] = []

    for key, tensor in weights:
        target = key
        if target.startswith("transformer."):
            target = target[len("transformer.") :]
        if target.startswith("model."):
            target = target[len("model.") :]

        if target not in param_map:
            unknown.append(key)
            continue
        updates[target] = tensor

    if unknown:
        preview = ", ".join(unknown[:8])
        suffix = "…" if len(unknown) > 8 else ""
        logger.warning(
            "DiffRhythm 2 CFM: skipped %d checkpoint keys not in DiT param map (first: %s%s)",
            len(unknown),
            preview,
            suffix,
        )

    missing = [k for k in param_map if k not in updates]
    if missing:
        preview = ", ".join(missing[:8])
        suffix = "…" if len(missing) > 8 else ""
        raise RuntimeError(
            f"DiffRhythm 2 CFM weight load incomplete: {len(missing)} parameters missing "
            f"(first: {preview}{suffix})"
        )

    nested = _nest_flat_weights(updates)
    model.transformer.update(nested)
