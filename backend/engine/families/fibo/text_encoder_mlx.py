"""FIBO text encoder — MLX SmolLM3 (mflux-compatible) + PromptEncoder CFG batching."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from backend.engine.common.attention import scaled_dot_product_attention_bhsd_mx
from backend.engine.common.mlx_runtime_fallback import load_weights_dict
from backend.engine.common.text_encoders.qwen3_mlx import MlxRMSNorm

_FIBO_TOTAL_DIT_LAYERS = 46  # 8 joint + 38 single


def _remap_smollm3_weights(raw: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, tensor in raw.items():
        out[key[6:] if key.startswith("model.") else key] = tensor
    return out


class _SmolLM3RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int = 65_536, base: float = 5_000_000.0):
        super().__init__()
        self.inv_freq = 1.0 / (base ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))

    def __call__(self, seq_len: int) -> tuple[mx.array, mx.array]:
        positions = mx.arange(seq_len, dtype=mx.float32)
        freqs = mx.outer(positions, self.inv_freq)
        emb = mx.concatenate([freqs, freqs], axis=-1)
        cos = mx.expand_dims(mx.expand_dims(mx.cos(emb), axis=0), axis=0)
        sin = mx.expand_dims(mx.expand_dims(mx.sin(emb), axis=0), axis=0)
        return cos, sin


def _rotate_half(x: mx.array) -> mx.array:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return mx.concatenate([-x2, x1], axis=-1)


class _SmolLM3Attention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        rope_theta: float,
        max_position_embeddings: int,
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = hidden_size // num_attention_heads
        self.num_key_value_groups = num_attention_heads // num_key_value_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.q_proj = nn.Linear(hidden_size, num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_attention_heads * self.head_dim, hidden_size, bias=False)
        self.rotary_emb = _SmolLM3RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=max_position_embeddings,
            base=rope_theta,
        )

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: mx.array | None,
        cos_sin: tuple[mx.array, mx.array] | None,
    ) -> mx.array:
        batch_size, seq_len, _ = hidden_states.shape
        q = self.q_proj(hidden_states).astype(hidden_states.dtype)
        k = self.k_proj(hidden_states).astype(hidden_states.dtype)
        v = self.v_proj(hidden_states).astype(hidden_states.dtype)
        q = q.reshape(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(0, 2, 1, 3)
        if cos_sin is None:
            cos_sin = self.rotary_emb(seq_len)
        cos, sin = cos_sin
        q = (q.astype(mx.float32) * cos) + (_rotate_half(q.astype(mx.float32)) * sin)
        k = (k.astype(mx.float32) * cos) + (_rotate_half(k.astype(mx.float32)) * sin)
        q, k = q.astype(hidden_states.dtype), k.astype(hidden_states.dtype)
        if self.num_key_value_heads != self.num_attention_heads:
            k = self._repeat_kv(k)
            v = self._repeat_kv(v)
        attn_mask = None
        if attention_mask is not None:
            seq_len_k = k.shape[2]
            causal_mask = attention_mask[:, :, :, :seq_len_k]
            if causal_mask.shape[1] == 1:
                causal_mask = mx.broadcast_to(
                    causal_mask, (batch_size, self.num_attention_heads, seq_len, seq_len_k)
                )
            attn_mask = causal_mask.astype(q.dtype)
        attn_output = scaled_dot_product_attention_bhsd_mx(
            mx, q, k, v, scale=self.scale, mask=attn_mask,
        )
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        return self.o_proj(attn_output)

    def _repeat_kv(self, hidden_states: mx.array) -> mx.array:
        batch, num_kv_heads, seq_len, head_dim = hidden_states.shape
        hidden_states = mx.expand_dims(hidden_states, axis=2)
        hidden_states = mx.broadcast_to(
            hidden_states, (batch, num_kv_heads, self.num_key_value_groups, seq_len, head_dim)
        )
        return hidden_states.reshape(batch, num_kv_heads * self.num_key_value_groups, seq_len, head_dim)


class _SmolLM3MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        gate = self.gate_proj(hidden_states)
        gate = gate * mx.sigmoid(gate)
        return self.down_proj(gate * self.up_proj(hidden_states))


class _SmolLM3EncoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        intermediate_size: int,
        rms_norm_eps: float,
        rope_theta: float,
        max_position_embeddings: int,
    ):
        super().__init__()
        self.input_layernorm = MlxRMSNorm(hidden_size, eps=rms_norm_eps)
        self.self_attn = _SmolLM3Attention(
            hidden_size,
            num_attention_heads,
            num_key_value_heads,
            rope_theta,
            max_position_embeddings,
        )
        self.post_attention_layernorm = MlxRMSNorm(hidden_size, eps=rms_norm_eps)
        self.mlp = _SmolLM3MLP(hidden_size, intermediate_size)

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: mx.array | None,
        cos_sin: tuple[mx.array, mx.array],
    ) -> mx.array:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask, cos_sin)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class _SmolLM3TextEncoderModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        intermediate_size: int,
        num_hidden_layers: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        max_position_embeddings: int,
        rope_theta: float,
        rms_norm_eps: float,
    ):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = [
            _SmolLM3EncoderLayer(
                hidden_size,
                num_attention_heads,
                num_key_value_heads,
                intermediate_size,
                rms_norm_eps,
                rope_theta,
                max_position_embeddings,
            )
            for _ in range(num_hidden_layers)
        ]
        self.norm = MlxRMSNorm(hidden_size, eps=rms_norm_eps)
        self.rotary_emb = _SmolLM3RotaryEmbedding(
            hidden_size // num_attention_heads,
            max_position_embeddings=max_position_embeddings,
            base=rope_theta,
        )

    def forward_hidden_states(
        self,
        input_ids: mx.array,
        attention_mask: mx.array,
    ) -> list[mx.array]:
        batch_size, seq_len = input_ids.shape
        hidden_states = self.embed_tokens(input_ids)
        min_dtype = mx.finfo(mx.float32).min
        padding_mask = mx.where(
            attention_mask == 1,
            mx.zeros_like(attention_mask).astype(mx.float32),
            mx.ones_like(attention_mask).astype(mx.float32) * min_dtype,
        )
        padding_mask = mx.expand_dims(mx.expand_dims(padding_mask, axis=1), axis=1)
        idx = mx.arange(seq_len, dtype=mx.int32)
        tri = mx.expand_dims(idx, axis=0) > mx.expand_dims(idx, axis=1)
        causal = mx.where(
            tri,
            mx.ones((seq_len, seq_len), dtype=mx.float32) * min_dtype,
            mx.zeros((seq_len, seq_len), dtype=mx.float32),
        )
        causal = mx.expand_dims(mx.expand_dims(causal, axis=0), axis=0)
        causal = mx.broadcast_to(causal, (batch_size, 1, seq_len, seq_len))
        attention_mask_4d = (causal + padding_mask).astype(hidden_states.dtype)
        cos_sin = self.rotary_emb(seq_len)
        hidden_states_list: list[mx.array] = [hidden_states]
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask_4d, cos_sin)
            hidden_states_list.append(hidden_states)
        hidden_states_list[-1] = self.norm(hidden_states)
        return hidden_states_list


def _build_smollm3(model_path: str, ctx: Any) -> _SmolLM3TextEncoderModel:
    model_dir = Path(model_path)
    weights: dict[str, Any] = {}
    load_fn = getattr(ctx, "load_weights", None)
    for sf in sorted(model_dir.glob("*.safetensors")):
        weights.update(load_weights_dict(load_fn, str(sf)))
    config: dict[str, Any] = {}
    config_path = model_dir / "config.json"
    if config_path.is_file():
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)
    model = _SmolLM3TextEncoderModel(
        vocab_size=int(config.get("vocab_size", 128_256)),
        hidden_size=int(config.get("hidden_size", 2048)),
        intermediate_size=int(config.get("intermediate_size", 11_008)),
        num_hidden_layers=int(config.get("num_hidden_layers", 36)),
        num_attention_heads=int(config.get("num_attention_heads", 16)),
        num_key_value_heads=int(config.get("num_key_value_heads", 4)),
        max_position_embeddings=int(config.get("max_position_embeddings", 65_536)),
        rope_theta=float(config.get("rope_theta", 5_000_000.0)),
        rms_norm_eps=float(config.get("rms_norm_eps", 1e-6)),
    )
    model.load_weights(list(_remap_smollm3_weights(weights).items()), strict=False)
    ctx.eval(model.parameters())
    return model


def _pad_embedding(
    prompt_embeds: mx.array,
    max_tokens: int,
    attention_mask: mx.array | None = None,
) -> tuple[mx.array, mx.array]:
    batch_size, seq_len, dim = prompt_embeds.shape
    if attention_mask is None:
        attention_mask = mx.ones((batch_size, seq_len), dtype=prompt_embeds.dtype)
    else:
        attention_mask = attention_mask.astype(prompt_embeds.dtype)
    if max_tokens < seq_len:
        raise ValueError("`max_tokens` must be >= current sequence length.")
    if max_tokens > seq_len:
        pad_length = max_tokens - seq_len
        padding = mx.zeros((batch_size, pad_length, dim), dtype=prompt_embeds.dtype)
        prompt_embeds = mx.concatenate([prompt_embeds, padding], axis=1)
        mask_padding = mx.zeros((batch_size, pad_length), dtype=attention_mask.dtype)
        attention_mask = mx.concatenate([attention_mask, mask_padding], axis=1)
    return prompt_embeds, attention_mask


def _select_dit_layers(layers: list[mx.array]) -> list[mx.array]:
    if len(layers) >= _FIBO_TOTAL_DIT_LAYERS:
        return layers[len(layers) - _FIBO_TOTAL_DIT_LAYERS :]
    return layers + [layers[-1]] * (_FIBO_TOTAL_DIT_LAYERS - len(layers))


class FiboTextEncoder:
    """Encode JSON prompts with bundled SmolLM3 (MLX, mflux PromptEncoder parity)."""

    def __init__(
        self,
        ctx: Any,
        model_path: str,
        *,
        tokenizer_path: str | None = None,
        max_seq_len: int = 2048,
        **kwargs: Any,
    ):
        self.ctx = ctx
        self.model_path = model_path
        self._tokenizer_path = tokenizer_path or model_path
        self.max_seq_len = int(max_seq_len)
        self._tokenizer = None
        self._model: _SmolLM3TextEncoderModel | None = None

    def _tokenizer_lazy(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self._tokenizer_path)
        return self._tokenizer

    def _model_lazy(self) -> _SmolLM3TextEncoderModel:
        if self._model is None:
            self._model = _build_smollm3(self.model_path, self.ctx)
        return self._model

    def _tokenize(self, text: str) -> tuple[mx.array, mx.array]:
        import numpy as np

        tokenizer = self._tokenizer_lazy()
        tokens = tokenizer(
            [text],
            padding="longest",
            max_length=self.max_seq_len,
            truncation=True,
            return_tensors="np",
        )
        input_ids = self.ctx.array(tokens["input_ids"], dtype=mx.int32)
        attention_mask = self.ctx.array(tokens["attention_mask"], dtype=mx.int32)
        return input_ids, attention_mask

    def _encode_one(self, text: str) -> tuple[mx.array, list[mx.array], mx.array]:
        input_ids, attention_mask = self._tokenize(text)
        hidden_states_list = self._model_lazy().forward_hidden_states(input_ids, attention_mask)
        last = hidden_states_list[-1]
        prev = hidden_states_list[-2]
        prompt_embeds = mx.concatenate([last, prev], axis=-1)
        return prompt_embeds, hidden_states_list, attention_mask

    def encode(self, texts: list[str]) -> Any:
        if not texts:
            raise ValueError("FiboTextEncoder.encode requires non-empty texts")
        if len(texts) != 1:
            raise RuntimeError("FiboTextEncoder.encode supports one prompt per call")
        prompt_embeds, layers, _ = self._encode_one(texts[0])
        return prompt_embeds.astype(mx.bfloat16), layers

    def encode_prompt_cfg(
        self,
        prompt: str,
        negative_prompt: str | None,
        *,
        guidance: float,
    ) -> tuple[Any, list[Any]]:
        """mflux ``PromptEncoder.encode_prompt`` — batched [uncond, cond] on axis 0."""
        json.loads(prompt)
        pos_embeds, pos_layers, pos_mask = self._encode_one(prompt)
        if float(guidance) == 1.0:
            max_tokens = int(pos_embeds.shape[1])
            encoder_hidden_states, _ = _pad_embedding(pos_embeds, max_tokens, pos_mask)
            prompt_layers = [_pad_embedding(layer, max_tokens)[0] for layer in pos_layers]
            return encoder_hidden_states.astype(mx.bfloat16), _select_dit_layers(prompt_layers)

        neg_text = (negative_prompt or "").strip() or "ugly, blurry, low quality"
        neg_embeds, neg_layers, neg_mask = self._encode_one(neg_text)
        max_tokens = max(int(neg_embeds.shape[1]), int(pos_embeds.shape[1]))
        pos_embeds, pos_mask = _pad_embedding(pos_embeds, max_tokens, pos_mask)
        neg_embeds, neg_mask = _pad_embedding(neg_embeds, max_tokens, neg_mask)
        encoder_hidden_states = mx.concatenate([neg_embeds, pos_embeds], axis=0)
        pos_layers = [_pad_embedding(layer, max_tokens)[0] for layer in pos_layers]
        neg_layers = [_pad_embedding(layer, max_tokens)[0] for layer in neg_layers]
        prompt_layers = [
            mx.concatenate([neg_layers[i], pos_layers[i]], axis=0) for i in range(len(pos_layers))
        ]
        return encoder_hidden_states.astype(mx.bfloat16), _select_dit_layers(prompt_layers)
