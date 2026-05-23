"""Flux.1 CLIP-L — mflux ``CLIPEncoder`` 对齐（argmax pooled + ``Module.update`` 加载）。"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from backend.engine.common.attention import scaled_dot_product_attention_bhsd_mx

from backend.engine.families.flux1.weights import nest_flux1_clip_weights, remap_flux1_clip_weights
from backend.engine.runtime._base import RuntimeContext


def _to_bfloat16_weights(weights: dict[str, Any]) -> dict[str, Any]:
    return {
        k: (v if v.dtype == mx.bfloat16 else v.astype(mx.bfloat16))
        for k, v in weights.items()
    }


class _CLIPEmbeddings(nn.Module):
    def __init__(self, dims: int = 768):
        super().__init__()
        self.position_embedding = nn.Embedding(77, dims)
        self.token_embedding = nn.Embedding(49408, dims)

    def __call__(self, tokens: mx.array) -> mx.array:
        seq_length = tokens.shape[-1]
        position_ids = mx.arange(0, seq_length).reshape(1, seq_length)
        return self.token_embedding(tokens) + self.position_embedding(position_ids)


class _CLIPSdpaAttention(nn.Module):
    head_dimension = 64
    batch_size = 1
    num_heads = 12

    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(768, 768, bias=True)
        self.k_proj = nn.Linear(768, 768, bias=True)
        self.v_proj = nn.Linear(768, 768, bias=True)
        self.out_proj = nn.Linear(768, 768, bias=True)

    def __call__(self, hidden_states: mx.array, causal_attention_mask: mx.array) -> mx.array:
        mask = causal_attention_mask.astype(mx.bfloat16)
        query = self._reshape_and_transpose(self.q_proj(hidden_states))
        key = self._reshape_and_transpose(self.k_proj(hidden_states))
        value = self._reshape_and_transpose(self.v_proj(hidden_states))
        scale = float(query.shape[-1]) ** -0.5
        attn = scaled_dot_product_attention_bhsd_mx(mx, query, key, value, scale=scale, mask=mask)
        attn = mx.transpose(attn, (0, 2, 1, 3))
        attn = mx.reshape(attn, (self.batch_size, -1, self.num_heads * self.head_dimension))
        return self.out_proj(attn)

    def _reshape_and_transpose(self, x: mx.array) -> mx.array:
        return mx.transpose(
            mx.reshape(x, (self.batch_size, -1, self.num_heads, self.head_dimension)),
            (0, 2, 1, 3),
        )


class _CLIPMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(768, 3072, bias=True)
        self.fc2 = nn.Linear(3072, 768, bias=True)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        hidden_states = self.fc1(hidden_states)
        hidden_states = hidden_states * nn.sigmoid(1.702 * hidden_states)
        return self.fc2(hidden_states)


class _CLIPEncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _CLIPSdpaAttention()
        self.layer_norm1 = nn.LayerNorm(768)
        self.mlp = _CLIPMLP()
        self.layer_norm2 = nn.LayerNorm(768)

    def __call__(self, hidden_states: mx.array, causal_attention_mask: mx.array) -> mx.array:
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(hidden_states, causal_attention_mask)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class _EncoderCLIP(nn.Module):
    def __init__(self, num_encoder_layers: int = 12):
        super().__init__()
        self.layers = [_CLIPEncoderLayer() for _ in range(num_encoder_layers)]

    def __call__(self, tokens: mx.array, causal_attention_mask: mx.array) -> mx.array:
        hidden_states = tokens
        for layer in self.layers:
            hidden_states = layer(hidden_states, causal_attention_mask)
        return hidden_states


class _CLIPTextModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = _CLIPEmbeddings()
        self.encoder = _EncoderCLIP()
        self.final_layer_norm = nn.LayerNorm(768)

    @staticmethod
    def _create_causal_attention_mask(input_shape: tuple[int, ...]) -> mx.array:
        batch_size, query_length, _ = input_shape
        mask = mx.tril(mx.ones((query_length, query_length)), k=0)
        mask = 1 - mask
        mask = mask * -3.4e38
        mask = mask.reshape((1, 1, query_length, query_length))
        return mx.broadcast_to(mask, (batch_size, 1, query_length, query_length))

    def __call__(self, tokens: mx.array) -> mx.array:
        hidden_states = self.embeddings(tokens)
        causal_attention_mask = self._create_causal_attention_mask(hidden_states.shape)
        encoder_outputs = self.encoder(hidden_states, causal_attention_mask)
        last_hidden_state = self.final_layer_norm(encoder_outputs)
        return last_hidden_state[0, mx.argmax(tokens, axis=-1)]


class Flux1CLIPEncoder(nn.Module):
    """mflux-compatible CLIP text encoder for Flux.1 ``text_encoder`` bundle."""

    def __init__(
        self,
        ctx: RuntimeContext,
        model_path: str | Path,
        *,
        max_seq_len: int = 77,
        tokenizer_path: str | Path | None = None,
    ):
        super().__init__()
        self.ctx = ctx
        self.model_path = Path(model_path)
        self.max_seq_len = max_seq_len
        self._tokenizer_path = Path(tokenizer_path) if tokenizer_path else self.model_path
        self._tokenizer = None
        self.text_model = _CLIPTextModel()
        self._load_bundle_weights()

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import CLIPTokenizer

            self._tokenizer = CLIPTokenizer.from_pretrained(str(self._tokenizer_path))
        return self._tokenizer

    def _load_bundle_weights(self) -> None:
        weights: dict[str, Any] = {}
        for sf in sorted(self.model_path.glob("*.safetensors")):
            weights.update(self.ctx.load_weights(str(sf)))
        flat = _to_bfloat16_weights(remap_flux1_clip_weights(weights))
        nested = nest_flux1_clip_weights(flat)
        self.update(nested, strict=False)
        if getattr(self.ctx, "backend", None) == "mlx":
            self.ctx.eval(self.parameters())

    def encode(self, texts: list[str]) -> tuple[Any, Any]:
        tokenizer = self.tokenizer
        tokens = tokenizer(
            texts,
            padding="max_length",
            max_length=self.max_seq_len,
            truncation=True,
            return_tensors="np",
        )
        input_ids = self.ctx.array(tokens["input_ids"], dtype=mx.int32)
        pooled = self(input_ids)
        return pooled, None

    def __call__(self, input_ids: mx.array) -> mx.array:
        return self.text_model(input_ids)
