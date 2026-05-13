"""Z-Image / Flux2 共享：Qwen3 MLX 栈 + ``ZImageTextEncoder``（MLX / CUDA 双路径）。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn


def build_zimage_mlx_encoder(model_path: str) -> "_ZImageEncoderModel":
    """Load Z-Image Qwen3 text encoder weights from ``model_path`` (safetensors + config.json)."""
    model_dir = Path(model_path)
    weights: dict = {}
    for sf in sorted(model_dir.glob("*.safetensors")):
        weights.update(dict(mx.load(str(sf))))

    config_path = model_dir / "config.json"
    config: dict = {}
    if config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)

    model = _ZImageEncoderModel(
        vocab_size=config.get("vocab_size", 151936),
        hidden_size=config.get("hidden_size", 2560),
        num_hidden_layers=config.get("num_hidden_layers", 36),
        num_attention_heads=config.get("num_attention_heads", 32),
        num_key_value_heads=config.get("num_key_value_heads", 8),
        intermediate_size=config.get("intermediate_size", 9728),
        head_dim=config.get("head_dim", 128),
        max_position_embeddings=config.get("max_position_embeddings", 40960),
        rope_theta=config.get("rope_theta", 1000000.0),
        rms_norm_eps=config.get("rms_norm_eps", 1e-6),
    )

    remapped: dict = {}
    for key, tensor in weights.items():
        new_key = key[6:] if key.startswith("model.") else key
        remapped[new_key] = tensor

    model.load_weights(list(remapped.items()), strict=False)
    mx.eval(model.parameters())
    return model


class _ZImageEncoderModel(nn.Module):
    """Z-Image Qwen3 text encoder MLX model."""

    def __init__(
        self,
        vocab_size=151936,
        hidden_size=2560,
        num_hidden_layers=36,
        num_attention_heads=32,
        num_key_value_heads=8,
        intermediate_size=9728,
        head_dim=128,
        max_position_embeddings=40960,
        rope_theta=1000000.0,
        rms_norm_eps=1e-6,
    ):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = [
            _ZImageEncoderLayer(
                hidden_size,
                num_attention_heads,
                num_key_value_heads,
                intermediate_size,
                head_dim,
                rms_norm_eps,
            )
            for _ in range(num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(hidden_size, eps=rms_norm_eps)
        self.rotary_emb = _ZImageEncoderRotaryEmbedding(dim=head_dim, base=rope_theta)

    def __call__(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.shape
        hidden_states = self.embed_tokens(input_ids).astype(mx.float32)
        position_ids = mx.broadcast_to(mx.arange(seq_len, dtype=mx.int32)[None, :], (batch_size, seq_len))
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        causal_mask = _ZImageEncoderModel._get_causal_mask(attention_mask, batch_size, hidden_states, seq_len)
        all_hidden_states = [hidden_states]
        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                attention_mask=causal_mask,
                position_embeddings=position_embeddings,
            )
            all_hidden_states.append(hidden_states)
        return all_hidden_states[-2].astype(input_ids.dtype)

    @staticmethod
    def _get_causal_mask(attention_mask, batch_size, hidden_states, seq_len):
        causal_mask = _ZImageEncoderModel._create_causal_mask(seq_len, hidden_states.dtype)
        if attention_mask is not None:
            padding_mask = mx.where(
                attention_mask[:, None, None, :] == 1,
                mx.zeros((batch_size, 1, 1, seq_len), dtype=hidden_states.dtype),
                mx.full((batch_size, 1, 1, seq_len), float("-inf"), dtype=hidden_states.dtype),
            )
            causal_mask = causal_mask + padding_mask
        return causal_mask

    @staticmethod
    def _create_causal_mask(seq_len, dtype):
        idx = mx.arange(seq_len, dtype=mx.int32)
        mask = idx[:, None] >= idx[None, :]
        causal_mask = mx.where(
            mask,
            mx.zeros((seq_len, seq_len), dtype=dtype),
            mx.full((seq_len, seq_len), float("-inf"), dtype=dtype),
        )
        return causal_mask[None, None, :, :]


class _ZImageEncoderRotaryEmbedding(nn.Module):
    def __init__(self, dim, base=1000000.0):
        super().__init__()
        self.inv_freq = 1.0 / (base ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))

    def __call__(self, x, position_ids):
        seq_len = position_ids.shape[-1]
        freqs = mx.outer(mx.arange(seq_len, dtype=mx.float32), self.inv_freq)
        emb = mx.concatenate([freqs, freqs], axis=-1)
        cos = mx.cos(emb)[None, :, :]
        sin = mx.sin(emb)[None, :, :]
        return cos.astype(x.dtype), sin.astype(x.dtype)


class _ZImageEncoderMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class _ZImageEncoderAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, num_kv_heads, head_dim):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_kv_groups = num_heads // num_kv_heads
        self.scale = head_dim**-0.5
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        self.q_norm = nn.RMSNorm(head_dim)
        self.k_norm = nn.RMSNorm(head_dim)

    def __call__(self, hidden_states, attention_mask=None, position_embeddings=None):
        batch_size, seq_len, _ = hidden_states.shape
        q = self.q_proj(hidden_states).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        q = self.q_norm(q)
        k = self.k_norm(k)
        if position_embeddings is not None:
            cos, sin = position_embeddings
            cos = mx.expand_dims(cos, axis=2)
            sin = mx.expand_dims(sin, axis=2)
            q_embed = (q * cos) + (_ZImageEncoderAttention._rotate_half(q) * sin)
            k_embed = (k * cos) + (_ZImageEncoderAttention._rotate_half(k) * sin)
            q, k = q_embed, k_embed
        if self.num_kv_groups > 1:
            k = mx.repeat(k, self.num_kv_groups, axis=2)
            v = mx.repeat(v, self.num_kv_groups, axis=2)
        q = mx.transpose(q, axes=(0, 2, 1, 3))
        k = mx.transpose(k, axes=(0, 2, 1, 3))
        v = mx.transpose(v, axes=(0, 2, 1, 3))
        from mlx.core.fast import scaled_dot_product_attention

        q_f32, k_f32, v_f32 = q.astype(mx.float32), k.astype(mx.float32), v.astype(mx.float32)
        attn_output = scaled_dot_product_attention(q_f32, k_f32, v_f32, scale=self.scale, mask=attention_mask)
        attn_output = attn_output.astype(q.dtype)
        attn_output = mx.transpose(attn_output, axes=(0, 2, 1, 3)).reshape(batch_size, seq_len, -1)
        return self.o_proj(attn_output)

    @staticmethod
    def _rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return mx.concatenate([-x2, x1], axis=-1)


class _ZImageEncoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        num_key_value_heads,
        intermediate_size,
        head_dim,
        rms_norm_eps=1e-6,
    ):
        super().__init__()
        self.input_layernorm = nn.RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(hidden_size, eps=rms_norm_eps)
        self.self_attn = _ZImageEncoderAttention(hidden_size, num_attention_heads, num_key_value_heads, head_dim)
        self.mlp = _ZImageEncoderMLP(hidden_size, intermediate_size)

    def __call__(self, hidden_states, attention_mask=None, position_embeddings=None):
        residual = hidden_states
        hidden_states = self.self_attn(self.input_layernorm(hidden_states), attention_mask, position_embeddings)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.mlp(self.post_attention_layernorm(hidden_states))
        return residual + hidden_states


class ZImageTextEncoder:
    """Z-Image text encoder — matches reference z_image TextEncoder.

    Two output modes (controlled by hidden_state_layers):
    - None (z_image): returns penultimate hidden state, shape [B, seq_len, hidden_size]
    - (9, 18, 27) (flux2 already split to Flux2TextEncoder)
    """

    def __init__(
        self,
        ctx: Any,
        model_path: str,
        max_seq_len: int = 512,
        tokenizer_path: str = "",
        hidden_state_layers: tuple[int, ...] | None = None,
        enable_thinking: bool = False,
        **_kw: Any,
    ):
        self.ctx = ctx
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path or model_path
        self.max_seq_len = max_seq_len
        self.hidden_state_layers = hidden_state_layers
        self.enable_thinking = enable_thinking
        self._tokenizer = None
        self._model = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_path,
                trust_remote_code=True,
            )
        return self._tokenizer

    def encode(self, texts: list[str]) -> Any:
        tokenizer = self.tokenizer
        ctx = self.ctx

        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            chat_texts = []
            for text in texts:
                chat = [{"role": "user", "content": text}]
                chat_text = tokenizer.apply_chat_template(
                    chat,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=self.enable_thinking,
                )
                chat_texts.append(chat_text)
            tokens = tokenizer(
                chat_texts,
                padding="max_length",
                max_length=self.max_seq_len,
                truncation=True,
                return_tensors="np",
            )
        else:
            tokens = tokenizer(
                texts,
                padding="max_length",
                max_length=self.max_seq_len,
                truncation=True,
                return_tensors="np",
            )
        input_ids = tokens["input_ids"]
        attention_mask = tokens["attention_mask"]

        if ctx.backend == "mlx":
            input_ids = mx.array(input_ids, dtype=mx.int32)
            attention_mask = mx.array(attention_mask, dtype=mx.float32)
            return self._forward_mlx(input_ids, attention_mask)
        from backend.engine.families.z_image.text_encoder_cuda import (
            zimage_prepare_torch_ids,
            zimage_text_encoder_forward_torch,
        )

        tid, tam = zimage_prepare_torch_ids(ctx, input_ids, attention_mask)
        return zimage_text_encoder_forward_torch(self, tid, tam)

    def _forward_mlx(self, input_ids, attention_mask):
        if self._model is None:
            self._model = build_zimage_mlx_encoder(self.model_path)

        batch_size, seq_len = input_ids.shape
        hidden_states = self._model.embed_tokens(input_ids).astype(mx.float32)
        causal_mask = self._model._get_causal_mask(attention_mask, batch_size, hidden_states, seq_len)
        position_ids = mx.broadcast_to(mx.arange(seq_len, dtype=mx.int32)[None, :], (batch_size, seq_len))
        position_embeddings = self._model.rotary_emb(hidden_states, position_ids)

        all_hidden_states = [hidden_states]
        for layer in self._model.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                attention_mask=causal_mask,
                position_embeddings=position_embeddings,
            )
            all_hidden_states.append(hidden_states)

        if self.hidden_state_layers is not None:
            layer_outputs = [all_hidden_states[i] for i in self.hidden_state_layers]
            stacked = mx.stack(layer_outputs, axis=1)
            B, L, S, D = stacked.shape
            result = mx.transpose(stacked, (0, 2, 1, 3)).reshape(B, S, L * D)
        else:
            result = all_hidden_states[-2]

        num_valid = int(mx.sum(attention_mask).item())
        result = result[:, :num_valid, :]
        return result.astype(mx.bfloat16)
