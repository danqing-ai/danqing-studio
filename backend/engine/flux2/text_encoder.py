"""Flux2 Klein 专属文本编码器 — Float32RMSNorm + float32 SDPA + 3层拼接。"""
from __future__ import annotations
from json import load as json_load
from pathlib import Path
from typing import Any
import mlx.core as mx
import mlx.nn as nn
from backend.engine.z_image.text_encoder import (
    _ZImageEncoderMLP, _ZImageEncoderRotaryEmbedding,
)
# ============================================================================
# Flux2TextEncoder — Flux2 Klein 专属文本编码器
# 参考 mflux flux2 Qwen3TextEncoder 实现：Float32RMSNorm + float32 SDPA + 3层拼接
# ============================================================================

class Float32RMSNorm(nn.Module):
    """与 mflux Qwen3VLRMSNorm 一致：RMS 计算前强制转 float32。"""
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((hidden_size,))
        self.eps = eps
    def __call__(self, hidden_states):
        import mlx.core as mx
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.astype(mx.float32)
        variance = mx.mean(mx.square(hidden_states), axis=-1, keepdims=True)
        hidden_states = hidden_states * mx.rsqrt(variance + self.eps)
        return (self.weight.astype(mx.float32) * hidden_states).astype(input_dtype)


class _Flux2Attention(nn.Module):
    """Flux2 专属 Attention — Float32RMSNorm QK norm + float32 SDPA。"""

    def __init__(self, hidden_size, num_heads, num_kv_heads, head_dim, rms_norm_eps=1e-6):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_kv_groups = num_heads // num_kv_heads
        self.scale = head_dim ** -0.5
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        self.q_norm = Float32RMSNorm(head_dim, eps=rms_norm_eps)
        self.k_norm = Float32RMSNorm(head_dim, eps=rms_norm_eps)

    def __call__(self, hidden_states, attention_mask, position_embeddings, past_key_value):
        import mlx.core as mx
        from mlx.core.fast import scaled_dot_product_attention
        B, S, _ = hidden_states.shape
        q = self.q_proj(hidden_states).reshape(B, S, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).reshape(B, S, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).reshape(B, S, self.num_kv_heads, self.head_dim)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q = mx.transpose(q, axes=(0, 2, 1, 3))
        k = mx.transpose(k, axes=(0, 2, 1, 3))
        v = mx.transpose(v, axes=(0, 2, 1, 3))
        if position_embeddings is not None:
            cos, sin = position_embeddings
            cos = mx.expand_dims(cos, axis=1)
            sin = mx.expand_dims(sin, axis=1)
            q = (q * cos) + (_rotate_half_f2(q) * sin)
            k = (k * cos) + (_rotate_half_f2(k) * sin)
        if self.num_kv_groups > 1:
            k = mx.repeat(k, self.num_kv_groups, axis=1)
            v = mx.repeat(v, self.num_kv_groups, axis=1)
        q_f32, k_f32, v_f32 = q.astype(mx.float32), k.astype(mx.float32), v.astype(mx.float32)
        out = scaled_dot_product_attention(q_f32, k_f32, v_f32, scale=self.scale, mask=attention_mask)
        out = out.astype(q.dtype)
        out = mx.transpose(out, axes=(0, 2, 1, 3)).reshape(B, S, self.num_heads * self.head_dim)
        return self.o_proj(out), None


def _rotate_half_f2(x):
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    import mlx.core as mx
    return mx.concatenate([-x2, x1], axis=-1)


class _Flux2DecoderLayer(nn.Module):
    """Flux2 专属 DecoderLayer — Float32RMSNorm + _Flux2Attention。"""

    def __init__(self, hidden_size, num_heads, num_kv_heads, intermediate_size, head_dim, rms_norm_eps=1e-6):
        super().__init__()
        self.input_layernorm = Float32RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = Float32RMSNorm(hidden_size, eps=rms_norm_eps)
        self.self_attn = _Flux2Attention(hidden_size, num_heads, num_kv_heads, head_dim, rms_norm_eps)
        self.mlp = _ZImageEncoderMLP(hidden_size, intermediate_size)

    def __call__(self, hidden_states, attention_mask, position_embeddings, past_key_value=None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_out, _ = self.self_attn(hidden_states, attention_mask, position_embeddings, past_key_value)
        hidden_states = residual + attn_out
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states, None


class _Flux2EncoderModel(nn.Module):
    """Flux2 Qwen3 编码器模型。"""

    def __init__(self, vocab_size=151936, hidden_size=4096, num_hidden_layers=36,
                 num_attention_heads=32, num_key_value_heads=8, intermediate_size=12288,
                 head_dim=128, rope_theta=1000000.0, rms_norm_eps=1e-6):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = [
            _Flux2DecoderLayer(hidden_size, num_attention_heads, num_key_value_heads,
                              intermediate_size, head_dim, rms_norm_eps)
            for _ in range(num_hidden_layers)
        ]
        self.rotary_emb = _ZImageEncoderRotaryEmbedding(dim=head_dim, base=rope_theta)


class Flux2TextEncoder:
    """Flux2 Klein 专属文本编码器 — 与 mflux qwen3_text_encoder 一致。

    - Float32RMSNorm 替代 nn.RMSNorm
    - float32 SDPA
    - j > i causal mask
    - get_prompt_embeds 逻辑（取 (9,18,27) 三层拼接）
    - enable_thinking=False
    """

    def __init__(self, ctx: Any, model_path: str, tokenizer_path: str = "", max_seq_len: int = 512):
        self.ctx = ctx
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path or model_path
        self.max_seq_len = max_seq_len
        self._tokenizer = None
        self._model = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path, trust_remote_code=True)
        return self._tokenizer

    def encode(self, texts: list[str]) -> Any:
        tokenizer = self.tokenizer
        chat_texts = []
        for text in texts:
            chat = [{"role": "user", "content": text}]
            chat_texts.append(tokenizer.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True, enable_thinking=False,
            ))
        tokens = tokenizer(chat_texts, padding="max_length", max_length=self.max_seq_len,
                          truncation=True, return_tensors="np")
        input_ids = tokens["input_ids"]
        attention_mask = tokens["attention_mask"]
        import mlx.core as mx
        input_ids = mx.array(input_ids, dtype=mx.int32)
        attention_mask = mx.array(attention_mask, dtype=mx.float32)
        return self._forward(input_ids, attention_mask)

    def _forward(self, input_ids, attention_mask):
        import mlx.core as mx
        if self._model is None:
            self._model = self._build_model()
        B, S = input_ids.shape
        hidden = self._model.embed_tokens(input_ids).astype(mx.float32)
        pos_ids = mx.broadcast_to(mx.arange(S, dtype=mx.int32)[None, :], (B, S))
        pos_emb = self._model.rotary_emb(hidden, pos_ids)

        # j > i causal mask + padding mask
        mask_dtype = hidden.dtype
        pad = mx.where(attention_mask == 1,
                      mx.zeros(attention_mask.shape, dtype=mask_dtype),
                      mx.full(attention_mask.shape, -float("inf"), dtype=mask_dtype))
        pad = mx.expand_dims(mx.expand_dims(pad, axis=1), axis=1)
        causal = _flux2_causal_mask(S, mask_dtype, B)
        attn_mask = causal + pad

        all_hidden = [hidden]
        for layer in self._model.layers:
            hidden, _ = layer(hidden, attn_mask, pos_emb, None)
            all_hidden.append(hidden)

        # 取 (9, 18, 27) 三层拼接
        outputs = [all_hidden[i] for i in (9, 18, 27)]
        stacked = mx.stack(outputs, axis=1)
        _, L, _, D = stacked.shape
        result = mx.transpose(stacked, (0, 2, 1, 3)).reshape(B, -1, L * D)
        return result.astype(mx.bfloat16)

    def _build_model(self):
        import mlx.core as mx
        from pathlib import Path
        import json
        d = Path(self.model_path)
        w = {}
        for sf in sorted(d.glob("*.safetensors")): w.update(dict(mx.load(str(sf))))
        cfg = {}
        if (d / "config.json").exists():
            with open(d / "config.json") as f: cfg = json.load(f)
        model = _Flux2EncoderModel(
            vocab_size=cfg.get("vocab_size", 151936),
            hidden_size=cfg.get("hidden_size", 4096),
            num_hidden_layers=cfg.get("num_hidden_layers", 36),
            num_attention_heads=cfg.get("num_attention_heads", 32),
            num_key_value_heads=cfg.get("num_key_value_heads", 8),
            intermediate_size=cfg.get("intermediate_size", 12288),
            head_dim=cfg.get("head_dim", 128),
            rope_theta=cfg.get("rope_theta", 1000000.0),
            rms_norm_eps=cfg.get("rms_norm_eps", 1e-6),
        )
        rw = {k[6:] if k.startswith("model.") else k: v for k, v in w.items()}
        model.load_weights(list(rw.items()), strict=False)
        mx.eval(model.parameters())
        return model


def _flux2_causal_mask(seq_len: int, dtype, batch_size: int):
    import mlx.core as mx
    idx = mx.arange(seq_len, dtype=mx.int32)
    j = mx.expand_dims(idx, axis=0)
    i = mx.expand_dims(idx, axis=1)
    zeros = mx.zeros((seq_len, seq_len), dtype=dtype)
    neginf = mx.full((seq_len, seq_len), -float("inf"), dtype=dtype)
    causal = mx.where(j > i, neginf, zeros)
    causal = mx.expand_dims(mx.expand_dims(causal, axis=0), axis=0)
    return mx.broadcast_to(causal, (batch_size, 1, seq_len, seq_len))
