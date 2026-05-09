"""
文本编码器 — T5 / CLIP / Qwen3 编码器。

所有扩散模型的文本编码共享此实现。
参考 mflux 项目的 T5Encoder 实现和 mlx-video 的 T5 加载。
"""
from __future__ import annotations

from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn


class T5Encoder:
    """T5-XXL 文本编码器。

    用于: Flux1 / Flux2 / LTX / Wan / CogVideoX 全系列。
    文本 → [B, seq_len, text_dim] 嵌入。
    """

    def __init__(self, ctx: Any, model_path: str,
                 max_seq_len: int = 512, text_dim: int = 4096):
        self.ctx = ctx
        self.model_path = model_path
        self.max_seq_len = max_seq_len
        self.text_dim = text_dim
        self._tokenizer = None
        self._model = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import T5Tokenizer
            self._tokenizer = T5Tokenizer.from_pretrained(
                self.model_path, legacy=False,
            )
        return self._tokenizer

    def encode(self, texts: list[str]) -> Any:
        """编码文本列表。

        Args:
            texts: 文本列表

        Returns:
            prompt_embeds: [B, seq_len, text_dim]
        """
        tokenizer = self.tokenizer
        ctx = self.ctx

        tokens = tokenizer(
            texts,
            padding="max_length",
            max_length=self.max_seq_len,
            truncation=True,
            return_tensors="np",
        )

        input_ids = tokens["input_ids"]
        attention_mask = tokens["attention_mask"]

        # 用 ctx 创建张量
        if ctx.backend == "mlx":
            import mlx.core as mx
            input_ids = mx.array(input_ids, dtype=mx.int32)
            attention_mask = mx.array(attention_mask, dtype=mx.float32)
        else:
            import torch
            input_ids = torch.tensor(input_ids, dtype=torch.int32, device=ctx._device)
            attention_mask = torch.tensor(attention_mask, dtype=torch.float32, device=ctx._device)

        # 前向通过 T5
        embeds = self._forward(input_ids, attention_mask)
        return embeds

    def _forward(self, input_ids, attention_mask):
        """占位：T5 前向。具体实现由各后端完成。

        MLX T5 → mlx_lm.models.t5.T5Model
        CUDA T5 → transformers.T5EncoderModel
        """
        if self.ctx.backend == "mlx":
            return self._forward_mlx(input_ids, attention_mask)
        else:
            return self._forward_torch(input_ids, attention_mask)

    def _forward_mlx(self, input_ids, attention_mask):
        import mlx.core as mx
        try:
            from mlx_lm.models.t5 import T5Model, T5Config
        except ImportError:
            raise ImportError("mlx_lm not installed. Install with: pip install mlx-lm")
        if self._model is None:
            config = T5Config.from_pretrained(self.model_path)
            self._model = T5Model(config)
            self._model.load_weights(str(self.model_path))
            mx.eval(self._model.parameters())
        return self._model(input_ids, attention_mask)

    def _forward_torch(self, input_ids, attention_mask):
        import torch
        from transformers import T5EncoderModel

        if self._model is None:
            self._model = T5EncoderModel.from_pretrained(
                self.model_path, torch_dtype=torch.float32
            ).to(self.ctx._device)
            self._model.eval()

        with torch.no_grad():
            outputs = self._model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        return outputs.last_hidden_state


class CLIPEncoder:
    """CLIP 文本/图像编码器。

    用于: Flux1 系列的双编码器（T5 + CLIP）。
    """

    def __init__(self, ctx: Any, model_path: str,
                 max_seq_len: int = 77, embed_dim: int = 768):
        self.ctx = ctx
        self.model_path = model_path
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        self._tokenizer = None
        self._model = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import CLIPTokenizer
            self._tokenizer = CLIPTokenizer.from_pretrained(self.model_path)
        return self._tokenizer

    def encode(self, texts: list[str]) -> tuple[Any, Any]:
        """编码文本 → (pooled_output, hidden_states)。

        Returns:
            pooled: [B, embed_dim] — 用于 timestep embedding 注入
            hidden: [B, seq_len, embed_dim] — 用于交叉注意力
        """
        tokenizer = self.tokenizer
        ctx = self.ctx

        tokens = tokenizer(
            texts,
            padding="max_length",
            max_length=self.max_seq_len,
            truncation=True,
            return_tensors="np",
        )

        input_ids = tokens["input_ids"]

        if ctx.backend == "mlx":
            import mlx.core as mx
            input_ids = mx.array(input_ids, dtype=mx.int32)
        else:
            import torch
            input_ids = torch.tensor(input_ids, dtype=torch.int32, device=ctx._device)

        return self._forward(input_ids)

    def _forward(self, input_ids):
        if self.ctx.backend == "mlx":
            return self._forward_mlx(input_ids)
        else:
            return self._forward_torch(input_ids)

    def _forward_mlx(self, input_ids):
        # MLX CLIP 文本编码器
        import mlx.core as mx
        from mlx_lm.models.clip import CLIPTextModel, CLIPTextConfig

        if self._model is None:
            config = CLIPTextConfig.from_pretrained(self.model_path)
            self._model = CLIPTextModel(config)
            self._model.load_weights(str(self.model_path))
            mx.eval(self._model.parameters())

        pooled, hidden = self._model(input_ids)
        return pooled, hidden

    def _forward_torch(self, input_ids):
        import torch
        from transformers import CLIPTextModel

        if self._model is None:
            self._model = CLIPTextModel.from_pretrained(
                self.model_path, torch_dtype=torch.float32
            ).to(self.ctx._device)
            self._model.eval()

        with torch.no_grad():
            outputs = self._model(input_ids=input_ids)
        return outputs.pooler_output, outputs.last_hidden_state


class Qwen3TextEncoder:
    """Qwen3 文本编码器 — 统一 z_image / flux2。

    两种输出模式（hidden_state_layers 控制）：
    - None（z_image）: 返回倒数第二隐层，shape [B, seq_len, hidden_size]
    - (9, 18, 27)（flux2）: 返回指定层拼接，shape [B, seq_len, len(layers) * hidden_size]

    参考 mflux Qwen3TextEncoder + TextEncoder 实现，合并为一套代码。
    """

    def __init__(
        self,
        ctx: Any,
        model_path: str,
        max_seq_len: int = 512,
        tokenizer_path: str = "",
        hidden_state_layers: tuple[int, ...] | None = None,
        enable_thinking: bool = False,
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
                self.tokenizer_path, trust_remote_code=True,
            )
        return self._tokenizer

    def encode(self, texts: list[str]) -> Any:
        """编码文本列表。

        Returns:
            hidden_state_layers=None: [B, seq_len, hidden_size]（倒数第二层）
            hidden_state_layers=...:   [B, seq_len, len*layers, hidden_size]（拼接）
        """
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
                chat_texts, padding="max_length", max_length=self.max_seq_len,
                truncation=True, return_tensors="np",
            )
        else:
            tokens = tokenizer(
                texts, padding="max_length", max_length=self.max_seq_len,
                truncation=True, return_tensors="np",
            )
        input_ids = tokens["input_ids"]
        attention_mask = tokens["attention_mask"]

        if ctx.backend == "mlx":
            import mlx.core as mx
            input_ids = mx.array(input_ids, dtype=mx.int32)
            attention_mask = mx.array(attention_mask, dtype=mx.float32)
        else:
            import torch
            input_ids = torch.tensor(input_ids, dtype=torch.int32, device=ctx._device)
            attention_mask = torch.tensor(attention_mask, dtype=torch.float32, device=ctx._device)

        return self._forward(input_ids, attention_mask)

    def _forward(self, input_ids, attention_mask):
        if self.ctx.backend == "mlx":
            return self._forward_mlx(input_ids, attention_mask)
        else:
            return self._forward_torch(input_ids, attention_mask)

    def _forward_mlx(self, input_ids, attention_mask):
        import mlx.core as mx

        if self._model is None:
            self._model = self._build_mlx_model()

        # 统一手动前向：使用 model._get_causal_mask 确保 mask 与 mflux 一致
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
            # flux2 模式：拼接指定层输出 → [B, seq_len, L * D]
            layer_outputs = [all_hidden_states[i] for i in self.hidden_state_layers]
            stacked = mx.stack(layer_outputs, axis=1)
            B, L, S, D = stacked.shape
            result = mx.transpose(stacked, (0, 2, 1, 3)).reshape(B, S, L * D)
        else:
            # z_image 模式：返回倒数第二层
            result = all_hidden_states[-2]

        # 与 mflux 一致：输出转为 bfloat16（非 hidden_states.dtype=float32）
        import mlx.core as mx
        return result.astype(mx.bfloat16)

    def _build_mlx_model(self):
        """从 config.json 读取维度，构建 Qwen3 解码器。"""
        import mlx.core as mx
        from pathlib import Path
        import json

        model_dir = Path(self.model_path)
        weights = {}
        for sf in sorted(model_dir.glob("*.safetensors")):
            weights.update(dict(mx.load(str(sf))))

        config_path = model_dir / "config.json"
        config = {}
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)

        model = _Qwen3EncoderModel(
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

        remapped = {}
        for key, tensor in weights.items():
            new_key = key[6:] if key.startswith("model.") else key
            remapped[new_key] = tensor

        model.load_weights(list(remapped.items()), strict=False)
        mx.eval(model.parameters())
        return model

    def _forward_torch(self, input_ids, attention_mask):
        import torch
        from transformers import AutoModel

        if self._model is None:
            self._model = AutoModel.from_pretrained(
                self.model_path,
                torch_dtype=torch.float32,
                trust_remote_code=True,
            ).to(self.ctx._device)
            self._model.eval()

        with torch.no_grad():
            outputs = self._model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        last = outputs.last_hidden_state
        if self.hidden_state_layers is not None:
            B, S, D = last.shape
            L = len(self.hidden_state_layers)
            last = last.unsqueeze(2).expand(-1, -1, L, -1).reshape(B, S, L * D)
        return last


# ============================================================================
# _Qwen3Decoder — Qwen3 解码器模型，供 Qwen3TextEncoder 内部使用。
# 与 mflux 的 Qwen3VLDecoderLayer 架构一致，z_image / flux2 通用。
# ============================================================================

# _Qwen3EncoderModel — Qwen3 文本编码器 MLX 模型

class _Qwen3EncoderModel(nn.Module):
    """Qwen3 文本编码器 MLX 模型。"""

    def __init__(self, vocab_size=151936, hidden_size=2560, num_hidden_layers=36,
                 num_attention_heads=32, num_key_value_heads=8, intermediate_size=9728,
                 head_dim=128, max_position_embeddings=40960, rope_theta=1000000.0,
                 rms_norm_eps=1e-6):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = [
            _Qwen3EncoderLayer(hidden_size, num_attention_heads, num_key_value_heads,
                                intermediate_size, head_dim, rms_norm_eps)
            for _ in range(num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(hidden_size, eps=rms_norm_eps)
        self.rotary_emb = _Qwen3EncoderRotaryEmbedding(dim=head_dim, base=rope_theta)

    def __call__(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.shape
        hidden_states = self.embed_tokens(input_ids).astype(mx.float32)
        position_ids = mx.broadcast_to(mx.arange(seq_len, dtype=mx.int32)[None, :], (batch_size, seq_len))
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        causal_mask = _Qwen3EncoderModel._get_causal_mask(attention_mask, batch_size, hidden_states, seq_len)
        all_hidden_states = [hidden_states]
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, attention_mask=causal_mask, position_embeddings=position_embeddings)
            all_hidden_states.append(hidden_states)
        return all_hidden_states[-2].astype(input_ids.dtype)

    @staticmethod
    def _get_causal_mask(attention_mask, batch_size, hidden_states, seq_len):
        causal_mask = _Qwen3EncoderModel._create_causal_mask(seq_len, hidden_states.dtype)
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
        causal_mask = mx.where(mask, mx.zeros((seq_len, seq_len), dtype=dtype), mx.full((seq_len, seq_len), float("-inf"), dtype=dtype))
        return causal_mask[None, None, :, :]


class _Qwen3EncoderRotaryEmbedding(nn.Module):
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


class _Qwen3EncoderMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class _Qwen3EncoderAttention(nn.Module):
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
            q_embed = (q * cos) + (_Qwen3EncoderAttention._rotate_half(q) * sin)
            k_embed = (k * cos) + (_Qwen3EncoderAttention._rotate_half(k) * sin)
            q, k = q_embed, k_embed
        if self.num_kv_groups > 1:
            k = mx.repeat(k, self.num_kv_groups, axis=2)
            v = mx.repeat(v, self.num_kv_groups, axis=2)
        q = mx.transpose(q, axes=(0, 2, 1, 3))
        k = mx.transpose(k, axes=(0, 2, 1, 3))
        v = mx.transpose(v, axes=(0, 2, 1, 3))
        from mlx.core.fast import scaled_dot_product_attention
        # mflux flux2 Qwen3VLAttention 在 SDPA 前强制转 float32 保证精度
        q_f32, k_f32, v_f32 = q.astype(mx.float32), k.astype(mx.float32), v.astype(mx.float32)
        attn_output = scaled_dot_product_attention(q_f32, k_f32, v_f32, scale=self.scale, mask=attention_mask)
        attn_output = attn_output.astype(q.dtype)
        attn_output = mx.transpose(attn_output, axes=(0, 2, 1, 3)).reshape(batch_size, seq_len, -1)
        return self.o_proj(attn_output)

    @staticmethod
    def _rotate_half(x):
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return mx.concatenate([-x2, x1], axis=-1)


class _Qwen3EncoderLayer(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, num_key_value_heads,
                 intermediate_size, head_dim, rms_norm_eps=1e-6):
        super().__init__()
        self.input_layernorm = nn.RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(hidden_size, eps=rms_norm_eps)
        self.self_attn = _Qwen3EncoderAttention(hidden_size, num_attention_heads, num_key_value_heads, head_dim)
        self.mlp = _Qwen3EncoderMLP(hidden_size, intermediate_size)

    def __call__(self, hidden_states, attention_mask=None, position_embeddings=None):
        residual = hidden_states
        hidden_states = self.self_attn(self.input_layernorm(hidden_states), attention_mask, position_embeddings)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.mlp(self.post_attention_layernorm(hidden_states))
        return residual + hidden_states


class _Qwen3MLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class _Qwen3DecoderLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, num_kv_heads, head_dim, intermediate_size, rms_norm_eps=1e-6):
        super().__init__()
        self.input_layernorm = nn.RMSNorm(hidden_size, eps=rms_norm_eps)
        self.self_attn = _Qwen3Attention(hidden_size, num_heads, num_kv_heads, head_dim, rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(hidden_size, eps=rms_norm_eps)
        self.mlp = _Qwen3MLP(hidden_size, intermediate_size)

    def __call__(self, hidden_states, attention_mask=None, position_embeddings=None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask, position_embeddings)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class _Qwen3RotaryEmbedding(nn.Module):
    def __init__(self, dim, base=1000000.0):
        super().__init__()
        self.inv_freq = 1.0 / (base ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))

    def __call__(self, x, position_ids):
        if position_ids.ndim == 1:
            position_ids = mx.expand_dims(position_ids, 0)
        inv_freq = mx.expand_dims(mx.expand_dims(self.inv_freq, 0), 0)
        pos = mx.expand_dims(position_ids.astype(mx.float32), -1)
        freqs = pos * inv_freq
        emb = mx.concatenate([freqs, freqs], axis=-1)
        return mx.cos(emb).astype(x.dtype), mx.sin(emb).astype(x.dtype)


class _Qwen3Decoder(nn.Module):
    """Qwen3 解码器，供 Qwen3TextEncoder 内部驱动。"""

    def __init__(self, vocab_size=151936, hidden_size=2560, num_hidden_layers=36,
                 num_attention_heads=32, num_key_value_heads=8, intermediate_size=9728,
                 head_dim=128, max_position_embeddings=40960, rope_theta=1000000.0,
                 rms_norm_eps=1e-6):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = [
            _Qwen3DecoderLayer(
                hidden_size, num_attention_heads, num_key_value_heads,
                head_dim, intermediate_size, rms_norm_eps,
            )
            for _ in range(num_hidden_layers)
        ]
        self.rotary_emb = _Qwen3RotaryEmbedding(dim=head_dim, base=rope_theta)
