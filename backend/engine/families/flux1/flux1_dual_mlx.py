"""Flux.1 dual text encoders — T5-XXL + CLIP-L + ``Flux1TextEncoder`` facade."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from backend.engine.common.model.base import _collect_params
from backend.engine.common.ops.attention import scaled_dot_product_attention_bhsd_mx
from backend.engine.common.bundle.layout import t5_encoder_bundle_paths
from backend.engine.families.flux1.weights import (
    nest_flux1_clip_weights,
    remap_flux1_clip_weights,
    remap_flux1_t5_weights,
)
from backend.engine.runtime._base import RuntimeContext

def _to_bfloat16_weights(weights: dict[str, Any]) -> dict[str, Any]:
    return {
        k: (v if v.dtype == mx.bfloat16 else v.astype(mx.bfloat16))
        for k, v in weights.items()
    }

class _T5LayerNorm:
    def __init__(self, dim: int, ctx: RuntimeContext):
        self.ctx = ctx
        self.weight = ctx.ones((dim,))
        self.variance_epsilon = 1e-6

    def parameters(self) -> dict[str, Any]:
        return {"weight": self.weight}

    def forward(self, hidden_states: Any) -> Any:
        # Match reference T5LayerNorm exactly: variance in fp32, then rescale original dtype.
        variance = mx.mean(
            mx.power(hidden_states.astype(mx.float32), 2),
            axis=-1,
            keepdims=True,
        )
        hidden_states = hidden_states * mx.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states

class _T5SelfAttention:
    def __init__(self, ctx: RuntimeContext, dim: int = 4096, heads: int = 64):
        nn = ctx
        self.ctx = ctx
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.relative_attention_bias = nn.Embedding(32, 64)
        self.o = nn.Linear(dim, dim, bias=False)

    def forward(self, hidden_states: Any) -> Any:
        query_states = self._shape(self.q(hidden_states))
        key_states = self._shape(self.k(hidden_states))
        value_states = self._shape(self.v(hidden_states))
        scores = mx.matmul(query_states, mx.transpose(key_states, (0, 1, 3, 2)))
        seq_length = hidden_states.shape[1]
        scores = scores + self._compute_bias(seq_length)
        attn_weights = mx.softmax(scores, axis=-1)
        attn_output = self._unshape(mx.matmul(attn_weights, value_states))
        return self.o(attn_output)

    def _shape(self, states: Any) -> Any:
        return mx.transpose(
            mx.reshape(states, (states.shape[0], -1, self.heads, self.head_dim)),
            (0, 2, 1, 3),
        )

    def _unshape(self, states: Any) -> Any:
        b = states.shape[0]
        return mx.reshape(
            mx.transpose(states, (0, 2, 1, 3)),
            (b, -1, self.dim),
        )

    def _compute_bias(self, seq_length: int) -> Any:
        context_position = mx.arange(0, seq_length, dtype=mx.int32)[:, None]
        memory_position = mx.arange(0, seq_length, dtype=mx.int32)[None, :]
        relative_position = memory_position - context_position
        relative_position_bucket = self._relative_position_bucket(relative_position)
        values = self.relative_attention_bias(relative_position_bucket)
        values = mx.transpose(values, (2, 0, 1))
        return mx.expand_dims(values, 0)

    @staticmethod
    def _relative_position_bucket(
        relative_position: Any,
        *,
        bidirectional: bool = True,
        num_buckets: int = 32,
        max_distance: int = 128,
    ) -> Any:
        # Reference halves num_buckets before the log-bin formula (see T5SelfAttention).
        relative_buckets = mx.zeros_like(relative_position)
        num_buckets //= 2
        relative_buckets = relative_buckets + mx.where(
            relative_position > 0, num_buckets, 0
        )
        relative_position = mx.abs(relative_position)
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact
        relative_position_if_large = max_exact + mx.floor(
            mx.log(relative_position.astype(mx.float32) / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).astype(mx.int32)
        relative_position_if_large = mx.minimum(
            relative_position_if_large,
            mx.full(relative_position_if_large.shape, num_buckets - 1),
        )
        relative_buckets = relative_buckets + mx.where(
            is_small, relative_position, relative_position_if_large
        )
        return relative_buckets

class _T5DenseReluDense:
    def __init__(self, ctx: RuntimeContext, dim: int = 4096, ff_dim: int = 10240):
        nn = ctx
        self.wi_0 = nn.Linear(dim, ff_dim, bias=False)
        self.wi_1 = nn.Linear(dim, ff_dim, bias=False)
        self.wo = nn.Linear(ff_dim, dim, bias=False)

    @staticmethod
    def _new_gelu(x: Any) -> Any:
        return (
            0.5
            * x
            * (1.0 + mx.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * mx.power(x, 3.0))))
        )

    def forward(self, hidden_states: Any) -> Any:
        hidden_gelu = self._new_gelu(self.wi_0(hidden_states))
        hidden_linear = self.wi_1(hidden_states)
        hidden_states = hidden_gelu * hidden_linear
        return self.wo(hidden_states)

class _T5Attention:
    def __init__(self, ctx: RuntimeContext):
        self.layer_norm = _T5LayerNorm(4096, ctx)
        self.SelfAttention = _T5SelfAttention(ctx)

    def forward(self, hidden_states: Any) -> Any:
        normed = self.layer_norm.forward(hidden_states)
        return hidden_states + self.SelfAttention.forward(normed)

class _T5FeedForward:
    def __init__(self, ctx: RuntimeContext):
        self.layer_norm = _T5LayerNorm(4096, ctx)
        self.DenseReluDense = _T5DenseReluDense(ctx)

    def forward(self, hidden_states: Any) -> Any:
        normed = self.layer_norm.forward(hidden_states)
        return hidden_states + self.DenseReluDense.forward(normed)

class _T5Block:
    def __init__(self, ctx: RuntimeContext):
        self.attention = _T5Attention(ctx)
        self.ff = _T5FeedForward(ctx)

    def forward(self, hidden_states: Any) -> Any:
        hidden_states = self.attention.forward(hidden_states)
        return self.ff.forward(hidden_states)

class Flux1T5Encoder:
    """reference-compatible T5-XXL for Flux.1 bundles (``text_encoder_2``)."""

    def __init__(
        self,
        ctx: RuntimeContext,
        model_path: str | Path,
        *,
        max_seq_len: int = 512,
        num_blocks: int = 24,
        tokenizer_path: str | Path | None = None,
    ):
        self.ctx = ctx
        self.model_path = Path(model_path)
        self.max_seq_len = max_seq_len
        self._tokenizer_path = Path(tokenizer_path) if tokenizer_path else self.model_path
        self._tokenizer = None

        nn = ctx
        self.shared = nn.Embedding(32128, 4096)
        self.t5_blocks = [_T5Block(ctx) for _ in range(num_blocks)]
        self.final_layer_norm = _T5LayerNorm(4096, ctx)

        self._param_map: dict[str, Any] = {}
        _collect_params(self, "", self._param_map)
        self._load_bundle_weights()

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import T5Tokenizer

            self._tokenizer = T5Tokenizer.from_pretrained(
                str(self._tokenizer_path), legacy=False,
            )
        return self._tokenizer

    def _load_bundle_weights(self) -> None:
        weights: dict[str, Any] = {}
        for sf in sorted(self.model_path.glob("*.safetensors")):
            weights.update(self.ctx.load_weights(str(sf)))
        remapped = remap_flux1_t5_weights(weights)
        missing = []
        for key, param in self._param_map.items():
            if key not in remapped:
                missing.append(key)
                continue
            tensor = remapped[key]
            if tuple(param.shape) != tuple(tensor.shape):
                raise RuntimeError(
                    f"Flux1T5Encoder weight shape mismatch for {key!r}: "
                    f"param {tuple(param.shape)} vs checkpoint {tuple(tensor.shape)}"
                )
            param[:] = tensor
        if missing:
            raise RuntimeError(
                f"Flux1T5Encoder: {len(missing)} parameter(s) missing from checkpoint "
                f"(first: {missing[:8]})"
            )
        for param in self._param_map.values():
            if param.dtype != mx.bfloat16:
                param[:] = param.astype(mx.bfloat16)
        if getattr(self.ctx, "backend", None) == "mlx":
            self.ctx.eval(*self._param_map.values())

    def encode(self, texts: list[str]) -> Any:
        tokenizer = self.tokenizer
        tokens = tokenizer(
            texts,
            padding="max_length",
            max_length=self.max_seq_len,
            truncation=True,
            return_tensors="np",
        )
        input_ids = self.ctx.array(tokens["input_ids"], dtype=mx.int32)
        return self.forward(input_ids)

    def forward(self, input_ids: Any) -> Any:
        hidden_states = self.shared(input_ids)
        for block in self.t5_blocks:
            hidden_states = block.forward(hidden_states)
        return self.final_layer_norm.forward(hidden_states)

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
    """reference-compatible CLIP text encoder for Flux.1 ``text_encoder`` bundle."""

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

class Flux1TextEncoder:
    """与 diffusers ``FluxPipeline`` 对齐：T5 → ``context_embedder``，CLIP pooled → ``time_text_embed``。"""

    def __init__(
        self,
        ctx: Any,
        bundle_root: str | Path,
        *,
        max_seq_len: int = 512,
        text_dim: int = 4096,
        pooled_dim: int = 768,
    ):
        del pooled_dim
        root = Path(bundle_root)
        t5_dir, t5_tok = t5_encoder_bundle_paths(root)
        clip_dir = root / "text_encoder"
        if not clip_dir.is_dir():
            raise RuntimeError(
                f"Flux.1 bundle missing CLIP text_encoder under {clip_dir}"
            )
        self._t5 = Flux1T5Encoder(
            ctx,
            t5_dir,
            max_seq_len=max_seq_len,
            tokenizer_path=t5_tok,
        )
        clip_tok = root / "tokenizer"
        self._clip = Flux1CLIPEncoder(
            ctx,
            str(clip_dir),
            tokenizer_path=str(clip_tok) if clip_tok.is_dir() else str(clip_dir),
        )
        self.ctx = ctx

    def encode(self, texts: list[str]) -> tuple[Any, Any]:
        """Returns ``(t5_hidden_states, clip_pooled)`` — second value is not an attention mask."""
        txt = self._t5.encode(texts)
        pooled, _hidden = self._clip.encode(texts)
        return txt, pooled

    def release_weights(self) -> None:
        """Drop T5 + CLIP MLX weights before DiT load (tokenizers kept)."""
        self._t5 = None
        self._clip = None
        clear_cache_fn = getattr(self.ctx, "clear_cache", None)
        if clear_cache_fn is not None:
            clear_cache_fn()
        else:
            import importlib
            importlib.import_module("mlx.core").clear_cache()
