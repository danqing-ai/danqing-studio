"""Flux.1 T5-XXL — mflux ``T5Encoder`` 对齐（共享实现，族内 ``flux1/text_encoder`` 薄封装）。"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import mlx.core as mx
from backend.engine.common._base import _collect_params
from backend.engine.families.flux1.weights import remap_flux1_t5_weights
from backend.engine.runtime._base import RuntimeContext


class _T5LayerNorm:
    def __init__(self, dim: int, ctx: RuntimeContext):
        self.ctx = ctx
        self.weight = ctx.ones((dim,))
        self.variance_epsilon = 1e-6

    def parameters(self) -> dict[str, Any]:
        return {"weight": self.weight}

    def forward(self, hidden_states: Any) -> Any:
        # Match mflux T5LayerNorm exactly: variance in fp32, then rescale original dtype.
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
        # mflux halves num_buckets before the log-bin formula (see T5SelfAttention).
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
    """mflux-compatible T5-XXL for Flux.1 bundles (``text_encoder_2``)."""

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
