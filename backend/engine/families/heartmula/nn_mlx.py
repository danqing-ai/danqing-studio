"""HeartMuLa merged module: nn_mlx.py."""
from __future__ import annotations

# --- from mlx/nn/conv.py ---

from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from backend.engine.common.mlx_runtime_fallback import random_uniform


class CausalConv1d(nn.Module):
    """1D Causal Convolution with left padding.

    This ensures the output at time t only depends on inputs at time <= t.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Size of the convolutional kernel.
        stride: Stride of the convolution.
        dilation: Dilation factor.
        groups: Number of groups for grouped convolution.
        bias: Whether to include a bias term.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups

        # Calculate padding for causal convolution
        self.padding = (kernel_size - 1) * dilation

        # MLX Conv1d: weight shape is (out_channels, kernel_size, in_channels // groups)
        scale = 1.0 / (in_channels * kernel_size) ** 0.5
        self.weight = random_uniform(
            None,
            low=-scale,
            high=scale,
            shape=(out_channels, kernel_size, in_channels // groups),
        )

        if bias:
            self.bias = mx.zeros((out_channels,))
        else:
            self.bias = None

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, length, channels).

        Returns:
            Output tensor of shape (batch, length, out_channels).
        """
        # Apply left padding for causal convolution
        if self.padding > 0:
            # Pad on the left (time dimension)
            x = mx.pad(x, [(0, 0), (self.padding, 0), (0, 0)])

        # Perform convolution
        y = mx.conv1d(
            x,
            self.weight,
            stride=self.stride,
            padding=0,  # We already applied causal padding
            dilation=self.dilation,
            groups=self.groups,
        )

        if self.bias is not None:
            y = y + self.bias

        return y


class WeightNormConv1d(nn.Module):
    """1D Convolution with weight normalization.

    Implements weight normalization as: w = g * (v / ||v||)
    where g is the magnitude and v is the direction.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Size of the convolutional kernel.
        stride: Stride of the convolution.
        padding: Padding to apply.
        dilation: Dilation factor.
        groups: Number of groups for grouped convolution.
        bias: Whether to include a bias term.
        causal: Whether to use causal (left) padding.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        causal: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.causal = causal

        # For causal convolution, override padding
        if causal:
            self.causal_padding = (kernel_size - 1) * dilation
        else:
            self.causal_padding = 0

        # Weight normalization: weight = g * (v / ||v||)
        # v is the direction vector
        scale = 1.0 / (in_channels * kernel_size) ** 0.5
        self.weight_v = random_uniform(
            None,
            low=-scale,
            high=scale,
            shape=(out_channels, kernel_size, in_channels // groups),
        )

        # g is the magnitude (per output channel)
        # Initialize to the norm of v
        v_norm = mx.sqrt(mx.sum(self.weight_v ** 2, axis=(1, 2), keepdims=True))
        self.weight_g = v_norm.squeeze((1, 2))

        if bias:
            self.bias = mx.zeros((out_channels,))
        else:
            self.bias = None

    def _get_normalized_weight(self) -> mx.array:
        """Compute the weight-normalized weight tensor."""
        # Normalize v
        v_norm = mx.sqrt(mx.sum(self.weight_v ** 2, axis=(1, 2), keepdims=True) + 1e-8)
        v_normalized = self.weight_v / v_norm

        # Scale by g
        weight = self.weight_g[:, None, None] * v_normalized
        return weight

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, length, channels).

        Returns:
            Output tensor of shape (batch, length, out_channels).
        """
        weight = self._get_normalized_weight()

        # Apply causal padding if needed
        if self.causal and self.causal_padding > 0:
            x = mx.pad(x, [(0, 0), (self.causal_padding, 0), (0, 0)])
            padding = 0
        else:
            padding = self.padding

        # Perform convolution
        y = mx.conv1d(
            x,
            weight,
            stride=self.stride,
            padding=padding,
            dilation=self.dilation,
            groups=self.groups,
        )

        if self.bias is not None:
            y = y + self.bias

        return y


class WeightNormConvTranspose1d(nn.Module):
    """1D Transposed Convolution with weight normalization.

    This layer supports two modes:
    1. Standard weight normalization: weight = g * v / ||v||
    2. Pre-computed weights: weight is stored directly (for converted PyTorch weights)

    The pre-computed mode is used when loading weights from PyTorch, where the
    weight normalization has already been applied and we just need to store
    the final weight matrix.

    For causal mode (matching PyTorch HeartCodec):
    - Uses padding=0
    - Trims output by stride samples from the end

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Size of the convolutional kernel.
        stride: Stride of the convolution.
        padding: Padding to apply.
        output_padding: Additional padding on the output.
        dilation: Dilation factor.
        groups: Number of groups for grouped convolution.
        bias: Whether to include a bias term.
        causal: Whether to use causal mode (trims stride samples from end).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        causal: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        self.groups = groups
        self.causal = causal

        # Weight storage - we store the pre-computed weight directly
        # Shape: (out_channels, kernel_size, in_channels // groups)
        scale = 1.0 / (in_channels * kernel_size) ** 0.5
        self.weight = random_uniform(
            None,
            low=-scale,
            high=scale,
            shape=(out_channels, kernel_size, in_channels // groups),
        )

        if bias:
            self.bias = mx.zeros((out_channels,))
        else:
            self.bias = None

    def _get_weight(self) -> mx.array:
        """Get the weight tensor."""
        return self.weight

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, length, channels).

        Returns:
            Output tensor of shape (batch, output_length, out_channels).
        """
        batch_size, seq_len, channels = x.shape
        weight = self._get_weight()

        # Compute output length
        output_len = (seq_len - 1) * self.stride - 2 * self.padding + self.kernel_size + self.output_padding

        # For transposed convolution, we:
        # 1. Insert zeros between input elements (upsampling)
        # 2. Apply convolution with flipped, transposed weight

        # Insert zeros between input samples (upsample) BEFORE padding
        if self.stride > 1:
            # Upsample by interleaving with zeros
            # [x0, x1, x2] with stride=2 -> [x0, 0, x1, 0, x2]
            # Approach: reshape and concatenate
            x_expanded = x[:, :, None, :]  # (batch, seq_len, 1, channels)
            zeros = mx.zeros((batch_size, seq_len, self.stride - 1, channels))
            interleaved = mx.concatenate([x_expanded, zeros], axis=2)  # (batch, seq_len, stride, channels)
            x = interleaved.reshape(batch_size, seq_len * self.stride, channels)
            # Trim trailing zeros: we want seq_len + (seq_len-1)*(stride-1) = seq_len*stride - stride + 1
            upsampled_len = seq_len * self.stride - self.stride + 1
            x = x[:, :upsampled_len, :]

        # For transposed conv, flip the kernel along the spatial dimension
        # Weight shape: (out_channels, kernel_size, in_channels // groups)
        # Keep the same shape, just flip the kernel
        weight_flipped = weight[:, ::-1, :]

        # Pad for transposed convolution
        pad_amount = self.kernel_size - 1 - self.padding
        if pad_amount > 0:
            x = mx.pad(x, [(0, 0), (pad_amount, pad_amount), (0, 0)])

        # Regular convolution with flipped weight
        y = mx.conv1d(
            x,
            weight_flipped,
            stride=1,
            padding=0,
            dilation=self.dilation,
            groups=self.groups,
        )

        # Handle output padding
        if self.output_padding > 0:
            y = mx.pad(y, [(0, 0), (0, self.output_padding), (0, 0)])

        # Adjust output size if needed
        if y.shape[1] > output_len:
            y = y[:, :output_len, :]
        elif y.shape[1] < output_len:
            pad_size = output_len - y.shape[1]
            y = mx.pad(y, [(0, 0), (0, pad_size), (0, 0)])

        # Causal mode: trim stride samples from end (matches PyTorch HeartCodec)
        if self.causal and self.stride > 0:
            y = y[:, :-self.stride, :]

        if self.bias is not None:
            y = y + self.bias

        return y


class Conv1d(nn.Module):
    """Standard 1D Convolution layer.

    Wrapper around MLX's conv1d with a familiar interface.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Size of the convolutional kernel.
        stride: Stride of the convolution.
        padding: Padding to apply.
        dilation: Dilation factor.
        groups: Number of groups for grouped convolution.
        bias: Whether to include a bias term.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        scale = 1.0 / (in_channels * kernel_size) ** 0.5
        self.weight = random_uniform(
            None,
            low=-scale,
            high=scale,
            shape=(out_channels, kernel_size, in_channels // groups),
        )

        if bias:
            self.bias = mx.zeros((out_channels,))
        else:
            self.bias = None

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, length, channels).

        Returns:
            Output tensor of shape (batch, output_length, out_channels).
        """
        y = mx.conv1d(
            x,
            self.weight,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )

        if self.bias is not None:
            y = y + self.bias

        return y

# --- from mlx/nn/rope.py ---

from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn


def _rotate_half_interleaved(x: mx.array) -> mx.array:
    """Interleaved RoPE rotation (torchtune format)."""
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    rotated = mx.stack([-x2, x1], axis=-1)
    return rotated.reshape(x.shape)


class RotaryPositionEmbedding(nn.Module):
    """Rotary Position Embedding.

    Implements RoPE as described in "RoFormer: Enhanced Transformer with
    Rotary Position Embedding" (Su et al., 2021).

    Args:
        dim: Dimension of the embedding (must be even).
        max_seq_len: Maximum sequence length for caching.
        base: Base for the frequency computation.
        scaling_factor: Optional scaling factor for extended context.
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 8192,
        base: float = 10000.0,
        scaling_factor: float = 1.0,
    ):
        super().__init__()
        assert dim % 2 == 0, "Dimension must be even for RoPE"

        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        self.scaling_factor = scaling_factor

        # Compute inverse frequencies
        inv_freq = 1.0 / (base ** (mx.arange(0, dim, 2).astype(mx.float32) / dim))
        self._inv_freq = inv_freq

        # Cache for cos/sin values
        self._cos_cache: Optional[mx.array] = None
        self._sin_cache: Optional[mx.array] = None
        self._cached_seq_len = 0

    def _build_cache(self, seq_len: int) -> None:
        """Build the cos/sin cache for the given sequence length.

        Uses interleaved format to match torchtune:
        cos/sin are computed for dim/2 frequencies and then interleaved
        to match the interleaved rotation format.
        """
        if seq_len <= self._cached_seq_len and self._cos_cache is not None:
            return

        # Create position indices
        positions = mx.arange(seq_len).astype(mx.float32) / self.scaling_factor

        # Compute frequencies: shape (seq_len, dim/2)
        freqs = mx.outer(positions, self._inv_freq)

        # Compute cos and sin for each frequency: shape (seq_len, dim/2)
        cos_half = mx.cos(freqs)
        sin_half = mx.sin(freqs)

        # Interleave to match rotation format: [cos0, cos0, cos1, cos1, ...]
        # Stack and reshape: (seq_len, dim/2, 2) -> (seq_len, dim)
        self._cos_cache = mx.stack([cos_half, cos_half], axis=-1).reshape(seq_len, -1)
        self._sin_cache = mx.stack([sin_half, sin_half], axis=-1).reshape(seq_len, -1)
        self._cached_seq_len = seq_len

    def __call__(
        self,
        q: mx.array,
        k: mx.array,
        offset: int = 0,
    ) -> Tuple[mx.array, mx.array]:
        """Apply rotary embeddings to query and key tensors.

        Args:
            q: Query tensor of shape (batch, seq_len, n_heads, head_dim) or
               (batch, n_heads, seq_len, head_dim).
            k: Key tensor with same shape as q.
            offset: Position offset for cached KV.

        Returns:
            Tuple of rotated (q, k) tensors.
        """
        # q, k shape: (batch, seq_len, n_heads, head_dim)
        seq_len = q.shape[1]

        # Build cache if needed
        self._build_cache(offset + seq_len)

        # Get the relevant slice of cos/sin
        assert self._cos_cache is not None and self._sin_cache is not None
        cos = self._cos_cache[offset : offset + seq_len]
        sin = self._sin_cache[offset : offset + seq_len]

        # Reshape for broadcasting with (batch, seq_len, n_heads, head_dim)
        # cos/sin are (seq_len, dim), need to become (1, seq_len, 1, dim)
        cos = cos[None, :, None, :]  # (1, seq_len, 1, dim)
        sin = sin[None, :, None, :]

        # Apply rotation
        q_rot = q * cos + _rotate_half_interleaved(q) * sin
        k_rot = k * cos + _rotate_half_interleaved(k) * sin

        return q_rot, k_rot

    def forward_one(
        self,
        x: mx.array,
        offset: int = 0,
    ) -> mx.array:
        """Apply rotary embeddings to a single tensor.

        Args:
            x: Input tensor of shape (batch, seq_len, n_heads, head_dim).
            offset: Position offset.

        Returns:
            Rotated tensor.
        """
        seq_len = x.shape[1]
        self._build_cache(offset + seq_len)

        assert self._cos_cache is not None and self._sin_cache is not None
        cos = self._cos_cache[offset : offset + seq_len]
        sin = self._sin_cache[offset : offset + seq_len]

        # Reshape for broadcasting with (batch, seq_len, n_heads, head_dim)
        cos = cos[None, :, None, :]  # (1, seq_len, 1, dim)
        sin = sin[None, :, None, :]

        return x * cos + _rotate_half_interleaved(x) * sin


def apply_rotary_pos_emb(
    q: mx.array,
    k: mx.array,
    cos: mx.array,
    sin: mx.array,
) -> Tuple[mx.array, mx.array]:
    """Apply rotary position embeddings to q and k.

    This is a functional version for cases where cos/sin are precomputed.
    Uses interleaved format to match torchtune.

    Args:
        q: Query tensor.
        k: Key tensor.
        cos: Cosine values (interleaved format).
        sin: Sine values (interleaved format).

    Returns:
        Tuple of rotated (q, k).
    """
    q_rot = q * cos + _rotate_half_interleaved(q) * sin
    k_rot = k * cos + _rotate_half_interleaved(k) * sin
    return q_rot, k_rot


def precompute_freqs_cis(
    dim: int,
    max_seq_len: int,
    base: float = 10000.0,
) -> Tuple[mx.array, mx.array]:
    """Precompute cosine and sine frequencies for RoPE.

    Uses interleaved format to match torchtune.

    Args:
        dim: Head dimension.
        max_seq_len: Maximum sequence length.
        base: Base for frequency computation.

    Returns:
        Tuple of (cos, sin) arrays of shape (max_seq_len, dim) in interleaved format.
    """
    inv_freq = 1.0 / (base ** (mx.arange(0, dim, 2).astype(mx.float32) / dim))
    positions = mx.arange(max_seq_len).astype(mx.float32)
    freqs = mx.outer(positions, inv_freq)

    # Compute cos/sin and interleave: [cos0, cos0, cos1, cos1, ...]
    cos_half = mx.cos(freqs)
    sin_half = mx.sin(freqs)
    cos = mx.stack([cos_half, cos_half], axis=-1).reshape(max_seq_len, -1)
    sin = mx.stack([sin_half, sin_half], axis=-1).reshape(max_seq_len, -1)
    return cos, sin

# --- from mlx/nn/kv_cache.py ---


from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import mlx.core as mx

CacheState = Union[Tuple[mx.array, mx.array], "KVLayerCache"]


@dataclass
class KVLayerCache:
    """Single-layer K/V buffer with running length (in-place append)."""

    k: mx.array
    v: mx.array
    length: int = 0

    @classmethod
    def create(
        cls,
        *,
        batch_size: int,
        max_seq_len: int,
        n_kv_heads: int,
        head_dim: int,
        dtype: mx.Dtype = mx.bfloat16,
    ) -> "KVLayerCache":
        shape = (batch_size, max_seq_len, n_kv_heads, head_dim)
        return cls(k=mx.zeros(shape, dtype=dtype), v=mx.zeros(shape, dtype=dtype), length=0)

    def as_tuple(self) -> Optional[Tuple[mx.array, mx.array]]:
        if self.length <= 0:
            return None
        return self.k[:, : self.length], self.v[:, : self.length]

    def append(
        self,
        k_new: mx.array,
        v_new: mx.array,
    ) -> Tuple[mx.array, mx.array]:
        new_len = int(k_new.shape[1])
        start = self.length
        end = start + new_len
        max_len = int(self.k.shape[1])
        if end > max_len:
            raise RuntimeError(
                f"KV cache overflow: need {end} positions, max_seq_len={max_len}"
            )
        self.k[:, start:end] = k_new
        self.v[:, start:end] = v_new
        self.length = end
        return self.k[:, :end], self.v[:, :end]

    def reset(self) -> None:
        self.length = 0


class KVCache:
    """Per-layer pre-allocated caches for a transformer stack."""

    def __init__(
        self,
        *,
        batch_size: int,
        max_seq_len: int,
        n_kv_heads: int,
        head_dim: int,
        n_layers: int,
        dtype: mx.Dtype = mx.bfloat16,
    ):
        self.max_seq_len = max_seq_len
        self.layers: List[KVLayerCache] = [
            KVLayerCache.create(
                batch_size=batch_size,
                max_seq_len=max_seq_len,
                n_kv_heads=n_kv_heads,
                head_dim=head_dim,
                dtype=dtype,
            )
            for _ in range(n_layers)
        ]

    def reset(self) -> None:
        for layer in self.layers:
            layer.reset()

    def __getitem__(self, index: int) -> KVLayerCache:
        """Layer access (compat with legacy per-layer cache lists)."""
        return self.layers[index]

    def __len__(self) -> int:
        return len(self.layers)


# Legacy aliases kept for imports elsewhere
RotatingKVCache = KVCache
HierarchicalKVCache = KVCache

# --- from mlx/nn/transformer.py ---

from typing import Optional, Tuple, Callable

import mlx.core as mx
import mlx.nn as nn

from backend.engine.common.attention import (
    build_causal_with_offset_bias,
    scaled_dot_product_attention_bhsd_mx,
)
from backend.engine.common.mlx_runtime_fallback import run_eval
from backend.engine.common.embeddings import sinusoidal_timestep_proj
from backend.engine.runtime.mlx import MLXContext
from backend.engine.common.text_encoders.qwen3_mlx import (
    LlamaMLP,
    MlxRMSNorm,
    MlxTimestepEmbeddingMLPWide,
)

_MLX_CTX = MLXContext()


def forward_llama_transformer_stack(
    *,
    layers: list,
    norm: MlxRMSNorm,
    hidden_states: mx.array,
    mask: Optional[mx.array] = None,
    cache: Optional[KVCache | list] = None,
) -> Tuple[mx.array, KVCache | list]:
    """Shared LLaMA-style stack forward for HeartMuLa backbone and decoder."""
    offset = 0
    if isinstance(cache, KVCache):
        offset = cache.layers[0].length
    elif cache is not None and cache[0] is not None:
        if isinstance(cache[0], KVLayerCache):
            offset = cache[0].length
        else:
            offset = cache[0][0].shape[1]

    if mask is None:
        seq_len = hidden_states.shape[1]
        mask = build_causal_with_offset_bias(mx, seq_len, offset, dtype=mx.float32)

    new_caches = []
    x = hidden_states
    for i, layer in enumerate(layers):
        if isinstance(cache, KVCache):
            layer_cache = cache.layers[i]
        elif cache is not None:
            layer_cache = cache[i]
        else:
            layer_cache = None
        x, new_cache = layer(x, mask=mask, cache=layer_cache, offset=offset)
        new_caches.append(new_cache)

    x = norm(x)
    if isinstance(cache, KVCache):
        return x, cache
    return x, new_caches


def _run_eval(*values) -> None:
    run_eval(None, *values)


HeartmulaRMSNorm = MlxRMSNorm
RMSNorm = MlxRMSNorm


class LlamaAttention(nn.Module):
    """Multi-head attention with rotary position embeddings.

    Implements the attention mechanism used in LLaMA models with
    support for grouped-query attention (GQA).

    Args:
        dim: Model dimension.
        n_heads: Number of attention heads.
        n_kv_heads: Number of key/value heads (for GQA). If None, uses n_heads.
        head_dim: Dimension of each attention head. If None, computed as dim // n_heads.
        max_seq_len: Maximum sequence length for RoPE cache.
        rope_base: Base for RoPE frequency computation.
        bias: Whether to use bias in projections.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        max_seq_len: int = 8192,
        rope_base: float = 10000.0,
        bias: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads or n_heads
        self.head_dim = head_dim or (dim // n_heads)
        self.n_rep = self.n_heads // self.n_kv_heads  # Repeat factor for GQA

        self.scale = self.head_dim ** -0.5

        # Projections
        self.q_proj = nn.Linear(dim, n_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(n_heads * self.head_dim, dim, bias=bias)

        # Rotary embeddings
        self.rope = RotaryPositionEmbedding(
            dim=self.head_dim,
            max_seq_len=max_seq_len,
            base=rope_base,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
        offset: int = 0,
    ) -> Tuple[mx.array, Tuple[mx.array, mx.array]]:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, dim).
            mask: Attention mask of shape (batch, 1, seq_len, total_len).
            cache: Optional tuple of (k, v) cached tensors.
            offset: Position offset for RoPE (when using cache).

        Returns:
            Tuple of (output, (k, v)) where output has shape (batch, seq_len, dim).
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape to (batch, seq_len, n_heads, head_dim)
        q = q.reshape(batch_size, seq_len, self.n_heads, self.head_dim)
        k = k.reshape(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        v = v.reshape(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        if isinstance(cache, KVLayerCache):
            offset = cache.length

        # Apply rotary embeddings
        q, k = self.rope(q, k, offset=offset)

        # Handle cache
        if isinstance(cache, KVLayerCache):
            k, v = cache.append(k, v)
            new_cache: KVLayerCache | Tuple[mx.array, mx.array] = cache
        elif cache is not None:
            k_cache, v_cache = cache
            k = mx.concatenate([k_cache, k], axis=1)
            v = mx.concatenate([v_cache, v], axis=1)
            _run_eval(k, v)
            new_cache = (k, v)
        else:
            new_cache = (k, v)

        # Transpose to (batch, n_heads, seq_len, head_dim) for SDPA
        # Note: SDPA handles GQA natively, no need to repeat K/V
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        # Use fused scaled dot-product attention (Flash Attention)
        # SDPA handles GQA automatically when n_kv_heads < n_heads
        output = scaled_dot_product_attention_bhsd_mx(
            mx,
            q,
            k,
            v,
            scale=self.scale,
            mask=mask,
        )

        # Reshape back to (batch, seq_len, dim)
        output = output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        output = self.o_proj(output)

        # Transpose K/V back to (batch, seq_len, n_kv_heads, head_dim) for cache
        k_out = k.transpose(0, 2, 1, 3)
        v_out = v.transpose(0, 2, 1, 3)

        if isinstance(new_cache, KVLayerCache):
            return output, new_cache
        return output, (k_out, v_out)


class LlamaTransformerBlock(nn.Module):
    """Transformer block with pre-normalization (LLaMA style).

    Consists of:
    1. RMSNorm + Multi-head attention + residual
    2. RMSNorm + MLP + residual

    Args:
        dim: Model dimension.
        n_heads: Number of attention heads.
        n_kv_heads: Number of key/value heads (for GQA).
        hidden_dim: MLP hidden dimension.
        max_seq_len: Maximum sequence length.
        norm_eps: Epsilon for RMSNorm.
        rope_base: Base for RoPE.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        max_seq_len: int = 8192,
        norm_eps: float = 1e-6,
        rope_base: float = 10000.0,
    ):
        super().__init__()

        self.attention = LlamaAttention(
            dim=dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            max_seq_len=max_seq_len,
            rope_base=rope_base,
        )
        self.mlp = LlamaMLP(dim=dim, hidden_dim=hidden_dim)

        self.attention_norm = HeartmulaRMSNorm(dim, eps=norm_eps)
        self.mlp_norm = HeartmulaRMSNorm(dim, eps=norm_eps)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
        offset: int = 0,
    ) -> Tuple[mx.array, Tuple[mx.array, mx.array]]:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, dim).
            mask: Attention mask.
            cache: Optional KV cache.
            offset: Position offset for RoPE.

        Returns:
            Tuple of (output, new_cache).
        """
        # Attention with residual
        h = self.attention_norm(x)
        attn_out, new_cache = self.attention(h, mask=mask, cache=cache, offset=offset)
        x = x + attn_out

        # MLP with residual
        x = x + self.mlp(self.mlp_norm(x))

        return x, new_cache


def build_llama_transformer_layers(
    *,
    n_layers: int,
    dim: int,
    n_heads: int,
    n_kv_heads: int,
    hidden_dim: int,
    max_seq_len: int,
    norm_eps: float,
    rope_base: float,
) -> list[LlamaTransformerBlock]:
    return [
        LlamaTransformerBlock(
            dim=dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            hidden_dim=hidden_dim,
            max_seq_len=max_seq_len,
            norm_eps=norm_eps,
            rope_base=rope_base,
        )
        for _ in range(n_layers)
    ]


def setup_llama_kv_cache(
    *,
    batch_size: int,
    max_seq_len: int,
    n_kv_heads: int,
    head_dim: int,
    n_layers: int,
) -> KVCache:
    return KVCache(
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        n_layers=n_layers,
    )


def llama_stack_from_config(cls: type, config: dict):
    return cls(
        dim=config["dim"],
        n_heads=config["n_heads"],
        n_kv_heads=config.get("n_kv_heads", config["n_heads"]),
        n_layers=config["n_layers"],
        hidden_dim=config.get("hidden_dim"),
        norm_eps=config.get("norm_eps", 1e-5),
        rope_base=config.get("rope_base", 10000.0),
    )


class LlamaTransformer(nn.Module):
    """Full LLaMA-style transformer.

    Stack of transformer blocks with embedding and output projection.

    Args:
        vocab_size: Size of the vocabulary.
        dim: Model dimension.
        n_layers: Number of transformer layers.
        n_heads: Number of attention heads.
        n_kv_heads: Number of key/value heads (for GQA).
        hidden_dim: MLP hidden dimension.
        max_seq_len: Maximum sequence length.
        norm_eps: Epsilon for RMSNorm.
        rope_base: Base for RoPE.
    """

    def __init__(
        self,
        vocab_size: int,
        dim: int,
        n_layers: int,
        n_heads: int,
        n_kv_heads: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        max_seq_len: int = 8192,
        norm_eps: float = 1e-6,
        rope_base: float = 10000.0,
        tie_word_embeddings: bool = False,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads or n_heads
        self.head_dim = dim // n_heads

        # Token embedding
        self.embed_tokens = nn.Embedding(vocab_size, dim)

        # Transformer layers
        self.layers = [
            LlamaTransformerBlock(
                dim=dim,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                hidden_dim=hidden_dim,
                max_seq_len=max_seq_len,
                norm_eps=norm_eps,
                rope_base=rope_base,
            )
            for _ in range(n_layers)
        ]

        # Output norm and projection
        self.norm = HeartmulaRMSNorm(dim, eps=norm_eps)

        if tie_word_embeddings:
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(dim, vocab_size, bias=False)

        self.tie_word_embeddings = tie_word_embeddings

    def __call__(
        self,
        input_ids: Optional[mx.array] = None,
        inputs_embeds: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache: Optional[list] = None,
    ) -> Tuple[mx.array, list]:
        """Forward pass.

        Args:
            input_ids: Token IDs of shape (batch, seq_len).
            inputs_embeds: Pre-computed embeddings of shape (batch, seq_len, dim).
            mask: Attention mask.
            cache: List of KV caches for each layer.

        Returns:
            Tuple of (logits, new_caches).
        """
        if inputs_embeds is None:
            assert input_ids is not None
            x = self.embed_tokens(input_ids)
        else:
            x = inputs_embeds

        batch_size, seq_len, _ = x.shape

        # Create causal mask if not provided
        if mask is None:
            cache_len = 0
            if cache is not None and cache[0] is not None:
                cache_len = int(cache[0][0].shape[1])
            mask = build_causal_with_offset_bias(mx, seq_len, cache_len, dtype=mx.float32)

        # Compute offset from cache
        offset = 0
        if cache is not None and cache[0] is not None:
            offset = cache[0][0].shape[1]

        # Process through layers
        new_caches = []
        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache is not None else None
            x, new_cache = layer(x, mask=mask, cache=layer_cache, offset=offset)
            new_caches.append(new_cache)

        # Final norm
        x = self.norm(x)

        # Output projection
        if self.lm_head is not None:
            logits = self.lm_head(x)
        else:
            # Tied embeddings
            logits = x @ self.embed_tokens.weight.T

        return logits, new_caches


class AdaLayerNormSingle(nn.Module):
    """Adaptive Layer Normalization for diffusion models.

    Modulates the normalization with scale and shift parameters
    predicted from a timestep embedding.

    Args:
        dim: Feature dimension.
        eps: Epsilon for normalization.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.norm = HeartmulaRMSNorm(dim, eps=eps)
        # Project timestep to scale and shift
        self.linear = nn.Linear(dim, 2 * dim)

    def __call__(self, x: mx.array, timestep_emb: mx.array) -> mx.array:
        """Apply adaptive normalization.

        Args:
            x: Input tensor of shape (batch, seq_len, dim).
            timestep_emb: Timestep embedding of shape (batch, dim).

        Returns:
            Normalized and modulated tensor.
        """
        # Get scale and shift from timestep
        scale_shift = self.linear(timestep_emb)
        scale, shift = mx.split(scale_shift, 2, axis=-1)

        # Normalize and modulate
        x = self.norm(x)
        x = x * (1 + scale[:, None, :]) + shift[:, None, :]

        return x


class TimestepEmbedding(MlxTimestepEmbeddingMLPWide):
    """Sinusoidal timestep embedding + wide MLP (HeartMuLa weight keys: ``linear1``/``linear2``)."""

    def __init__(self, dim: int, max_period: int = 10000):
        super().__init__(dim, dim, expansion=4)
        self.dim = dim
        self.max_period = max_period

    def __call__(self, timesteps: mx.array) -> mx.array:
        embedding = sinusoidal_timestep_proj(
            _MLX_CTX, timesteps, self.dim, sin_first=False, max_period=float(self.max_period)
        )
        return super().__call__(embedding)
