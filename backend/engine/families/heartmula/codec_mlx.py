"""HeartMuLa merged module: codec_mlx.py."""
from __future__ import annotations

# --- from mlx/heartcodec/configuration.py ---

from dataclasses import dataclass, field
from typing import List, Union
import json
from pathlib import Path


@dataclass
class HeartCodecConfig:
    """Configuration for HeartCodec neural audio codec.

    HeartCodec is a 12.5Hz neural audio codec that combines:
    1. A scalar quantization codec (SQ Codec) for audio encoding/decoding
    2. A flow matching decoder with LlamaTransformer for high-quality synthesis

    Attributes:
        # RVQ (Residual Vector Quantization) Parameters
        dim: Embedding dimension for codebooks (default: 512).
        codebook_size: Number of entries per codebook (default: 8192).
        decay: EMA decay rate for codebook updates (default: 0.9).
        commitment_weight: Weight for commitment loss (default: 1.0).
        threshold_ema_dead_code: Threshold for replacing dead codes (default: 2).
        use_cosine_sim: Use cosine similarity for codebook lookup (default: False).
        codebook_dim: Dimension of codebook entries (default: 32).
        num_quantizers: Number of RVQ codebooks (default: 8).

        # Diffusion Transformer Parameters
        attention_head_dim: Dimension per attention head (default: 64).
        in_channels: Input channels to flow matching (default: 1024).
        norm_type: Normalization type (default: "ada_norm_single").
        num_attention_heads: Number of attention heads (default: 24).
        num_layers: Number of transformer layers in main stack (default: 24).
        num_layers_2: Number of layers in second stack (default: 6).
        out_channels: Output channels from flow matching (default: 256).

        # SQ Codec Parameters
        num_bands: Number of frequency bands (default: 1).
        sample_rate: Audio sample rate in Hz (default: 48000).
        causal: Use causal convolutions (default: True).
        num_samples: Number of samples for codec (default: 2).
        downsample_factors: Downsampling factors per stage.
        downsample_kernel_sizes: Kernel sizes for downsampling.
        upsample_factors: Upsampling factors per stage.
        upsample_kernel_sizes: Kernel sizes for upsampling.
        latent_hidden_dim: Hidden dimension in latent space (default: 128).
        default_kernel_size: Default conv kernel size (default: 7).
        delay_kernel_size: Delay conv kernel size (default: 5).
        init_channel: Initial channel count (default: 64).
        res_kernel_size: Residual block kernel size (default: 7).
    """

    # Model type
    model_type: str = "heartcodec"

    # RVQ Parameters
    dim: int = 512
    codebook_size: int = 8192
    decay: float = 0.9
    commitment_weight: float = 1.0
    threshold_ema_dead_code: int = 2
    use_cosine_sim: bool = False
    codebook_dim: int = 32
    num_quantizers: int = 8

    # Diffusion Transformer Parameters
    attention_head_dim: int = 64
    in_channels: int = 1024
    norm_type: str = "ada_norm_single"
    num_attention_heads: int = 24
    num_layers: int = 24
    num_layers_2: int = 6
    out_channels: int = 128  # Must match latent_hidden_dim for scalar model

    # SQ Codec Parameters
    num_bands: int = 1
    sample_rate: int = 48000
    causal: bool = True
    num_samples: int = 2
    downsample_factors: List[int] = field(default_factory=lambda: [4, 4, 8, 8])
    downsample_kernel_sizes: List[int] = field(default_factory=lambda: [8, 8, 16, 16])
    upsample_factors: List[int] = field(default_factory=lambda: [8, 8, 4, 4])
    upsample_kernel_sizes: List[int] = field(default_factory=lambda: [16, 16, 8, 8])
    latent_hidden_dim: int = 128
    default_kernel_size: int = 7
    delay_kernel_size: int = 5
    init_channel: int = 64
    res_kernel_size: int = 7

    @classmethod
    def from_pretrained(cls, path: str) -> "HeartCodecConfig":
        """Load configuration from a pretrained model directory.

        Args:
            path: Path to the model directory containing config.json.

        Returns:
            HeartCodecConfig instance.
        """
        config_path = Path(path) / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r") as f:
            config_dict = json.load(f)

        return cls(**{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__})

    def save_pretrained(self, path: Union[str, Path]) -> None:
        """Save configuration to a directory.

        Args:
            path: Path to save the configuration.
        """
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)

        config_dict = {
            k: getattr(self, k) for k in self.__dataclass_fields__
        }

        with open(save_path / "config.json", "w") as f:
            json.dump(config_dict, f, indent=2)

    def to_dict(self) -> dict:
        """Convert configuration to dictionary.

        Returns:
            Configuration as dictionary.
        """
        return {k: getattr(self, k) for k in self.__dataclass_fields__}

    @property
    def total_stride(self) -> int:
        """Total downsampling stride of the codec."""
        stride = 1
        for f in self.downsample_factors:
            stride *= f
        return stride

    @property
    def frame_rate(self) -> float:
        """Frame rate of the codec codes in Hz.

        HeartCodec operates at 12.5 Hz code rate:
        - Codes: 12.5 Hz
        - Flow matching latent: 25 Hz (2x codes)
        - Scalar model stride: 960
        - Scalar model num_samples: 2
        - Audio: 48000 Hz

        The formula is: sample_rate / (total_stride * num_samples * 2)
        where the extra *2 accounts for the flow matching 2x upsampling.
        """
        return self.sample_rate / (self.total_stride * self.num_samples * 2)

    @property
    def transformer_dim(self) -> int:
        """Dimension of the transformer in flow matching decoder."""
        return self.num_attention_heads * self.attention_head_dim

# --- from mlx/heartcodec/activations.py ---

import mlx.core as mx
import mlx.nn as nn


class Snake(nn.Module):
    """Snake activation function.

    Defined as: x + (1/α) * sin²(αx)

    This activation is particularly effective for audio generation tasks
    as it introduces periodic inductive bias.

    Reference: "Neural Networks Fail to Learn Periodic Functions and
    How to Fix It" (Ziyin et al., 2020)

    Args:
        channels: Number of channels for learnable alpha parameter.
        alpha_init: Initial value for alpha. Higher values give more
            oscillations in the activation.
    """

    def __init__(self, channels: int, alpha_init: float = 1.0):
        super().__init__()
        self.channels = channels
        # Learnable frequency parameter per channel
        self.alpha = mx.ones((channels,)) * alpha_init

    def __call__(self, x: mx.array) -> mx.array:
        """Apply Snake activation.

        Args:
            x: Input tensor of shape (..., channels).

        Returns:
            Activated tensor with same shape.
        """
        # Ensure alpha broadcasts correctly
        alpha = self.alpha
        # Reshape for broadcasting if needed
        while alpha.ndim < x.ndim:
            alpha = alpha[None, ...]

        # Snake: x + (1/α) * sin²(αx)
        sin_term = mx.sin(alpha * x)
        return x + (1.0 / (alpha + 1e-8)) * (sin_term * sin_term)


class SnakeBeta(nn.Module):
    """Snake activation with separate beta parameter.

    Defined as: x + (1/β) * sin²(αx)

    Separates the frequency (α) and amplitude (β) parameters.

    Args:
        channels: Number of channels.
        alpha_init: Initial frequency parameter.
        beta_init: Initial amplitude parameter.
    """

    def __init__(
        self,
        channels: int,
        alpha_init: float = 1.0,
        beta_init: float = 1.0,
    ):
        super().__init__()
        self.channels = channels
        self.alpha = mx.ones((channels,)) * alpha_init
        self.beta = mx.ones((channels,)) * beta_init

    def __call__(self, x: mx.array) -> mx.array:
        """Apply SnakeBeta activation.

        Args:
            x: Input tensor.

        Returns:
            Activated tensor.
        """
        alpha = self.alpha
        beta = self.beta

        while alpha.ndim < x.ndim:
            alpha = alpha[None, ...]
            beta = beta[None, ...]

        sin_term = mx.sin(alpha * x)
        return x + (1.0 / (beta + 1e-8)) * (sin_term * sin_term)


def snake(x: mx.array, alpha: float = 1.0) -> mx.array:
    """Functional Snake activation.

    Args:
        x: Input tensor.
        alpha: Frequency parameter.

    Returns:
        Activated tensor.
    """
    sin_term = mx.sin(alpha * x)
    return x + (1.0 / alpha) * (sin_term * sin_term)


class PReLU(nn.Module):
    """Parametric ReLU activation.

    Defined as: max(0, x) + α * min(0, x)

    Args:
        num_parameters: Number of learnable parameters.
        init: Initial value for negative slope.
    """

    def __init__(self, num_parameters: int = 1, init: float = 0.25):
        super().__init__()
        self.weight = mx.ones((num_parameters,)) * init

    def __call__(self, x: mx.array) -> mx.array:
        """Apply PReLU.

        Args:
            x: Input tensor.

        Returns:
            Activated tensor.
        """
        weight = self.weight
        while weight.ndim < x.ndim:
            weight = weight[None, ...]

        return mx.maximum(0, x) + weight * mx.minimum(0, x)


class Swish(nn.Module):
    """Swish (SiLU) activation.

    Defined as: x * sigmoid(x)
    """

    def __call__(self, x: mx.array) -> mx.array:
        """Apply Swish activation.

        Args:
            x: Input tensor.

        Returns:
            Activated tensor.
        """
        return x * mx.sigmoid(x)

# --- from mlx/heartcodec/quantizer.py ---

from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from backend.engine.common.mlx_runtime_fallback import random_normal


class EMACodebook(nn.Module):
    """EMA-updated codebook matching PyTorch structure.

    Stores embeddings in _codebook.embed to match PyTorch weights.
    """

    def __init__(self, codebook_size: int, codebook_dim: int):
        super().__init__()
        # PyTorch stores as (1, codebook_size, codebook_dim)
        self.embed = random_normal(None, (1, codebook_size, codebook_dim)) * 0.02
        # EMA tracking (not used in inference, but needed for weight loading)
        self.cluster_size = mx.zeros((1, codebook_size))
        self.embed_avg = mx.zeros((1, codebook_size, codebook_dim))
        self.initted = mx.zeros((1,))


class VectorQuantizer(nn.Module):
    """Single-level Vector Quantizer.

    Maps continuous embeddings to discrete codes using a codebook.

    Args:
        codebook_size: Number of codes in the codebook.
        codebook_dim: Dimension of each code vector.
        use_cosine_sim: Use cosine similarity instead of L2 distance.
    """

    def __init__(
        self,
        codebook_size: int = 8192,
        codebook_dim: int = 512,
        use_cosine_sim: bool = False,
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.use_cosine_sim = use_cosine_sim

        # Use EMACodebook structure to match PyTorch weights
        # Note: Named 'codebook' (not '_codebook') so MLX tracks it as a parameter
        # PyTorch uses '_codebook' but we map in conversion
        self.codebook = EMACodebook(codebook_size, codebook_dim)

    def get_codebook_embeddings(self) -> mx.array:
        """Get the codebook embeddings (2D view)."""
        return self.codebook.embed[0]  # Remove leading batch dim

    def encode(self, x: mx.array) -> mx.array:
        """Encode continuous embeddings to discrete codes.

        Args:
            x: Input embeddings of shape (..., codebook_dim).

        Returns:
            Codes of shape (...).
        """
        # Flatten for distance computation
        original_shape = x.shape[:-1]
        x_flat = x.reshape(-1, self.codebook_dim)

        # Get codebook embeddings
        cb = self.get_codebook_embeddings()

        if self.use_cosine_sim:
            # Normalize both x and codebook
            x_norm = x_flat / (mx.linalg.norm(x_flat, axis=-1, keepdims=True) + 1e-8)
            cb_norm = cb / (mx.linalg.norm(cb, axis=-1, keepdims=True) + 1e-8)
            # Cosine similarity (higher is better, so negate for argmin)
            distances = -mx.matmul(x_norm, cb_norm.T)
        else:
            # L2 distance
            # ||x - c||^2 = ||x||^2 + ||c||^2 - 2 * x.c
            x_sq = mx.sum(x_flat ** 2, axis=-1, keepdims=True)
            cb_sq = mx.sum(cb ** 2, axis=-1)
            distances = x_sq + cb_sq - 2 * mx.matmul(x_flat, cb.T)

        # Find nearest code
        codes = mx.argmin(distances, axis=-1)
        return codes.reshape(original_shape)

    def decode(self, codes: mx.array) -> mx.array:
        """Decode discrete codes to continuous embeddings.

        Args:
            codes: Code indices of shape (...).

        Returns:
            Embeddings of shape (..., codebook_dim).
        """
        # Lookup embeddings from codebook
        cb = self.get_codebook_embeddings()
        return cb[codes]

    def __call__(
        self,
        x: mx.array,
    ) -> Tuple[mx.array, mx.array, mx.array]:
        """Forward pass with quantization.

        Args:
            x: Input embeddings of shape (..., codebook_dim).

        Returns:
            Tuple of (quantized, codes, commitment_loss).
        """
        codes = self.encode(x)
        quantized = self.decode(codes)

        # Commitment loss (for training)
        commitment_loss = mx.mean((x - mx.stop_gradient(quantized)) ** 2)

        # Straight-through estimator
        quantized = x + mx.stop_gradient(quantized - x)

        return quantized, codes, commitment_loss


class ResidualVQ(nn.Module):
    """Residual Vector Quantization.

    Applies multiple levels of vector quantization, where each level
    quantizes the residual from the previous level.

    Args:
        num_quantizers: Number of quantization levels (codebooks).
        codebook_size: Number of codes per codebook.
        codebook_dim: Dimension of code vectors (internal VQ dimension).
        dim: Input/output dimension (projected to/from codebook_dim).
        use_cosine_sim: Use cosine similarity for quantization.
    """

    def __init__(
        self,
        num_quantizers: int = 8,
        codebook_size: int = 8192,
        codebook_dim: int = 32,
        dim: int = 512,
        use_cosine_sim: bool = False,
    ):
        super().__init__()
        self.num_quantizers = num_quantizers
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.dim = dim

        # Input/output projections (always present to match PyTorch)
        # PyTorch: project_in maps dim -> codebook_dim
        # PyTorch: project_out maps codebook_dim -> dim
        self.project_in = nn.Linear(dim, codebook_dim, bias=True)
        self.project_out = nn.Linear(codebook_dim, dim, bias=True)

        # Create quantizer layers (named 'layers' to match PyTorch)
        self.layers = [
            VectorQuantizer(
                codebook_size=codebook_size,
                codebook_dim=codebook_dim,
                use_cosine_sim=use_cosine_sim,
            )
            for _ in range(num_quantizers)
        ]

    def encode(self, x: mx.array) -> mx.array:
        """Encode input to multi-level codes.

        Args:
            x: Input of shape (batch, seq_len, dim).

        Returns:
            Codes of shape (batch, seq_len, num_quantizers).
        """
        # Project to codebook dimension
        x = self.project_in(x)

        all_codes = []
        residual = x

        for quantizer in self.layers:
            codes = quantizer.encode(residual)
            all_codes.append(codes)

            # Compute residual for next level
            quantized = quantizer.decode(codes)
            residual = residual - quantized

        # Stack codes: (batch, seq_len, num_quantizers)
        return mx.stack(all_codes, axis=-1)

    def decode(self, codes: mx.array) -> mx.array:
        """Decode multi-level codes to embeddings.

        Args:
            codes: Codes of shape (batch, seq_len, num_quantizers).

        Returns:
            Embeddings of shape (batch, seq_len, dim).
        """
        # Sum up all quantized levels
        quantized = mx.zeros((*codes.shape[:-1], self.codebook_dim))

        for i, quantizer in enumerate(self.layers):
            level_codes = codes[..., i]
            quantized = quantized + quantizer.decode(level_codes)

        # Project back to output dimension
        quantized = self.project_out(quantized)

        return quantized

    def from_codes(self, codes: mx.array) -> mx.array:
        """Alias for decode() for compatibility.

        Args:
            codes: Codes of shape (batch, seq_len, num_quantizers).

        Returns:
            Embeddings of shape (batch, seq_len, dim).
        """
        return self.decode(codes)

    def __call__(
        self,
        x: mx.array,
        n_quantizers: Optional[int] = None,
    ) -> Tuple[mx.array, mx.array, mx.array]:
        """Forward pass with residual quantization.

        Args:
            x: Input of shape (batch, seq_len, dim).
            n_quantizers: Number of quantizers to use (default: all).

        Returns:
            Tuple of (quantized, codes, total_commitment_loss).
        """
        n_q = n_quantizers or self.num_quantizers

        # Project to codebook dimension
        x = self.project_in(x)

        all_codes = []
        all_quantized = []
        total_loss = mx.zeros((), dtype=x.dtype)
        residual = x

        for quantizer in self.layers[:n_q]:
            quantized, codes, loss = quantizer(residual)
            all_codes.append(codes)
            all_quantized.append(quantized)
            total_loss = total_loss + loss

            # Update residual
            residual = residual - mx.stop_gradient(quantized)

        # Sum all quantized levels
        final_quantized = sum(all_quantized)

        # Project back to output dimension
        final_quantized = self.project_out(final_quantized)

        # Stack codes
        codes = mx.stack(all_codes, axis=-1)

        return final_quantized, codes, total_loss / n_q


class ScalarQuantizer(nn.Module):
    """Scalar Quantizer using rounding.

    Quantizes values by rounding to the nearest 1/9th.
    This is a simple fixed-point quantization without learned codebooks.
    """

    def __init__(self, num_levels: int = 9):
        super().__init__()
        self.num_levels = num_levels

    def encode(self, x: mx.array) -> mx.array:
        """Quantize by rounding.

        Args:
            x: Input tensor.

        Returns:
            Quantized tensor (still continuous for reconstruction).
        """
        return mx.round(self.num_levels * x) / self.num_levels

    def __call__(self, x: mx.array) -> Tuple[mx.array, mx.array]:
        """Forward pass with straight-through gradient.

        Args:
            x: Input tensor.

        Returns:
            Tuple of (quantized, quantized_detached).
        """
        quantized = self.encode(x)
        # Straight-through: gradient flows through as if no quantization
        quantized_st = x + mx.stop_gradient(quantized - x)
        return quantized_st, quantized

# --- from mlx/heartcodec/scalar_codec.py ---

from typing import List, Tuple, Optional

import mlx.core as mx
import mlx.nn as nn

from backend.engine.families.heartmula.nn_mlx import CausalConv1d, Conv1d, WeightNormConv1d, WeightNormConvTranspose1d


class ResidualUnit(nn.Module):
    """Residual unit with dilated convolutions.

    Uses weight-normalized convolutions with PReLU activation and
    dilated receptive fields for efficient context modeling.

    Matches PyTorch HeartCodec structure with two separate PReLU activations.

    Args:
        channels: Number of input/output channels.
        kernel_size: Convolution kernel size.
        dilation: Dilation factor.
        causal: Whether to use causal convolutions.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 7,
        dilation: int = 1,
        causal: bool = True,
    ):
        super().__init__()

        # Dilated conv
        self.conv1 = WeightNormConv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            dilation=dilation,
            causal=causal,
        )

        # Pointwise conv
        self.conv2 = WeightNormConv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=1,
            causal=False,
        )

        # Two separate PReLU with single parameter (matches PyTorch)
        self.activation = PReLU(1)
        self.activation2 = PReLU(1)

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass with residual connection.

        Args:
            x: Input of shape (batch, length, channels).

        Returns:
            Output with same shape.
        """
        residual = x
        x = self.activation(self.conv1(x))
        x = self.activation2(self.conv2(x))
        return x + residual


class ResEncoderBlock(nn.Module):
    """Encoder block with residual units and downsampling.

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        kernel_size: Residual unit kernel size.
        stride: Downsampling stride.
        causal: Whether to use causal convolutions.
        num_residual: Number of residual units.
        dilations: Dilation factors for residual units.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 7,
        stride: int = 4,
        causal: bool = True,
        num_residual: int = 5,
        dilations: Optional[List[int]] = None,
    ):
        super().__init__()

        dilations = dilations or [1, 3, 5, 7, 9]

        # Residual units
        self.residual_units = [
            ResidualUnit(
                channels=in_channels,
                kernel_size=kernel_size,
                dilation=dilations[i % len(dilations)],
                causal=causal,
            )
            for i in range(num_residual)
        ]

        # Downsampling convolution
        self.downsample = WeightNormConv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=stride * 2,
            stride=stride,
            causal=causal,
        )

        # Single-parameter PReLU (matches PyTorch)
        self.activation = PReLU(1)

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass.

        Args:
            x: Input of shape (batch, length, in_channels).

        Returns:
            Downsampled output of shape (batch, length // stride, out_channels).
        """
        for unit in self.residual_units:
            x = unit(x)

        x = self.activation(self.downsample(x))
        return x


class ResDecoderBlock(nn.Module):
    """Decoder block with upsampling and residual units.

    Note: PyTorch version has activation=None in UpsampleLayer,
    so no activation is applied after upsampling.

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        kernel_size: Residual unit kernel size.
        stride: Upsampling stride.
        causal: Whether to use causal convolutions.
        num_residual: Number of residual units.
        dilations: Dilation factors for residual units.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 7,
        stride: int = 4,
        causal: bool = True,
        num_residual: int = 5,
        dilations: Optional[List[int]] = None,
    ):
        super().__init__()

        dilations = dilations or [1, 3, 5, 7, 9]

        # Upsampling convolution (transposed)
        # Note: PyTorch uses activation=None here
        # For causal mode: padding=0 and output is trimmed by stride samples
        self.upsample = WeightNormConvTranspose1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=stride * 2,
            stride=stride,
            padding=0 if causal else stride // 2,
            causal=causal,
        )

        # No activation after upsample (matches PyTorch)

        # Residual units
        self.residual_units = [
            ResidualUnit(
                channels=out_channels,
                kernel_size=kernel_size,
                dilation=dilations[i % len(dilations)],
                causal=causal,
            )
            for i in range(num_residual)
        ]

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass.

        Args:
            x: Input of shape (batch, length, in_channels).

        Returns:
            Upsampled output of shape (batch, length * stride, out_channels).
        """
        # No activation after upsample (matches PyTorch)
        x = self.upsample(x)

        for unit in self.residual_units:
            x = unit(x)

        return x


class ScalarModel(nn.Module):
    """Scalar Codec model for audio encoding/decoding.

    Encodes audio waveforms to quantized latent representations
    and decodes them back to waveforms.

    Args:
        num_bands: Number of frequency bands (default: 1 for raw audio).
        sample_rate: Audio sample rate in Hz.
        causal: Whether to use causal convolutions.
        downsample_factors: Downsampling factors per encoder stage.
        downsample_kernel_sizes: Kernel sizes for downsampling.
        upsample_factors: Upsampling factors per decoder stage.
        upsample_kernel_sizes: Kernel sizes for upsampling.
        latent_hidden_dim: Hidden dimension in latent space.
        default_kernel_size: Default convolution kernel size.
        init_channel: Initial channel count.
        res_kernel_size: Residual block kernel size.
    """

    def __init__(
        self,
        num_bands: int = 1,
        sample_rate: int = 48000,
        causal: bool = True,
        num_samples: int = 2,
        downsample_factors: Optional[List[int]] = None,
        downsample_kernel_sizes: Optional[List[int]] = None,
        upsample_factors: Optional[List[int]] = None,
        upsample_kernel_sizes: Optional[List[int]] = None,
        latent_hidden_dim: int = 128,
        default_kernel_size: int = 7,
        delay_kernel_size: int = 5,
        init_channel: int = 64,
        res_kernel_size: int = 7,
    ):
        super().__init__()
        self.num_samples = num_samples

        downsample_factors = downsample_factors or [4, 4, 8, 8]
        downsample_kernel_sizes = downsample_kernel_sizes or [8, 8, 16, 16]
        upsample_factors = upsample_factors or [8, 8, 4, 4]
        upsample_kernel_sizes = upsample_kernel_sizes or [16, 16, 8, 8]

        self.num_bands = num_bands
        self.sample_rate = sample_rate
        self.causal = causal

        # Calculate channel progression
        channels = [init_channel]
        for _ in downsample_factors:
            channels.append(channels[-1] * 2)

        # ========== Encoder ==========
        # Initial convolution
        self.encoder_in = WeightNormConv1d(
            in_channels=num_bands,
            out_channels=init_channel,
            kernel_size=default_kernel_size,
            causal=causal,
        )

        # Encoder blocks
        self.encoder_blocks = []
        for i, (stride, kernel_size) in enumerate(zip(downsample_factors, downsample_kernel_sizes)):
            self.encoder_blocks.append(
                ResEncoderBlock(
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                    kernel_size=res_kernel_size,
                    stride=stride,
                    causal=causal,
                )
            )

        # Final encoder conv to latent
        self.encoder_out = WeightNormConv1d(
            in_channels=channels[-1],
            out_channels=latent_hidden_dim,
            kernel_size=default_kernel_size,
            causal=causal,
        )

        # Scalar quantizer
        self.quantizer = ScalarQuantizer()

        # ========== Decoder ==========
        # Initial decoder conv from latent (uses delay_kernel_size per PyTorch)
        # Non-causal with symmetric padding to preserve length
        self.decoder_in = WeightNormConv1d(
            in_channels=latent_hidden_dim,
            out_channels=channels[-1],
            kernel_size=delay_kernel_size,
            padding=(delay_kernel_size - 1) // 2,  # Symmetric padding for non-causal
            causal=False,
        )

        # Decoder blocks (reverse order)
        self.decoder_blocks = []
        for i, (stride, kernel_size) in enumerate(zip(upsample_factors, upsample_kernel_sizes)):
            in_ch = channels[-(i + 1)]
            out_ch = channels[-(i + 2)]
            self.decoder_blocks.append(
                ResDecoderBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=res_kernel_size,
                    stride=stride,
                    causal=causal,
                )
            )

        # PostProcessor conv (decoder.6.conv in PyTorch)
        # This uses causal padding when causal=True
        if causal:
            self.post_conv = CausalConv1d(
                in_channels=init_channel,
                out_channels=init_channel,
                kernel_size=default_kernel_size,
            )
        else:
            self.post_conv = Conv1d(
                in_channels=init_channel,
                out_channels=init_channel,
                kernel_size=default_kernel_size,
                padding=(default_kernel_size - 1) // 2,  # Same padding for non-causal
            )

        # Final decoder conv
        self.decoder_out = WeightNormConv1d(
            in_channels=init_channel,
            out_channels=num_bands,
            kernel_size=default_kernel_size,
            causal=causal,
        )

        # Single-parameter PReLU (matches PyTorch)
        self.activation = PReLU(1)
        self.tanh = lambda x: mx.tanh(x)

    def encode(self, x: mx.array) -> mx.array:
        """Encode audio to quantized latent.

        Args:
            x: Audio waveform of shape (batch, samples, bands) or (batch, samples).

        Returns:
            Quantized latent of shape (batch, frames, latent_dim).
        """
        # Ensure 3D input
        if x.ndim == 2:
            x = x[:, :, None]

        # Encoder
        x = self.encoder_in(x)
        for block in self.encoder_blocks:
            x = block(x)
        x = self.encoder_out(x)

        # Apply tanh and quantize
        x = self.tanh(x)
        quantized, _ = self.quantizer(x)

        return quantized

    def decode(self, z: mx.array) -> mx.array:
        """Decode quantized latent to audio.

        Args:
            z: Quantized latent of shape (batch, frames, latent_dim).

        Returns:
            Audio waveform of shape (batch, samples, bands).
        """
        x = self.decoder_in(z)
        for block in self.decoder_blocks:
            x = block(x)

        # PostProcessor (decoder.6 in PyTorch):
        # 1. Repeat each sample num_samples times (sample-level upsampling)
        # 2. Apply conv
        # 3. Apply activation
        if self.num_samples > 1:
            # x shape: (batch, time, channels)
            batch, time, channels = x.shape
            # Repeat each time step num_samples times
            # (batch, time, channels) -> (batch, time, num_samples, channels) -> (batch, time*num_samples, channels)
            x = mx.repeat(x[:, :, None, :], self.num_samples, axis=2)
            x = x.reshape(batch, time * self.num_samples, channels)

        x = self.post_conv(x)
        x = self.activation(x)

        # Final output conv (decoder.7 in PyTorch)
        x = self.decoder_out(x)
        # Note: No tanh at the end - tanh is only used in encoding

        return x

    def __call__(
        self,
        x: mx.array,
    ) -> Tuple[mx.array, mx.array, mx.array]:
        """Forward pass: encode, quantize, decode.

        Args:
            x: Audio waveform of shape (batch, samples, bands).

        Returns:
            Tuple of (reconstruction, embedding, quantized_embedding).
        """
        # Ensure 3D
        if x.ndim == 2:
            x = x[:, :, None]

        # Encode
        h = self.encoder_in(x)
        for block in self.encoder_blocks:
            h = block(h)
        h = self.encoder_out(h)
        embedding = self.tanh(h)

        # Quantize
        quantized, _ = self.quantizer(embedding)

        # Decode
        out = self.decode(quantized)

        return out, embedding, quantized

    def inference(self, x: mx.array) -> Tuple[mx.array, mx.array, mx.array]:
        """Inference mode: encode, quantize, decode.

        Args:
            x: Audio waveform.

        Returns:
            Tuple of (embedding, quantized, reconstruction).
        """
        _, embedding, quantized = self(x)
        reconstruction = self.decode(quantized)
        return embedding, quantized, reconstruction

# --- from mlx/heartcodec/flow_matching.py ---

from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from backend.engine.common.embeddings import sinusoidal_timestep_proj
from backend.engine.common.text_encoders.qwen3_mlx import MlxTimestepEmbeddingMLP
from backend.engine.common.mlx_runtime_fallback import random_normal
from backend.engine.runtime.mlx import MLXContext
from backend.engine.families.heartmula.nn_mlx import RMSNorm, LlamaAttention, LlamaMLP
from backend.engine.families.heartmula.ode_mlx import euler_solve

_MLX_CTX = MLXContext()


class FFNBlock(nn.Module):
    """FFN projection block with Conv1d + Linear.

    PyTorch uses this pattern for proj_in, proj_out, connection_proj.

    Args:
        in_features: Input features.
        out_features: Output features.
        hidden_features: Hidden dimension (for ffn_2).
        kernel_size: Conv1d kernel size.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: Optional[int] = None,
        kernel_size: int = 3,
    ):
        super().__init__()
        hidden_features = hidden_features or out_features
        self.kernel_size = kernel_size

        # Conv1d with kernel_size (no dilation)
        # PyTorch weight: (out, in, k)
        # MLX weight: (out, k, in) - we'll handle in conversion
        padding = kernel_size // 2
        self.ffn_1 = nn.Conv1d(
            in_channels=in_features,
            out_channels=hidden_features,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.ffn_2 = nn.Linear(hidden_features, out_features)

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass.

        Args:
            x: Input of shape (batch, seq_len, in_features).

        Returns:
            Output of shape (batch, seq_len, out_features).
        """
        # Conv1d expects (batch, seq, channels)
        x = self.ffn_1(x)
        # Apply scaling factor (matches PyTorch's ProjectLayer: x * kernel_size**-0.5)
        x = x * (self.kernel_size ** -0.5)
        # NOTE: PyTorch's ProjectLayer has NO activation between conv and linear
        x = self.ffn_2(x)
        return x


class TimestepEmbedder(nn.Module):
    """Timestep embedding with sinusoidal encoding + MLP.

    Args:
        hidden_size: Output dimension.
        frequency_embedding_size: Sinusoidal embedding dimension.
    """

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 512):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = MlxTimestepEmbeddingMLP(frequency_embedding_size, hidden_size)

    def __call__(self, t: mx.array) -> mx.array:
        """Forward pass.

        Args:
            t: Timestep of shape (batch,).

        Returns:
            Embedding of shape (batch, hidden_size).
        """
        t_emb = sinusoidal_timestep_proj(
            _MLX_CTX, t, self.frequency_embedding_size, sin_first=False, scale=1000.0
        )
        return self.mlp(t_emb)


class AdaLNSingle(nn.Module):
    """Adaptive LayerNorm Single for flow matching.

    Projects timestep embedding to scale/shift for all blocks.

    Args:
        dim: Model dimension.
        num_outputs: Number of output values (6 = shift1, scale1, gate1, shift2, scale2, gate2).
    """

    def __init__(self, dim: int, num_outputs: int = 6):
        super().__init__()
        self.emb = nn.Module()
        self.emb.timestep_embedder = TimestepEmbedder(dim)
        self.linear = nn.Linear(dim, num_outputs * dim, bias=True)

    def __call__(self, t: mx.array) -> tuple[mx.array, mx.array]:
        """Forward pass.

        Args:
            t: Timestep of shape (batch,).

        Returns:
            Tuple of:
            - conditioning: (batch, num_outputs, dim) for transformer blocks
            - embedded_timestep: (batch, dim) for scale_shift_table modulation
        """
        t_emb = self.emb.timestep_embedder(t)
        # Apply silu before linear (matches PyTorch: linear(silu(embedded_timestep)))
        conditioning = self.linear(nn.silu(t_emb))
        # Reshape to (batch, num_outputs, dim)
        batch_size = conditioning.shape[0]
        dim = t_emb.shape[-1]
        num_outputs = conditioning.shape[-1] // dim
        conditioning = conditioning.reshape(batch_size, num_outputs, dim)
        return conditioning, t_emb


class FlowMatchingTransformerBlock(nn.Module):
    """Transformer block with scale_shift_table for flow matching.

    Uses per-block learnable scale/shift that combines with timestep embedding.

    Args:
        dim: Model dimension.
        n_heads: Number of attention heads.
        head_dim: Dimension per head.
        mlp_hidden_dim: MLP hidden dimension.
        norm_eps: Epsilon for RMSNorm.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        head_dim: int = 64,
        mlp_hidden_dim: Optional[int] = None,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.dim = dim
        mlp_hidden_dim = mlp_hidden_dim or int(dim * 8 / 3)  # Default SwiGLU ratio

        # Norms
        self.attn_norm = RMSNorm(dim, eps=norm_eps)
        self.mlp_norm = RMSNorm(dim, eps=norm_eps)

        # Attention
        self.attn = LlamaAttention(
            dim=dim,
            n_heads=n_heads,
            head_dim=head_dim,
            bias=False,
        )

        # MLP
        self.mlp = LlamaMLP(dim=dim, hidden_dim=mlp_hidden_dim)

        # Per-block scale/shift table: (6, dim)
        # [shift1, scale1, gate1, shift2, scale2, gate2]
        self.scale_shift_table = mx.zeros((6, dim))

    def __call__(
        self,
        x: mx.array,
        adaln_cond: mx.array,
        mask: Optional[mx.array] = None,
    ) -> mx.array:
        """Forward pass.

        Args:
            x: Input of shape (batch, seq_len, dim).
            adaln_cond: AdaLN conditioning of shape (batch, 6, dim).
            mask: Optional attention mask.

        Returns:
            Output of shape (batch, seq_len, dim).
        """
        # Combine per-block table with shared conditioning
        # scale_shift_table: (6, dim), adaln_cond: (batch, 6, dim)
        cond = self.scale_shift_table[None, :, :] + adaln_cond  # (batch, 6, dim)

        # Split into components
        shift1 = cond[:, 0:1, :]  # (batch, 1, dim)
        scale1 = cond[:, 1:2, :]
        gate1 = cond[:, 2:3, :]
        shift2 = cond[:, 3:4, :]
        scale2 = cond[:, 4:5, :]
        gate2 = cond[:, 5:6, :]

        # Attention with adaptive norm
        h = self.attn_norm(x)
        h = h * (1 + scale1) + shift1
        attn_out, _ = self.attn(h, mask=mask)
        x = x + gate1 * attn_out

        # MLP with adaptive norm
        h = self.mlp_norm(x)
        h = h * (1 + scale2) + shift2
        x = x + gate2 * self.mlp(h)

        return x


class LlamaTransformerForFlowMatching(nn.Module):
    """Two-stage transformer for flow matching velocity prediction.

    Stage 1: 24 layers at dim=1536
    Stage 2: 6 layers at dim=3072 (doubled)

    Args:
        dim: Stage 1 dimension (1536).
        dim_2: Stage 2 dimension (3072).
        n_heads: Number of attention heads for stage 1.
        n_heads_2: Number of attention heads for stage 2.
        head_dim: Dimension per head.
        num_layers: Layers in stage 1.
        num_layers_2: Layers in stage 2.
        in_channels: Input conditioning dimension.
        out_channels: Output velocity dimension.
        mlp_hidden_dim: MLP hidden for stage 1.
        mlp_hidden_dim_2: MLP hidden for stage 2.
    """

    def __init__(
        self,
        dim: int = 1536,
        dim_2: int = 3072,
        n_heads: int = 24,
        n_heads_2: int = 48,
        head_dim: int = 64,
        num_layers: int = 24,
        num_layers_2: int = 6,
        in_channels: int = 1024,
        out_channels: int = 256,
        mlp_hidden_dim: int = 4096,
        mlp_hidden_dim_2: int = 8192,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.dim = dim
        self.dim_2 = dim_2

        # Input projection: FFN block with Conv1d
        self.proj_in = FFNBlock(in_channels, dim, hidden_features=dim)

        # Stage 1 time embedding
        self.adaln_single = AdaLNSingle(dim, num_outputs=6)

        # Stage 1 transformer blocks
        self.transformer_blocks = [
            FlowMatchingTransformerBlock(
                dim=dim,
                n_heads=n_heads,
                head_dim=head_dim,
                mlp_hidden_dim=mlp_hidden_dim,
                norm_eps=norm_eps,
            )
            for _ in range(num_layers)
        ]

        # Output norm for stage 1 (LayerNorm without affine, matches PyTorch)
        self.norm_out = nn.LayerNorm(dim, eps=norm_eps, affine=False)

        # Scale/shift for final stage 1 output
        self.scale_shift_table = mx.zeros((2, dim))

        # Connection projection: stage 1 output + latent -> stage 2 input
        # PyTorch has in_features=2560 which is 1536 + 1024
        # Actually looking at the shapes, it seems to be 1536 + out_channels*4
        # Let's use 1536 + 1024 = 2560
        self.connection_proj = FFNBlock(dim + in_channels, dim_2, hidden_features=dim_2)

        # Stage 2 time embedding
        self.adaln_single_2 = AdaLNSingle(dim_2, num_outputs=6)

        # Stage 2 transformer blocks
        self.transformer_blocks_2 = [
            FlowMatchingTransformerBlock(
                dim=dim_2,
                n_heads=n_heads_2,
                head_dim=head_dim,
                mlp_hidden_dim=mlp_hidden_dim_2,
                norm_eps=norm_eps,
            )
            for _ in range(num_layers_2)
        ]

        # Output norm for stage 2 (LayerNorm without affine, matches PyTorch)
        self.norm_out_2 = nn.LayerNorm(dim_2, eps=norm_eps, affine=False)

        # Scale/shift for final stage 2 output
        self.scale_shift_table_2 = mx.zeros((2, dim_2))

        # Output projection
        self.proj_out = FFNBlock(dim_2, out_channels, hidden_features=out_channels)

    def __call__(
        self,
        t: mx.array,
        hidden_states: mx.array,
    ) -> mx.array:
        """Forward pass to predict velocity.

        Args:
            t: Timestep of shape (batch,) or (1,).
            hidden_states: Concatenated input of shape (batch, seq_len, in_channels).
                           This is [x, incontext_x, mu] concatenated along channels:
                           - x: (batch, seq, 256) noisy latent
                           - incontext_x: (batch, seq, 256) context latent
                           - mu: (batch, seq, 512) VQ embeddings
                           Total: 256 + 256 + 512 = 1024

        Returns:
            Predicted velocity of shape (batch, seq_len, out_channels).
        """
        batch_size = hidden_states.shape[0]

        # Expand t if needed
        if t.shape[0] == 1 and batch_size > 1:
            t = mx.broadcast_to(t, (batch_size,))

        # Project concatenated hidden_states to stage 1 dim
        s = self.proj_in(hidden_states)

        # Stage 1 processing
        adaln_cond, embedded_timestep = self.adaln_single(t)  # (batch, 6, dim), (batch, dim)
        for block in self.transformer_blocks:
            s = block(s, adaln_cond)

        # Apply final stage 1 norm and scale/shift (matches PyTorch)
        # PyTorch: shift, scale = (scale_shift_table[None] + embedded_timestep[:, None]).chunk(2, dim=1)
        s = self.norm_out(s)
        # Combine scale_shift_table with embedded_timestep: (1, 2, dim) + (batch, 1, dim) -> (batch, 2, dim)
        combined = self.scale_shift_table[None, :, :] + embedded_timestep[:, None, :]
        shift = combined[:, 0:1, :]  # (batch, 1, dim)
        scale = combined[:, 1:2, :]  # (batch, 1, dim)
        s = s * (1 + scale) + shift

        # Concatenate original hidden_states with stage 1 output for connection
        # (matches PyTorch: x = torch.cat([hidden_states, s], dim=-1))
        h = mx.concatenate([hidden_states, s], axis=-1)
        h = self.connection_proj(h)

        # Stage 2 processing
        adaln_cond_2, embedded_timestep_2 = self.adaln_single_2(t)  # (batch, 6, dim_2), (batch, dim_2)
        for block in self.transformer_blocks_2:
            h = block(h, adaln_cond_2)

        # Apply final stage 2 norm and scale/shift (matches PyTorch)
        h = self.norm_out_2(h)
        # Combine scale_shift_table_2 with embedded_timestep_2
        combined_2 = self.scale_shift_table_2[None, :, :] + embedded_timestep_2[:, None, :]
        shift2 = combined_2[:, 0:1, :]  # (batch, 1, dim_2)
        scale2 = combined_2[:, 1:2, :]  # (batch, 1, dim_2)
        h = h * (1 + scale2) + shift2

        # Output projection
        velocity = self.proj_out(h)

        return velocity


class FlowMatchingDecoder(nn.Module):
    """Flow Matching Decoder for HeartCodec.

    Combines:
    1. ResidualVQ for encoding audio codes to embeddings
    2. LlamaTransformer for velocity estimation
    3. ODE solver for generating latents from codes

    Args:
        dim: RVQ embedding dimension.
        codebook_size: Number of codes per codebook.
        codebook_dim: Dimension of code vectors.
        num_quantizers: Number of RVQ levels.
        attention_head_dim: Dimension per attention head.
        in_channels: Conditioning input channels.
        num_attention_heads: Number of attention heads.
        num_layers: Transformer layers in first stage.
        num_layers_2: Transformer layers in second stage.
        out_channels: Output latent dimension.
        use_cosine_sim: Use cosine similarity in RVQ.
    """

    def __init__(
        self,
        dim: int = 512,
        codebook_size: int = 8192,
        codebook_dim: int = 32,
        num_quantizers: int = 8,
        attention_head_dim: int = 64,
        in_channels: int = 1024,
        num_attention_heads: int = 24,
        num_layers: int = 24,
        num_layers_2: int = 6,
        out_channels: int = 256,
        use_cosine_sim: bool = False,
        decay: float = 0.9,
        commitment_weight: float = 1.0,
        threshold_ema_dead_code: int = 2,
    ):
        super().__init__()

        self.dim = dim
        self.out_channels = out_channels
        self.in_channels = in_channels

        # VQ embedding for code lookup
        self.vq_embed = ResidualVQ(
            num_quantizers=num_quantizers,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            dim=dim,
            use_cosine_sim=use_cosine_sim,
        )

        # Projection from VQ embeddings (used for conditioning)
        self.cond_feature_emb = nn.Linear(dim, dim, bias=True)

        # Zero embedding for classifier-free guidance
        self.zero_cond_embedding1 = mx.zeros((dim,))

        # Velocity estimator
        transformer_dim = num_attention_heads * attention_head_dim  # 24 * 64 = 1536
        transformer_dim_2 = transformer_dim * 2  # 3072

        self.estimator = LlamaTransformerForFlowMatching(
            dim=transformer_dim,
            dim_2=transformer_dim_2,
            n_heads=num_attention_heads,
            n_heads_2=num_attention_heads * 2,  # 48
            head_dim=attention_head_dim,
            num_layers=num_layers,
            num_layers_2=num_layers_2,
            in_channels=in_channels,
            out_channels=out_channels,
            mlp_hidden_dim=int(transformer_dim * 8 / 3),  # ~4096
            mlp_hidden_dim_2=int(transformer_dim_2 * 8 / 3),  # ~8192
        )
        self._solve_euler_compiled = None

    def solve_euler_compiled(
        self,
        x: mx.array,
        incontext_x: mx.array,
        incontext_length: int,
        t_span: mx.array,
        mu: mx.array,
        guidance_scale: float,
    ) -> mx.array:
        if self._solve_euler_compiled is None:
            self._solve_euler_compiled = mx.compile(self.solve_euler)
        return self._solve_euler_compiled(
            x, incontext_x, incontext_length, t_span, mu, guidance_scale
        )

    def solve_euler(
        self,
        x: mx.array,
        incontext_x: mx.array,
        incontext_length: int,
        t_span: mx.array,
        mu: mx.array,
        guidance_scale: float,
    ) -> mx.array:
        """Euler ODE solver matching PyTorch's implementation.

        Args:
            x: Initial noise (batch, seq, latent_dim).
            incontext_x: Context latent (batch, seq, latent_dim).
            incontext_length: Number of context frames.
            t_span: Time steps array.
            mu: Conditioning from VQ embeddings (batch, seq, 512).
            guidance_scale: CFG scale.

        Returns:
            Generated latent.
        """
        t = t_span[0]
        dt = t_span[1] - t_span[0]
        noise = x

        for step in range(1, len(t_span)):
            # Interpolate noise and context for incontext frames
            if incontext_length > 0:
                interp_factor = (1 - (1 - 1e-6) * t)
                x = mx.concatenate([
                    interp_factor * noise[:, :incontext_length, :] + t * incontext_x[:, :incontext_length, :],
                    x[:, incontext_length:, :]
                ], axis=1)

            if guidance_scale > 1.0:
                # Double batch for CFG
                x_doubled = mx.concatenate([x, x], axis=0)
                incontext_doubled = mx.concatenate([incontext_x, incontext_x], axis=0)
                # Unconditional has zeros for mu
                mu_uncond = mx.zeros_like(mu)
                mu_doubled = mx.concatenate([mu_uncond, mu], axis=0)

                # Concatenate [x, incontext_x, mu] along channel dim
                hidden_states = mx.concatenate([x_doubled, incontext_doubled, mu_doubled], axis=-1)
                t_tensor = mx.full((2,), t)

                # Run estimator
                dphi_dt = self.estimator(t_tensor, hidden_states)

                # Split and apply CFG
                dphi_dt_uncond, dphi_dt_cond = mx.split(dphi_dt, 2, axis=0)
                dphi_dt = dphi_dt_uncond + guidance_scale * (dphi_dt_cond - dphi_dt_uncond)
            else:
                # Concatenate [x, incontext_x, mu] along channel dim
                hidden_states = mx.concatenate([x, incontext_x, mu], axis=-1)
                t_tensor = mx.full((1,), t)
                dphi_dt = self.estimator(t_tensor, hidden_states)

            x = x + dt * dphi_dt
            t = t + dt

            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t_span[step]

        return x

    def inference_codes(
        self,
        codes: mx.array,
        true_latents: Optional[mx.array] = None,
        latent_length: Optional[int] = None,
        incontext_length: int = 0,
        num_steps: int = 10,
        guidance_scale: float = 1.25,
        scenario: str = "start_seg",
    ) -> mx.array:
        """Generate latents from codes (heartlib ``FlowMatching.inference_codes``).

        Args:
            codes: ``(batch, seq_len, num_quantizers)``.
            true_latents: Seed/context latent ``(batch, T, out_channels)``.
            latent_length: Frames to mark for generation (mask=2).
            incontext_length: Overlap context length (mask=1 when ``scenario='other_seg'``).
            num_steps: ODE steps.
            guidance_scale: CFG scale.
            scenario: ``start_seg`` or ``other_seg`` (chunked decode).

        Returns:
            Latent ``(batch, seq_len * 2, out_channels)``.
        """
        batch_size, _, _ = codes.shape

        embeddings = self.vq_embed.from_codes(codes)
        mu = self.cond_feature_emb(embeddings)
        mu = mx.repeat(mu, 2, axis=1)

        num_frames = int(mu.shape[1])
        if latent_length is None:
            latent_length = num_frames

        if true_latents is None:
            true_latents = random_normal(None, (batch_size, num_frames, self.out_channels))
        elif true_latents.shape[1] < num_frames:
            pad_t = num_frames - true_latents.shape[1]
            true_latents = mx.concatenate(
                [
                    true_latents,
                    random_normal(
                        None,
                        (batch_size, pad_t, self.out_channels),
                        dtype=true_latents.dtype,
                    ),
                ],
                axis=1,
            )
        elif true_latents.shape[1] > num_frames:
            true_latents = true_latents[:, :num_frames, :]

        latents = random_normal(None, (batch_size, num_frames, self.out_channels))

        latent_masks = mx.zeros((batch_size, num_frames), dtype=mx.int32)
        latent_masks[:, : int(latent_length)] = 2
        if scenario == "other_seg" and incontext_length > 0:
            latent_masks[:, : int(incontext_length)] = 1

        cond_mask = (latent_masks > 0.5).astype(mu.dtype)[:, :, None]
        zero_mask = (latent_masks < 0.5).astype(mu.dtype)[:, :, None]
        zce = self.zero_cond_embedding1.astype(mu.dtype)
        mu = cond_mask * mu + zero_mask * zce

        inctx_mask = (
            (latent_masks > 0.5) * (latent_masks < 1.5)
        ).astype(true_latents.dtype)[:, :, None]
        incontext_latents = true_latents * inctx_mask
        incontext_len = int(incontext_length)

        t_span = mx.linspace(0, 1, num_steps + 1)
        latents = self.solve_euler_compiled(
            x=latents,
            incontext_x=incontext_latents,
            incontext_length=incontext_len,
            t_span=t_span,
            mu=mu,
            guidance_scale=guidance_scale,
        )

        if incontext_len > 0:
            latents = mx.concatenate(
                [
                    incontext_latents[:, :incontext_len, :],
                    latents[:, incontext_len:, :],
                ],
                axis=1,
            )

        return latents

    def __call__(
        self,
        codes: mx.array,
        num_steps: int = 10,
        guidance_scale: float = 1.25,
    ) -> mx.array:
        """Forward pass for inference.

        Args:
            codes: Audio codes.
            num_steps: ODE integration steps.
            guidance_scale: CFG scale.

        Returns:
            Generated latent.
        """
        return self.inference_codes(codes, num_steps, guidance_scale)

# --- from mlx/heartcodec/modeling.py ---


import math
from typing import List, Optional, Union
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from backend.engine.common.mlx_runtime_fallback import (
    load_weights_dict,
    random_normal,
    run_eval,
)

# heartlib ``detokenize()`` uses this default for hop/chunk sizing (not full song length).
_HEARTLIB_CHUNK_DURATION_SEC = 29.76



def _run_eval(*values) -> None:
    run_eval(None, *values)


class HeartCodec(nn.Module):
    """HeartCodec: Neural Audio Codec with Flow Matching Decoder.

    HeartCodec is a 12.5Hz neural audio codec that combines:
    1. ScalarModel: Convolutional encoder/decoder for audio
    2. FlowMatchingDecoder: Transformer-based generative decoder

    The codec operates at 48kHz sample rate with a frame rate of
    12.5Hz (3840 samples per frame).

    Args:
        config: HeartCodecConfig with model hyperparameters.
    """

    def __init__(self, config: HeartCodecConfig):
        super().__init__()
        self.config = config

        # Scalar codec for audio encoding/decoding
        self.scalar_model = ScalarModel(
            num_bands=config.num_bands,
            sample_rate=config.sample_rate,
            causal=config.causal,
            num_samples=config.num_samples,
            downsample_factors=config.downsample_factors,
            downsample_kernel_sizes=config.downsample_kernel_sizes,
            upsample_factors=config.upsample_factors,
            upsample_kernel_sizes=config.upsample_kernel_sizes,
            latent_hidden_dim=config.latent_hidden_dim,
            default_kernel_size=config.default_kernel_size,
            delay_kernel_size=config.delay_kernel_size,
            init_channel=config.init_channel,
            res_kernel_size=config.res_kernel_size,
        )

        # Flow matching decoder for high-quality synthesis
        self.flow_matching = FlowMatchingDecoder(
            dim=config.dim,
            codebook_size=config.codebook_size,
            codebook_dim=config.codebook_dim,
            num_quantizers=config.num_quantizers,
            attention_head_dim=config.attention_head_dim,
            in_channels=config.in_channels,
            num_attention_heads=config.num_attention_heads,
            num_layers=config.num_layers,
            num_layers_2=config.num_layers_2,
            out_channels=config.out_channels,
            use_cosine_sim=config.use_cosine_sim,
            decay=config.decay,
            commitment_weight=config.commitment_weight,
            threshold_ema_dead_code=config.threshold_ema_dead_code,
        )

    def encode(self, audio: mx.array) -> mx.array:
        """Encode audio to quantized latent representation.

        Args:
            audio: Audio waveform of shape (batch, samples) or (batch, samples, 1).

        Returns:
            Quantized latent of shape (batch, frames, latent_dim).
        """
        return self.scalar_model.encode(audio)

    def decode(self, latent: mx.array) -> mx.array:
        """Decode quantized latent to audio waveform.

        Args:
            latent: Quantized latent of shape (batch, frames, latent_dim).

        Returns:
            Audio waveform of shape (batch, samples, 1).
        """
        return self.scalar_model.decode(latent)

    def _normalize_codes_layout(self, codes: mx.array) -> mx.array:
        """Accept ``(batch, time, K)`` or heartlib ``(batch, K, time)`` layout."""
        if codes.ndim != 3:
            raise RuntimeError(
                f"HeartCodec codes must be rank-3, got shape {tuple(codes.shape)}"
            )
        nq = int(self.config.num_quantizers)
        _, d1, d2 = codes.shape
        if d2 == nq:
            return codes
        if d1 == nq and d2 != nq:
            return codes.transpose(0, 2, 1)
        raise RuntimeError(
            f"HeartCodec codes layout unrecognized: shape={tuple(codes.shape)}, "
            f"expected (batch, time, {nq}) or (batch, {nq}, time)"
        )

    def _latent_to_waveform_chunk(self, latent: mx.array) -> mx.array:
        """Scalar decode one overlap chunk; returns ``(samples,)`` mono."""
        bsz, t, f = latent.shape
        latent = latent.reshape(bsz, t, 2, f // 2)
        latent = latent.transpose(0, 2, 1, 3)
        latent = latent.reshape(bsz * 2, t, f // 2)
        audio = self.scalar_model.decode(latent)
        samples = int(audio.shape[1])
        audio = audio.reshape(bsz, 2, samples, 1)
        audio = mx.mean(audio, axis=1)
        return audio[0, :, 0]

    def detokenize(
        self,
        codes: mx.array,
        duration: float = _HEARTLIB_CHUNK_DURATION_SEC,
        num_steps: int = 10,
        guidance_scale: float = 1.25,
    ) -> mx.array:
        """Convert discrete codes to waveform (heartlib chunked overlap-add).

        Args:
            codes: ``(batch, time, K)`` or ``(batch, K, time)``.
            duration: Chunk template seconds for hop sizing (default 29.76, per heartlib).
                Output length is derived from code frame count, not this value.
            num_steps: Flow-matching ODE steps.
            guidance_scale: Codec CFG scale.

        Returns:
            Mono waveform ``(batch, samples, 1)``.
        """
        codes = self._normalize_codes_layout(codes)
        batch_size = codes.shape[0]
        nq = int(codes.shape[2])
        content_frames = int(codes.shape[1])
        frame_rate = float(self.config.frame_rate)
        sample_rate = int(self.config.sample_rate)

        target_samples = int(content_frames / frame_rate * sample_rate)
        chunk_duration = float(duration)
        min_samples = int(chunk_duration * frame_rate)
        hop_samples = min_samples // 93 * 80
        ovlp_samples = min_samples - hop_samples
        ovlp_frames = ovlp_samples * 2
        latent_length = int(chunk_duration * 25)

        def _pad_codes_time(target_len: int) -> None:
            nonlocal codes
            cur = int(codes.shape[1])
            if cur >= target_len:
                if cur > target_len:
                    codes = codes[:, :target_len, :]
                return
            if cur == 0:
                pad = mx.zeros((batch_size, target_len - cur, nq), dtype=codes.dtype)
            else:
                last = codes[:, -1:, :]
                pad = mx.broadcast_to(last, (batch_size, target_len - cur, nq))
            codes = mx.concatenate([codes, pad], axis=1)

        first_latent = random_normal(
            None, (batch_size, latent_length, self.flow_matching.out_channels)
        )
        first_latent_length = 0

        if content_frames < min_samples:
            _pad_codes_time(min_samples)

        if (content_frames - ovlp_frames) % hop_samples > 0:
            len_codes = (
                math.ceil((content_frames - ovlp_samples) / float(hop_samples))
                * hop_samples
                + ovlp_samples
            )
            _pad_codes_time(len_codes)

        codes_len = int(codes.shape[1])

        latent_list: List[mx.array] = []
        for sinx in range(0, codes_len - hop_samples + 1, hop_samples):
            codes_chunk = codes[:, sinx : sinx + min_samples, :]
            if sinx == 0 or ovlp_frames == 0:
                latents = self.flow_matching.inference_codes(
                    codes_chunk,
                    true_latents=first_latent,
                    latent_length=latent_length,
                    incontext_length=first_latent_length,
                    num_steps=num_steps,
                    guidance_scale=guidance_scale,
                    scenario="other_seg",
                )
                latent_list.append(latents)
            else:
                prev = latent_list[-1]
                true_latent = prev[:, -ovlp_frames:, :]
                # heartlib: incontext_length is overlap width *before* padding to latent_length
                incontext_length = int(true_latent.shape[1])
                len_add = latent_length - incontext_length
                if len_add > 0:
                    true_latent = mx.concatenate(
                        [
                            true_latent,
                            random_normal(
                                None,
                                (
                                    batch_size,
                                    len_add,
                                    self.flow_matching.out_channels,
                                ),
                                dtype=true_latent.dtype,
                            ),
                        ],
                        axis=1,
                    )
                latents = self.flow_matching.inference_codes(
                    codes_chunk,
                    true_latents=true_latent,
                    latent_length=latent_length,
                    incontext_length=incontext_length,
                    num_steps=num_steps,
                    guidance_scale=guidance_scale,
                    scenario="other_seg",
                )
                latent_list.append(latents)

        latent_list[0] = latent_list[0][:, first_latent_length:, :]
        min_audio_samples = int(chunk_duration * sample_rate)
        hop_audio = min_audio_samples // 93 * 80
        ovlp_audio = min_audio_samples - hop_audio

        output: Optional[mx.array] = None
        for i, latent in enumerate(latent_list):
            cur = self._latent_to_waveform_chunk(latent)
            cur = cur[:min_audio_samples]
            if output is None:
                output = cur
            elif ovlp_audio == 0:
                output = mx.concatenate([output, cur], axis=0)
            else:
                t = mx.linspace(0.0, 1.0, ovlp_audio)
                ov_win = mx.concatenate([t, 1.0 - t], axis=0)
                tail = output[-ovlp_audio:] * ov_win[-ovlp_audio:]
                head = cur[:ovlp_audio] * ov_win[:ovlp_audio]
                output = mx.concatenate(
                    [output[:-ovlp_audio], tail + head, cur[ovlp_audio:]],
                    axis=0,
                )

        assert output is not None
        output = output[:target_samples]
        return output[:, None, None]

    def tokenize(self, audio: mx.array) -> mx.array:
        """Encode audio to discrete codes.

        Args:
            audio: Audio waveform of shape (batch, samples) or (batch, samples, 1).

        Returns:
            Audio codes of shape (batch, frames, num_quantizers).
        """
        # Get quantized latent
        latent = self.encode(audio)

        # Quantize to codes using flow matching's VQ
        codes = self.flow_matching.vq.encode(latent)

        return codes

    @classmethod
    def from_pretrained(
        cls,
        path: Union[str, Path],
        dtype: mx.Dtype = mx.bfloat16,
    ) -> "HeartCodec":
        """Load a pretrained HeartCodec model.

        Args:
            path: Path to the model directory.
            dtype: Data type for model weights.

        Returns:
            HeartCodec instance with loaded weights.
        """
        path = Path(path)

        # Load config
        config = HeartCodecConfig.from_pretrained(path)

        # Create model
        model = cls(config)

        # Load weights using MLX's native loader (handles bfloat16 properly)
        weights_path = path / "model.safetensors"
        if weights_path.exists():
            weights = load_weights_dict(None, str(weights_path))

            # Convert to target dtype if different
            weights = {k: v.astype(dtype) for k, v in weights.items()}

            # Load into model (strict=False to ignore PreProcessor/PostProcessor weights)
            model.load_weights(list(weights.items()), strict=False)
            _run_eval(model.parameters())

        return model

    def save_pretrained(self, path: Union[str, Path]) -> None:
        """Save the model to a directory.

        Args:
            path: Path to save the model.
        """
        from safetensors.numpy import save_file
        import numpy as np

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save config
        self.config.save_pretrained(path)

        # Save weights
        weights = dict(self.parameters())
        # Convert to numpy for safetensors
        np_weights = {k: np.array(v) for k, v in weights.items()}
        save_file(np_weights, str(path / "model.safetensors"))

    def __call__(
        self,
        audio: Optional[mx.array] = None,
        codes: Optional[mx.array] = None,
        num_steps: int = 10,
        guidance_scale: float = 1.25,
    ) -> mx.array:
        """Forward pass for encoding or decoding.

        Args:
            audio: Input audio for encoding (optional).
            codes: Input codes for decoding (optional).
            num_steps: ODE integration steps for decoding.
            guidance_scale: CFG scale for decoding.

        Returns:
            Encoded codes (if audio provided) or decoded audio (if codes provided).
        """
        if audio is not None:
            return self.tokenize(audio)
        elif codes is not None:
            return self.detokenize(codes, num_steps=num_steps, guidance_scale=guidance_scale)
        else:
            raise ValueError("Either audio or codes must be provided")


CHUNK_CODEC_DURATION_SECONDS = 29.76
SINGLE_PASS_CODEC_DURATION_SECONDS = 120.0


def chunk_code_frames(frame_rate: float) -> int:
    """Code frames per codec detokenize chunk (heartlib default ~29.76s)."""
    return int(CHUNK_CODEC_DURATION_SECONDS * float(frame_rate))


def single_pass_frame_limit(frame_rate: float) -> int:
    """Max code frames for single-pass detokenize; longer sequences use chunked decode."""
    return int(SINGLE_PASS_CODEC_DURATION_SECONDS * float(frame_rate))
