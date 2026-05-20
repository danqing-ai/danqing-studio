"""
ACE-Step audio VAE — pure MLX implementation for Apple Silicon.

Mirrors ``acestep/models/mlx/vae_model.py`` (Oobleck Autoencoder).
Architecture: Snake1d → OobleckResidualUnit → EncoderBlock/DecoderBlock
→ OobleckEncoder/OobleckDecoder → MLXAutoEncoderOobleck.

Data flows in NLC (batch, length, channels) throughout.
"""
from __future__ import annotations

import math
import logging
from pathlib import Path
from typing import Any, List, Optional

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Snake1d activation
# ---------------------------------------------------------------------------

class _Snake1d(nn.Module):
    """Snake activation: x + (1/beta) * sin(alpha * x)^2."""

    def __init__(self, hidden_dim: int, logscale: bool = True):
        super().__init__()
        self.alpha = mx.zeros(hidden_dim)
        self.beta = mx.zeros(hidden_dim)
        self.logscale = logscale

    def __call__(self, x: mx.array) -> mx.array:
        alpha = mx.exp(self.alpha) if self.logscale else self.alpha
        beta = mx.exp(self.beta) if self.logscale else self.beta
        return x + mx.reciprocal(beta + 1e-9) * mx.power(mx.sin(alpha * x), 2)


# ---------------------------------------------------------------------------
# Residual unit
# ---------------------------------------------------------------------------

class _ResidualUnit(nn.Module):
    """Two weight-normalised Conv1d layers (k=7 dilated + k=1) with Snake1d and skip."""

    def __init__(self, dimension: int = 16, dilation: int = 1):
        super().__init__()
        pad = ((7 - 1) * dilation) // 2
        self.snake1 = _Snake1d(dimension)
        self.conv1 = nn.Conv1d(dimension, dimension, kernel_size=7, dilation=dilation, padding=pad)
        self.snake2 = _Snake1d(dimension)
        self.conv2 = nn.Conv1d(dimension, dimension, kernel_size=1)

    def __call__(self, hidden_state: mx.array) -> mx.array:
        output = self.conv1(self.snake1(hidden_state))
        output = self.conv2(self.snake2(output))
        padding = (hidden_state.shape[1] - output.shape[1]) // 2
        if padding > 0:
            hidden_state = hidden_state[:, padding:-padding, :]
        return hidden_state + output


# ---------------------------------------------------------------------------
# Encoder / Decoder blocks
# ---------------------------------------------------------------------------

class _EncoderBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, stride: int = 1):
        super().__init__()
        self.res_unit1 = _ResidualUnit(input_dim, dilation=1)
        self.res_unit2 = _ResidualUnit(input_dim, dilation=3)
        self.res_unit3 = _ResidualUnit(input_dim, dilation=9)
        self.snake1 = _Snake1d(input_dim)
        self.conv1 = nn.Conv1d(
            input_dim, output_dim,
            kernel_size=2 * stride,
            stride=stride,
            padding=math.ceil(stride / 2),
        )

    def __call__(self, hidden_state: mx.array) -> mx.array:
        hidden_state = self.res_unit1(hidden_state)
        hidden_state = self.res_unit2(hidden_state)
        hidden_state = self.snake1(self.res_unit3(hidden_state))
        return self.conv1(hidden_state)


class _DecoderBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, stride: int = 1):
        super().__init__()
        self.snake1 = _Snake1d(input_dim)
        self.conv_t1 = nn.ConvTranspose1d(
            input_dim, output_dim,
            kernel_size=2 * stride,
            stride=stride,
            padding=math.ceil(stride / 2),
        )
        self.res_unit1 = _ResidualUnit(output_dim, dilation=1)
        self.res_unit2 = _ResidualUnit(output_dim, dilation=3)
        self.res_unit3 = _ResidualUnit(output_dim, dilation=9)

    def __call__(self, hidden_state: mx.array) -> mx.array:
        hidden_state = self.snake1(hidden_state)
        hidden_state = self.conv_t1(hidden_state)
        hidden_state = self.res_unit1(hidden_state)
        hidden_state = self.res_unit2(hidden_state)
        return self.res_unit3(hidden_state)


# ---------------------------------------------------------------------------
# Encoder / Decoder
# ---------------------------------------------------------------------------

class _OobleckEncoder(nn.Module):
    def __init__(
        self,
        encoder_hidden_size: int,
        audio_channels: int,
        downsampling_ratios: List[int],
        channel_multiples: List[int],
    ):
        super().__init__()
        strides = downsampling_ratios
        cm = [1] + list(channel_multiples)

        self.conv1 = nn.Conv1d(audio_channels, encoder_hidden_size, kernel_size=7, padding=3)

        self.block = []
        for i, stride in enumerate(strides):
            self.block.append(
                _EncoderBlock(
                    input_dim=encoder_hidden_size * cm[i],
                    output_dim=encoder_hidden_size * cm[i + 1],
                    stride=stride,
                )
            )

        d_model = encoder_hidden_size * cm[-1]
        self.snake1 = _Snake1d(d_model)
        self.conv2 = nn.Conv1d(d_model, encoder_hidden_size, kernel_size=3, padding=1)

    def __call__(self, hidden_state: mx.array) -> mx.array:
        hidden_state = self.conv1(hidden_state)
        for module in self.block:
            hidden_state = module(hidden_state)
        hidden_state = self.snake1(hidden_state)
        return self.conv2(hidden_state)


class _OobleckDecoder(nn.Module):
    def __init__(
        self,
        channels: int,
        input_channels: int,
        audio_channels: int,
        upsampling_ratios: List[int],
        channel_multiples: List[int],
    ):
        super().__init__()
        strides = upsampling_ratios
        cm = [1] + list(channel_multiples)

        self.conv1 = nn.Conv1d(input_channels, channels * cm[-1], kernel_size=7, padding=3)

        self.block = []
        for i, stride in enumerate(strides):
            self.block.append(
                _DecoderBlock(
                    input_dim=channels * cm[len(strides) - i],
                    output_dim=channels * cm[len(strides) - i - 1],
                    stride=stride,
                )
            )

        self.snake1 = _Snake1d(channels)
        self.conv2 = nn.Conv1d(channels, audio_channels, kernel_size=7, padding=3, bias=False)

    def __call__(self, hidden_state: mx.array) -> mx.array:
        hidden_state = self.conv1(hidden_state)
        for layer in self.block:
            hidden_state = layer(hidden_state)
        hidden_state = self.snake1(hidden_state)
        return self.conv2(hidden_state)


# ---------------------------------------------------------------------------
# Full VAE
# ---------------------------------------------------------------------------

class AceStepVAEMLX(nn.Module):
    """Pure-MLX Oobleck autoencoder for ACE-Step audio.

    Default config matches the Stable Audio / ACE-Step VAE:
        encoder_hidden_size  = 128
        downsampling_ratios  = [2, 4, 4, 8, 8]   (hop_length = 2048)
        channel_multiples    = [1, 2, 4, 8, 16]
        decoder_channels     = 128
        decoder_input_channels = 64
        audio_channels       = 2  (stereo)
    """

    def __init__(
        self,
        encoder_hidden_size: int = 128,
        downsampling_ratios: Optional[List[int]] = None,
        channel_multiples: Optional[List[int]] = None,
        decoder_channels: int = 128,
        decoder_input_channels: int = 64,
        audio_channels: int = 2,
    ):
        super().__init__()
        if downsampling_ratios is None:
            downsampling_ratios = [2, 4, 4, 8, 8]
        if channel_multiples is None:
            channel_multiples = [1, 2, 4, 8, 16]

        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_input_channels = decoder_input_channels

        self.encoder = _OobleckEncoder(
            encoder_hidden_size=encoder_hidden_size,
            audio_channels=audio_channels,
            downsampling_ratios=downsampling_ratios,
            channel_multiples=channel_multiples,
        )
        self.decoder = _OobleckDecoder(
            channels=decoder_channels,
            input_channels=decoder_input_channels,
            audio_channels=audio_channels,
            upsampling_ratios=downsampling_ratios[::-1],
            channel_multiples=channel_multiples,
        )

    def encode_and_sample(self, audio_nlc: mx.array) -> mx.array:
        h = self.encoder(audio_nlc)
        mean, scale = mx.split(h, 2, axis=-1)
        std = mx.where(scale > 20.0, scale, mx.log(1.0 + mx.exp(scale))) + 1e-4
        noise = mx.random.normal(mean.shape)
        return mean + std * noise

    def encode_mean(self, audio_nlc: mx.array) -> mx.array:
        h = self.encoder(audio_nlc)
        mean, _scale = mx.split(h, 2, axis=-1)
        return mean

    def decode(self, latents_nlc: mx.array) -> mx.array:
        return self.decoder(latents_nlc)


# ---------------------------------------------------------------------------
# Weight conversion: PyTorch AutoencoderOobleck → MLX AceStepVAEMLX
# ---------------------------------------------------------------------------

def _fuse_weight_norm(weight_g, weight_v, eps: float = 1e-9):
    """Merge PyTorch weight_norm's g * v / ||v|| into a single weight tensor."""
    import numpy as np
    g = weight_g.detach().cpu().float().numpy() if hasattr(weight_g, "detach") else weight_g
    v = weight_v.detach().cpu().float().numpy() if hasattr(weight_v, "detach") else weight_v
    v_flat = v.reshape(v.shape[0], -1)
    norm = np.linalg.norm(v_flat, axis=1).reshape(g.shape)
    return g * v / (norm + eps)


def _state_dict_value_to_numpy(val: Any) -> Any:
    import numpy as np

    if hasattr(val, "detach"):
        return val.detach().cpu().float().numpy()
    return np.asarray(val, dtype=np.float32)


def convert_vae_weights_from_state_dict(state_dict: dict) -> list:
    """Convert Oobleck checkpoint ``state_dict`` (numpy or torch tensors) to MLX weights."""
    import mlx.core as mx
    import numpy as np

    weights: list[tuple[str, mx.array]] = []
    all_keys = sorted(state_dict.keys())
    processed: set[str] = set()

    for key in all_keys:
        if key in processed:
            continue

        if key.endswith(".weight_g"):
            base = key[: -len(".weight_g")]
            v_key = base + ".weight_v"
            if v_key not in state_dict:
                processed.add(key)
                continue

            w = _fuse_weight_norm(
                _state_dict_value_to_numpy(state_dict[key]),
                _state_dict_value_to_numpy(state_dict[v_key]),
            )

            if "conv_t1" in base:
                w = w.transpose(1, 2, 0)
            else:
                w = w.swapaxes(1, 2)

            weights.append((base + ".weight", mx.array(w)))
            processed.add(key)
            processed.add(v_key)
            continue

        if key.endswith(".weight_v"):
            continue

        if key.endswith(".alpha") or key.endswith(".beta"):
            val = np.asarray(_state_dict_value_to_numpy(state_dict[key])).squeeze()
            weights.append((key, mx.array(val)))
            processed.add(key)
            continue

        val = _state_dict_value_to_numpy(state_dict[key])
        weights.append((key, mx.array(val)))
        processed.add(key)

    return weights


def convert_vae_weights_from_pytorch(pytorch_vae) -> list:
    """Extract PyTorch ``AutoencoderOobleck`` weights and convert to MLX format."""
    return convert_vae_weights_from_state_dict(pytorch_vae.state_dict())


def load_vae_weights_from_bundle(vae_dir: str, mlx_vae: AceStepVAEMLX) -> None:
    """Load VAE from bundle ``diffusion_pytorch_model.safetensors`` (no diffusers on MLX path)."""
    import mlx.core as mx
    from safetensors import safe_open

    st_path = Path(vae_dir) / "diffusion_pytorch_model.safetensors"
    if not st_path.is_file():
        raise RuntimeError(
            f"ACE-Step VAE weights not found: {st_path}. "
            "Expected diffusers Oobleck safetensors under bundle vae/."
        )
    state_dict: dict[str, Any] = {}
    # Checkpoint may be bfloat16; read via PyTorch bridge then cast to float32 numpy.
    with safe_open(str(st_path), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            state_dict[key] = handle.get_tensor(key).detach().cpu().float().numpy()
    weights = convert_vae_weights_from_state_dict(state_dict)
    mlx_vae.load_weights(weights)
    mx.eval(mlx_vae.parameters())


def load_vae_weights_from_pytorch(pytorch_vae, mlx_vae: AceStepVAEMLX) -> None:
    """Load weights from a PyTorch VAE into an MLX VAE (CUDA-only helper)."""
    weights = convert_vae_weights_from_pytorch(pytorch_vae)
    mlx_vae.load_weights(weights)
    mx.eval(mlx_vae.parameters())
