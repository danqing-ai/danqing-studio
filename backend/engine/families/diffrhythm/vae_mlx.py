"""
DiffRhythm 2 BigVGAN decoder — MLX implementation.

Decodes 5 Hz Music VAE latents (mel_dim=64) to 48 kHz mono waveform via BigVGAN v2.
Upstream: ASLP-lab/DiffRhythm2 ``bigvgan/model.py`` (Generator + chunked ``decode_audio``).
"""
from __future__ import annotations

import json
import logging
import math
import re
from pathlib import Path
from typing import Any, Callable

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from backend.engine.runtime.mlx_runtime import run_eval

logger = logging.getLogger(__name__)

SAMPLE_RATE = 48_000
SAMPLES_PER_LATENT = 9600  # product([10, 10, 8, 3, 2, 2])

_DEFAULT_DECODER_HPARAMS = {
    "resblock": "1",
    "upsample_rates": [10, 10, 8, 3, 2, 2],
    "upsample_kernel_sizes": [20, 20, 16, 7, 4, 4],
    "upsample_initial_channel": 1536,
    "in_channels": 64,
    "sampling_rate": SAMPLE_RATE,
    "activation": "snakebeta",
    "use_tanh_at_final": False,
    "use_bias_at_final": False,
    "resblock_kernel_sizes": [3, 7, 11],
    "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
}


def _eval(*vals: Any) -> None:
    run_eval(None, *vals)


# ---------------------------------------------------------------------------
# SnakeBeta + alias-free Activation1d (adapted from ltx/vae_mlx.py)
# ---------------------------------------------------------------------------


class SnakeBeta(nn.Module):
    """SnakeBeta: x + (1/b) * sin^2(a * x); alpha/beta stored in log-scale."""

    def __init__(self, channels: int):
        super().__init__()
        self.alpha = mx.zeros((channels,))
        self.beta = mx.zeros((channels,))

    def __call__(self, x: mx.array) -> mx.array:
        alpha = mx.exp(self.alpha).reshape(1, 1, -1)
        beta = mx.exp(self.beta).reshape(1, 1, -1)
        return x + (1.0 / (beta + 1e-9)) * mx.power(mx.sin(alpha * x), 2)


class LowPassKernel(nn.Module):
    """Low-pass filter kernel — MLX Conv1d layout (O, K, I)."""

    def __init__(self, kernel_size: int = 12):
        super().__init__()
        self.filter = mx.ones((1, kernel_size, 1))


class DownSample1d(nn.Module):
    """Alias-free 2× downsample (upstream ``LowPassFilter1d``)."""

    def __init__(self, ratio: int = 2, kernel_size: int = 12):
        super().__init__()
        self.ratio = int(ratio)
        self.kernel_size = int(kernel_size)
        self.lowpass = LowPassKernel(kernel_size)
        even = 1 if kernel_size % 2 == 0 else 0
        self.pad_left = kernel_size // 2 - even
        self.pad_right = kernel_size // 2

    def __call__(self, x: mx.array) -> mx.array:
        B, T, C = x.shape
        x = x.transpose(0, 2, 1).reshape(B * C, T, 1)
        left_edge = mx.repeat(x[:, :1, :], self.pad_left, axis=1)
        right_edge = mx.repeat(x[:, -1:, :], self.pad_right, axis=1)
        x = mx.concatenate([left_edge, x, right_edge], axis=1)
        x = mx.conv1d(x, self.lowpass.filter, stride=self.ratio)
        T_out = x.shape[1]
        return x.reshape(B, C, T_out).transpose(0, 2, 1)


class UpSample1d(nn.Module):
    """Alias-free 2× upsample (upstream ``alias_free_activation.torch.resample.UpSample1d``)."""

    def __init__(self, ratio: int = 2, kernel_size: int = 12):
        super().__init__()
        self.ratio = int(ratio)
        self.kernel_size = int(kernel_size)
        self.stride = self.ratio
        self.pad = self.kernel_size // self.ratio - 1
        self.pad_left = self.pad * self.stride + (self.kernel_size - self.stride) // 2
        self.pad_right = self.pad * self.stride + (self.kernel_size - self.stride + 1) // 2
        self.filter = mx.ones((1, kernel_size, 1))

    def __call__(self, x: mx.array) -> mx.array:
        """x: (B, T, C) — matches upstream replicate-pad + grouped conv_transpose1d."""
        _B, _T, C = x.shape
        pad = self.pad
        ratio = self.ratio

        left_edge = mx.repeat(x[:, :1, :], pad, axis=1)
        right_edge = mx.repeat(x[:, -1:, :], pad, axis=1)
        x = mx.concatenate([left_edge, x, right_edge], axis=1)

        weight = mx.repeat(self.filter, C, axis=0)  # (C, K, 1)
        y = ratio * mx.conv_transpose1d(x, weight, stride=ratio, groups=C)

        if self.pad_right > 0:
            y = y[:, self.pad_left : -self.pad_right, :]
        else:
            y = y[:, self.pad_left :, :]
        return y


class Activation1d(nn.Module):
    def __init__(self, channels: int, ratio: int = 2, kernel_size: int = 12):
        super().__init__()
        self.act = SnakeBeta(channels)
        self.upsample = UpSample1d(ratio=ratio, kernel_size=kernel_size)
        self.downsample = DownSample1d(ratio=ratio, kernel_size=kernel_size)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.upsample(x)
        x = self.act(x)
        x = self.downsample(x)
        return x


class AMPBlock1(nn.Module):
    """Anti-aliased multi-periodicity block (resblock type 1)."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilations: tuple[int, ...] = (1, 3, 5),
    ):
        super().__init__()
        self.convs1 = []
        self.convs2 = []
        self.acts1 = []
        self.acts2 = []
        for d in dilations:
            padding = (kernel_size * d - d) // 2
            self.acts1.append(Activation1d(channels))
            self.convs1.append(
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    dilation=d,
                )
            )
            self.acts2.append(Activation1d(channels))
            self.convs2.append(
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                )
            )

    def __call__(self, x: mx.array) -> mx.array:
        for act1, conv1, act2, conv2 in zip(self.acts1, self.convs1, self.acts2, self.convs2):
            residual = x
            x = act1(x)
            x = conv1(x)
            x = act2(x)
            x = conv2(x)
            x = x + residual
        return x


class DiffRhythm2BigVGANMLX(nn.Module):
    """BigVGAN v2 vocoder: 64-dim latent frames -> mono waveform."""

    def __init__(
        self,
        in_channels: int = 64,
        upsample_initial_channel: int = 1536,
        upsample_rates: tuple[int, ...] = (10, 10, 8, 3, 2, 2),
        upsample_kernel_sizes: tuple[int, ...] = (20, 20, 16, 7, 4, 4),
        resblock_kernel_sizes: tuple[int, ...] = (3, 7, 11),
        resblock_dilation_sizes: tuple[tuple[int, ...], ...] = (
            (1, 3, 5),
            (1, 3, 5),
            (1, 3, 5),
        ),
        out_channels: int = 1,
        apply_final_activation: bool = False,
        use_bias_at_final: bool = False,
    ):
        super().__init__()
        self._apply_final_activation = apply_final_activation
        self._use_tanh_at_final = apply_final_activation

        self.conv_pre = nn.Conv1d(in_channels, upsample_initial_channel, kernel_size=7, padding=3)

        self.ups = []
        self.resblocks = []
        channels = upsample_initial_channel

        for rate, kernel in zip(upsample_rates, upsample_kernel_sizes):
            out_ch = channels // 2
            padding = (kernel - rate) // 2
            self.ups.append(
                nn.ConvTranspose1d(
                    channels,
                    out_ch,
                    kernel_size=kernel,
                    stride=rate,
                    padding=padding,
                )
            )
            for rk, rd in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(AMPBlock1(out_ch, rk, rd))
            channels = out_ch

        self.act_post = Activation1d(channels)
        self.conv_post = nn.Conv1d(
            channels,
            out_channels,
            kernel_size=7,
            padding=3,
            bias=use_bias_at_final,
        )

        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)

    def __call__(self, mel: mx.array) -> mx.array:
        """Forward: mel (B, T, in_channels) -> waveform (B, T_audio, out_channels)."""
        x = self.conv_pre(mel)

        for i in range(self.num_upsamples):
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                idx = i * self.num_kernels + j
                if xs is None:
                    xs = self.resblocks[idx](x)
                else:
                    xs = xs + self.resblocks[idx](x)
            x = xs / self.num_kernels

        x = self.act_post(x)
        x = self.conv_post(x)
        if self._use_tanh_at_final:
            x = mx.tanh(x)
        else:
            x = mx.clip(x, -1.0, 1.0)
        return x


# ---------------------------------------------------------------------------
# Weight loading (torch at load time only)
# ---------------------------------------------------------------------------


def _remap_pytorch_generator_key(key: str) -> str:
    """Map PyTorch BigVGAN state-dict keys to this MLX module layout."""
    m = re.match(r"ups\.(\d+)\.0\.(weight|bias)$", key)
    if m:
        return f"ups.{m.group(1)}.{m.group(2)}"

    m = re.match(r"resblocks\.(\d+)\.activations\.(\d+)\.(.+)$", key)
    if m:
        rb_idx, act_idx, rest = m.group(1), int(m.group(2)), m.group(3)
        slot = "acts1" if act_idx % 2 == 0 else "acts2"
        sub = act_idx // 2
        return f"resblocks.{rb_idx}.{slot}.{sub}.{rest}"

    if key.startswith("activation_post."):
        return "act_post." + key[len("activation_post.") :]

    return key


def _is_conv_transpose_key(key: str) -> bool:
    return bool(re.match(r"ups\.\d+\.weight$", key))


def _is_conv1d_weight_key(key: str) -> bool:
    if not key.endswith(".weight"):
        return False
    if _is_conv_transpose_key(key):
        return False
    return any(
        token in key
        for token in (
            "conv_pre",
            "conv_post",
            "convs1",
            "convs2",
        )
    )


def _convert_torch_tensor(key: str, tensor: Any, array_fn: Callable[[Any], mx.array]) -> mx.array:
    arr = np.asarray(tensor)
    if _is_conv1d_weight_key(key):
        arr = arr.transpose(0, 2, 1)
    elif _is_conv_transpose_key(key):
        arr = arr.transpose(1, 2, 0)
    elif key.endswith(".filter"):
        if arr.ndim == 3:
            arr = arr.transpose(0, 2, 1)
        elif arr.ndim == 2:
            arr = arr[:, :, None]
    return array_fn(arr)


def _fuse_weight_norm(weight_g: Any, weight_v: Any, *, eps: float = 1e-9) -> np.ndarray:
    """Merge PyTorch weight_norm ``g * v / ||v||`` into a dense weight array."""
    g = np.asarray(weight_g, dtype=np.float32)
    v = np.asarray(weight_v, dtype=np.float32)
    v_flat = v.reshape(v.shape[0], -1)
    norm = np.linalg.norm(v_flat, axis=1).reshape(g.shape)
    return g * v / (norm + eps)


def load_decoder_weights(
    decoder: DiffRhythm2BigVGANMLX,
    ckpt_path: str,
    array_fn: Callable[[Any], mx.array],
) -> None:
    """Load ``decoder.bin`` (``torch.save`` dict with key ``generator``) into *decoder*."""
    from backend.engine.common.bundle.pytorch_bin_numpy import load_pytorch_bin

    path = Path(ckpt_path)
    if not path.is_file():
        raise RuntimeError(f"DiffRhythm 2 decoder checkpoint missing: {path}")

    checkpoint = load_pytorch_bin(path)
    if not isinstance(checkpoint, dict) or "generator" not in checkpoint:
        raise RuntimeError(
            f"DiffRhythm 2 decoder checkpoint at {path} must be a dict with key 'generator'"
        )

    raw = checkpoint["generator"]
    if not isinstance(raw, dict):
        raise RuntimeError(f"DiffRhythm 2 decoder 'generator' entry must be a state dict, got {type(raw)!r}")

    mlx_weights: dict[str, mx.array] = {}
    processed: set[str] = set()
    for pt_key in sorted(raw.keys()):
        if pt_key in processed:
            continue

        if pt_key.endswith(".weight_g"):
            base = pt_key[: -len(".weight_g")]
            v_key = base + ".weight_v"
            if v_key not in raw:
                processed.add(pt_key)
                continue
            fused_key = _remap_pytorch_generator_key(base + ".weight")
            fused = _fuse_weight_norm(raw[pt_key], raw[v_key])
            mlx_weights[fused_key] = _convert_torch_tensor(fused_key, fused, array_fn)
            processed.add(pt_key)
            processed.add(v_key)
            continue

        if pt_key.endswith(".weight_v"):
            continue

        mlx_key = _remap_pytorch_generator_key(pt_key)
        mlx_weights[mlx_key] = _convert_torch_tensor(mlx_key, raw[pt_key], array_fn)
        processed.add(pt_key)

    decoder.load_weights(list(mlx_weights.items()), strict=False)


def load_decoder_hparams(config_path: Path) -> dict[str, Any]:
    """Load ``decoder.json`` merged with BigVGAN defaults."""
    hparams = dict(_DEFAULT_DECODER_HPARAMS)
    if config_path.is_file():
        with open(config_path, encoding="utf-8") as f:
            hparams.update(json.load(f))
    return hparams


# ---------------------------------------------------------------------------
# Chunked decode wrapper + stereo helper
# ---------------------------------------------------------------------------


def _latents_to_btc(latents_mx: mx.array, mel_dim: int = 64) -> mx.array:
    """Normalize latents to (B, T, mel_dim) for the vocoder."""
    if latents_mx.ndim != 3:
        raise RuntimeError(f"DiffRhythm 2 decode expects 3D latents, got shape {latents_mx.shape}")
    b, d1, d2 = latents_mx.shape
    if d1 == mel_dim:
        return latents_mx.transpose(0, 2, 1)
    if d2 == mel_dim:
        return latents_mx
    raise RuntimeError(
        f"DiffRhythm 2 latents must have mel_dim={mel_dim} on axis 1 or 2; got {latents_mx.shape}"
    )


def make_fake_stereo(audio_np: np.ndarray, sampling_rate: int) -> np.ndarray:
    """Mono -> fake stereo (upstream DiffRhythm2 ``inference.py``)."""
    left_channel = audio_np
    right_channel = audio_np.copy()
    right_channel = right_channel * 0.8
    delay_samples = int(0.01 * sampling_rate)
    right_channel = np.roll(right_channel, delay_samples, axis=-1)
    if right_channel.ndim == 2:
        right_channel[:, :delay_samples] = 0
    else:
        right_channel[:delay_samples] = 0
    return np.concatenate([left_channel, right_channel], axis=0)


class DiffRhythm2DecoderMLX(nn.Module):
    """BigVGAN decoder wrapper with chunked ``decode_audio`` (upstream Generator)."""

    def __init__(self, ctx: Any, vae_dir: str):
        super().__init__()
        self._ctx = ctx
        self._vae_dir = Path(vae_dir)
        self.h = load_decoder_hparams(self._vae_dir / "decoder.json")

        self.generator = DiffRhythm2BigVGANMLX(
            in_channels=int(self.h["in_channels"]),
            upsample_initial_channel=int(self.h["upsample_initial_channel"]),
            upsample_rates=tuple(int(x) for x in self.h["upsample_rates"]),
            upsample_kernel_sizes=tuple(int(x) for x in self.h["upsample_kernel_sizes"]),
            resblock_kernel_sizes=tuple(int(x) for x in self.h["resblock_kernel_sizes"]),
            resblock_dilation_sizes=tuple(tuple(int(d) for d in row) for row in self.h["resblock_dilation_sizes"]),
            out_channels=1,
            apply_final_activation=bool(self.h.get("use_tanh_at_final", False)),
            use_bias_at_final=bool(self.h.get("use_bias_at_final", False)),
        )
        self.sampling_rate = int(self.h.get("sampling_rate", SAMPLE_RATE))
        self.samples_per_latent = int(math.prod(self.h["upsample_rates"]))

        array_fn = getattr(ctx, "array", mx.array)
        ckpt = self._vae_dir / "decoder.bin"
        load_decoder_weights(self.generator, str(ckpt), array_fn)
        _eval(self.generator)

    def decode_audio(
        self,
        latents_mx: mx.array,
        overlap: int = 5,
        chunk_size: int = 20,
    ) -> mx.array:
        """Chunked latent decode -> waveform ``[B, T_samples, 1]``."""
        latents_btc = _latents_to_btc(latents_mx, mel_dim=int(self.h["in_channels"]))
        latents_bct = latents_btc.transpose(0, 2, 1)

        hop_size = chunk_size - overlap
        total_size = int(latents_bct.shape[2])
        batch_size = int(latents_bct.shape[0])

        chunk_slices: list[tuple[int, int]] = []
        i = 0
        while i <= total_size - chunk_size:
            chunk_slices.append((i, i + chunk_size))
            i += hop_size
        if not chunk_slices or chunk_slices[-1][1] != total_size:
            chunk_slices.append((max(0, total_size - chunk_size), total_size))

        y_size = total_size * self.samples_per_latent
        y_final = mx.zeros((batch_size, y_size, 1))
        num_chunks = len(chunk_slices)
        ol = (overlap // 2) * self.samples_per_latent

        for chunk_idx, (start, end) in enumerate(chunk_slices):
            x_chunk = latents_btc[:, start:end, :]
            y_chunk = self.generator(x_chunk)
            chunk_len = int(y_chunk.shape[1])

            if chunk_idx == num_chunks - 1:
                t_end = y_size
                t_start = t_end - chunk_len
            else:
                t_start = chunk_idx * hop_size * self.samples_per_latent
                t_end = t_start + chunk_size * self.samples_per_latent

            chunk_start = 0
            chunk_end = chunk_len
            if chunk_idx > 0:
                t_start += ol
                chunk_start += ol
            if chunk_idx < num_chunks - 1:
                t_end -= ol
                chunk_end -= ol

            y_final = y_final.at[:, t_start:t_end, :].add(
                y_chunk[:, chunk_start:chunk_end, :]
            )
            _eval(y_final)

        return y_final

    def decode(self, latents: mx.array) -> mx.array:
        """Decode latents to waveform ``[B, T_samples, 1]`` (alias for ``decode_audio``)."""
        return self.decode_audio(latents)

    def encode(self, audio: mx.array) -> mx.array:
        raise NotImplementedError(
            "DiffRhythm 2 MLX path provides BigVGAN decode only; encode is not implemented"
        )

    def encode_mean(self, audio: mx.array) -> mx.array:
        raise NotImplementedError(
            "DiffRhythm 2 MLX path provides BigVGAN decode only; encode_mean is not implemented"
        )


# Public alias for ``vae.py`` dispatcher.
DiffRhythmVAEMLX = DiffRhythm2DecoderMLX
