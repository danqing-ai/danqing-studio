"""Wan 2.2 3D causal VAE — PyTorch (CUDA) port of official ``Wan2_2_VAE``.

PyTorch checkpoint tensors use ``[out_c, in_c, kt, kh, kw]`` for Conv3d / ``[out_c, in_c, kh, kw]`` for Conv2d.
Internal activations use **NCTHW** between layers.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from backend.engine.common.ops.attention import scaled_dot_product_attention_bhsd_torch
from backend.engine.runtime.cuda import CudaContext

logger = logging.getLogger(__name__)

CACHE_T = 2

# Official Wan2.2 VAE latent mean / std (48-dim).
_WAN22_VAE_MEAN = [
    -0.2289, -0.0052, -0.1323, -0.2339, -0.2799, 0.0174, 0.1838, 0.1557,
    -0.1382, 0.0542, 0.2813, 0.0891, 0.1570, -0.0098, 0.0375, -0.1825,
    -0.2246, -0.1207, -0.0698, 0.5109, 0.2665, -0.2108, -0.2158, 0.2502,
    -0.2055, -0.0322, 0.1109, 0.1567, -0.0729, 0.0899, -0.2799, -0.1230,
    -0.0313, -0.1649, 0.0117, 0.0723, -0.2839, -0.2083, -0.0520, 0.3748,
    0.0152, 0.1957, 0.1433, -0.2944, 0.3573, -0.0548, -0.1681, -0.0667,
]
_WAN22_VAE_STD = [
    0.4765, 1.0364, 0.4514, 1.1677, 0.5313, 0.4990, 0.4818, 0.5013,
    0.8158, 1.0344, 0.5894, 1.0901, 0.6885, 0.6165, 0.8454, 0.4978,
    0.5759, 0.3523, 0.7135, 0.6804, 0.5833, 1.4146, 0.8986, 0.5659,
    0.7069, 0.5338, 0.4889, 0.4917, 0.4069, 0.4999, 0.6866, 0.4093,
    0.5709, 0.6065, 0.6415, 0.4944, 0.5726, 1.2042, 0.5458, 1.6887,
    0.3971, 1.0600, 0.3943, 0.5537, 0.5444, 0.4089, 0.7468, 0.7744,
]


def _pad_ncthw_torch(x: torch.Tensor, *, pad_t: int, pad_h: int, pad_w: int) -> torch.Tensor:
    return F.pad(x, (pad_w, pad_w, pad_h, pad_h, 0, pad_t))


def _l2_normalize_torch(x: torch.Tensor, axis: int) -> torch.Tensor:
    denom = torch.sqrt(torch.sum(x * x, dim=axis, keepdim=True) + 1e-12)
    return x / denom


def _upsample_nearest_2d_nchw_torch(x: torch.Tensor) -> torch.Tensor:
    return F.interpolate(x, scale_factor=2, mode="nearest")


def _zero_pad2d_nchw_torch(x: torch.Tensor, pad_left: int, pad_right: int, pad_top: int, pad_bottom: int) -> torch.Tensor:
    if pad_left <= 0 and pad_right <= 0 and pad_top <= 0 and pad_bottom <= 0:
        return x
    return F.pad(x, (pad_left, pad_right, pad_top, pad_bottom))


# ---------------------------------------------------------------------------
# CausalConv3d (torch)
# ---------------------------------------------------------------------------

class CausalConv3dTorch(nn.Module):
    """Official Wan causal 3D convolution (NCTHW activations)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int],
        stride: int | tuple[int, int, int] = 1,
        padding: int | tuple[int, int, int] = 0,
        bias: bool = True,
    ):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding, padding)
        pt, ph, pw = padding
        self._pad_t = 2 * pt
        self._pad_h = ph
        self._pad_w = pw
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,
            bias=bias,
        )

    def forward(self, x_ncthw: torch.Tensor, cache_x: torch.Tensor | None = None) -> torch.Tensor:
        pad_t = self._pad_t
        x = x_ncthw
        if cache_x is not None and pad_t > 0:
            x = torch.cat([cache_x, x], dim=2)
            pad_t -= int(cache_x.shape[2])
        x = _pad_ncthw_torch(x, pad_t=pad_t, pad_h=self._pad_h, pad_w=self._pad_w)
        return self.conv(x)


# ---------------------------------------------------------------------------
# RMSNorm (torch)
# ---------------------------------------------------------------------------

class WanVAERMSNormTorch(nn.Module):
    def __init__(self, dim: int, *, channel_first: bool = True, images: bool = True, bias: bool = False):
        super().__init__()
        del channel_first
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims)
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else None
        self._images = images

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        axis = 1 if x.ndim == 5 or (x.ndim == 4 and self._images) else -1
        out = _l2_normalize_torch(x, axis=axis) * self.scale * self.gamma
        if self.bias is not None:
            out = out + self.bias
        return out


# ---------------------------------------------------------------------------
# Resample (torch)
# ---------------------------------------------------------------------------

class ResampleTorch(nn.Module):
    def __init__(self, dim: int, mode: str):
        super().__init__()
        assert mode in ("none", "upsample2d", "upsample3d", "downsample2d", "downsample3d")
        self.dim = dim
        self.mode = mode
        if mode == "upsample2d":
            self.resample = ("upsample2d", dim)
        elif mode == "upsample3d":
            self.resample = ("upsample3d", dim)
            self.time_conv = CausalConv3dTorch(dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))
        elif mode == "downsample2d":
            self.resample = ("downsample2d", dim)
        elif mode == "downsample3d":
            self.resample = ("downsample3d", dim)
            self.time_conv = CausalConv3dTorch(dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))
        else:
            self.resample = ("none", dim)

        if mode in ("upsample2d", "upsample3d"):
            self.resample_conv = nn.Conv2d(dim, dim, 3, padding=1)
        elif mode in ("downsample2d", "downsample3d"):
            self.resample_conv = nn.Conv2d(dim, dim, 3, stride=(2, 2))

    def forward(
        self,
        x: torch.Tensor,
        feat_cache: list[Any] | None = None,
        feat_idx: list[int] | None = None,
        first_chunk: bool = False,
    ) -> torch.Tensor:
        feat_idx = feat_idx if feat_idx is not None else [0]
        b, c, t, h, w = x.shape
        if self.mode == "upsample3d":
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = "Rep"
                    feat_idx[0] += 1
                else:
                    cache_x = x[:, :, -CACHE_T:, :, :]
                    if int(cache_x.shape[2]) < 2 and feat_cache[idx] is not None and feat_cache[idx] != "Rep":
                        cache_x = torch.cat(
                            [feat_cache[idx][:, :, -1:, :, :], cache_x],
                            dim=2,
                        )
                    if int(cache_x.shape[2]) < 2 and feat_cache[idx] is not None and feat_cache[idx] == "Rep":
                        cache_x = torch.cat([torch.zeros_like(cache_x), cache_x], dim=2)
                    if feat_cache[idx] == "Rep":
                        x = self.time_conv(x)
                    else:
                        x = self.time_conv(x, feat_cache[idx])
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1
                    x = x.reshape(b, 2, c, t, h, w)
                    x = torch.stack((x[:, 0], x[:, 1]), dim=3)
                    x = x.reshape(b, c, t * 2, h, w)
            elif first_chunk and t > 1:
                first_frame = x[:, :, :1, :, :]
                rest = x[:, :, 1:, :, :]
                tc_out = self.time_conv(rest)
                tc_out = tc_out.reshape(b, 2, c, t - 1, h, w)
                stream0 = tc_out[:, 0]
                stream1 = tc_out[:, 1]
                interleaved = torch.stack((stream0, stream1), dim=3)
                interleaved = interleaved.reshape(b, c, (t - 1) * 2, h, w)
                x = torch.cat([first_frame, interleaved], dim=2)
            else:
                tc_out = self.time_conv(x)
                tc_out = tc_out.reshape(b, 2, c, t, h, w)
                stream0 = tc_out[:, 0]
                stream1 = tc_out[:, 1]
                x = torch.stack((stream0, stream1), dim=3)
                x = x.reshape(b, c, t * 2, h, w)
        t = int(x.shape[2])
        x_bt = x.reshape(b * t, c, h, w)
        mode_tag = self.resample[0]
        if mode_tag in ("upsample2d", "upsample3d"):
            x_bt = _upsample_nearest_2d_nchw_torch(x_bt)
        elif mode_tag in ("downsample2d", "downsample3d"):
            x_bt = _zero_pad2d_nchw_torch(x_bt, 0, 1, 0, 1)
        x_bt = self.resample_conv(x_bt)
        h2, w2 = int(x_bt.shape[2]), int(x_bt.shape[3])
        x = x_bt.reshape(b, c, t, h2, w2)

        if self.mode == "downsample3d":
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = x
                    feat_idx[0] += 1
                else:
                    cache_x = x[:, :, -1:, :, :]
                    x = self.time_conv(torch.cat([feat_cache[idx][:, :, -1:, :, :], x], dim=2))
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1
        return x


# ---------------------------------------------------------------------------
# ResidualBlock (torch)
# ---------------------------------------------------------------------------

class ResidualBlockTorch(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        del dropout
        self.shortcut = CausalConv3dTorch(in_dim, out_dim, 1) if in_dim != out_dim else None
        self.norm1 = WanVAERMSNormTorch(in_dim, images=False)
        self.conv1 = CausalConv3dTorch(in_dim, out_dim, 3, padding=1)
        self.norm2 = WanVAERMSNormTorch(out_dim, images=False)
        self.conv2 = CausalConv3dTorch(out_dim, out_dim, 3, padding=1)

    def forward(
        self,
        x: torch.Tensor,
        feat_cache: list[Any] | None = None,
        feat_idx: list[int] | None = None,
    ) -> torch.Tensor:
        feat_idx = feat_idx if feat_idx is not None else [0]
        h = x if self.shortcut is None else self.shortcut(x)
        layers = (self.norm1, "silu", self.conv1, self.norm2, "silu", self.conv2)
        for layer in layers:
            if layer == "silu":
                x = F.silu(x)
            elif isinstance(layer, CausalConv3dTorch) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :]
                if int(cache_x.shape[2]) < 2 and feat_cache[idx] is not None:
                    cache_x = torch.cat(
                        [feat_cache[idx][:, :, -1:, :, :], cache_x],
                        dim=2,
                    )
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            elif isinstance(layer, (WanVAERMSNormTorch, CausalConv3dTorch)):
                x = layer(x)
        return x + h


# ---------------------------------------------------------------------------
# AttentionBlock (torch)
# ---------------------------------------------------------------------------

class AttentionBlockTorch(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm = WanVAERMSNormTorch(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        b, c, t, h, w = x.shape
        x_bt = x.reshape(b * t, c, h, w)
        x_bt = self.norm(x_bt)
        qkv = self.to_qkv(x_bt)
        qkv = qkv.reshape(b * t, h * w, 3, c)
        qkv = qkv.permute(0, 2, 1, 3)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        q = q.reshape(b * t, 1, h * w, c)
        k = k.reshape(b * t, 1, h * w, c)
        v = v.reshape(b * t, 1, h * w, c)
        attn = scaled_dot_product_attention_bhsd_torch(q, k, v, scale=(c ** -0.5))
        attn = attn.reshape(b * t, c, h, w)
        out = self.proj(attn)
        out = out.reshape(b, c, t, h, w)
        return out + identity


# ---------------------------------------------------------------------------
# Patchify / Unpatchify (torch)
# ---------------------------------------------------------------------------

def patchify_torch(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    if patch_size == 1:
        return x
    if x.ndim == 4:
        b, c, h, w = x.shape
        q = patch_size
        x = x.reshape(b, c, h // q, q, w // q, q)
        x = x.permute(0, 1, 3, 5, 2, 4)
        return x.reshape(b, c * q * q, h // q, w // q)
    if x.ndim == 5:
        b, c, f, h, w = x.shape
        q = patch_size
        x = x.reshape(b, c, f, h // q, q, w // q, q)
        x = x.permute(0, 1, 6, 4, 2, 3, 5)
        return x.reshape(b, c * q * q, f, h // q, w // q)
    raise RuntimeError(f"Invalid input shape for patchify: {x.shape}")


def unpatchify_torch(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    if patch_size == 1:
        return x
    if x.ndim == 4:
        b, c, h, w = x.shape
        q = patch_size
        c0 = c // (q * q)
        x = x.reshape(b, c0, q, q, h, w)
        x = x.permute(0, 1, 4, 2, 5, 3)
        return x.reshape(b, c0, h * q, w * q)
    if x.ndim == 5:
        b, c, f, h, w = x.shape
        q = patch_size
        c0 = c // (q * q)
        x = x.reshape(b, c0, q, q, f, h, w)
        x = x.permute(0, 1, 4, 5, 3, 6, 2)
        return x.reshape(b, c0, f, h * q, w * q)
    raise RuntimeError(f"Invalid input shape for unpatchify: {x.shape}")


# ---------------------------------------------------------------------------
# AvgDown3D / DupUp3D (torch)
# ---------------------------------------------------------------------------

class AvgDown3DTorch(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, factor_t: int, factor_s: int = 1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor_t = factor_t
        self.factor_s = factor_s
        self.factor = factor_t * factor_s * factor_s
        if (in_channels * self.factor) % out_channels != 0:
            raise RuntimeError("AvgDown3D channel/group mismatch")
        self.group_size = in_channels * self.factor // out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad_t = (self.factor_t - int(x.shape[2]) % self.factor_t) % self.factor_t
        if pad_t > 0:
            x = F.pad(x, (0, 0, 0, 0, pad_t, 0))
        b, c, t, h, w = x.shape
        ft, fs = self.factor_t, self.factor_s
        x = x.reshape(b, c, t // ft, ft, h // fs, fs, w // fs, fs)
        x = x.permute(0, 1, 3, 5, 7, 2, 4, 6)
        x = x.reshape(b, c * self.factor, t // ft, h // fs, w // fs)
        x = x.reshape(b, self.out_channels, self.group_size, t // ft, h // fs, w // fs)
        return x.mean(dim=2)


class DupUp3DTorch(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, factor_t: int, factor_s: int = 1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor_t = factor_t
        self.factor_s = factor_s
        self.factor = factor_t * factor_s * factor_s
        if (out_channels * self.factor) % in_channels != 0:
            raise RuntimeError("DupUp3D channel/repeat mismatch")
        self.repeats = out_channels * self.factor // in_channels

    def forward(self, x: torch.Tensor, first_chunk: bool = False) -> torch.Tensor:
        x = _repeat_interleave_channels_torch(x, self.repeats)
        b, c, t, h, w = x.shape
        ft, fs = self.factor_t, self.factor_s
        x = x.reshape(b, self.out_channels, ft, fs, fs, t, h, w)
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4)
        x = x.reshape(b, self.out_channels, t * ft, h * fs, w * fs)
        if first_chunk:
            x = x[:, :, self.factor_t - 1 :, :, :]
        return x


def _repeat_interleave_channels_torch(x: torch.Tensor, repeats: int) -> torch.Tensor:
    if repeats <= 1:
        return x
    b, c, t, h, w = x.shape
    x = x.reshape(b, c, 1, t, h, w)
    x = x.repeat(1, 1, repeats, 1, 1, 1)
    return x.reshape(b, c * repeats, t, h, w)


# ---------------------------------------------------------------------------
# Down / Up Residual Blocks (torch)
# ---------------------------------------------------------------------------

class DownResidualBlockTorch(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dropout: float,
        mult: int,
        *,
        temperal_downsample: bool = False,
        down_flag: bool = False,
    ):
        super().__init__()
        self.avg_shortcut = AvgDown3DTorch(
            in_dim,
            out_dim,
            factor_t=2 if temperal_downsample else 1,
            factor_s=2 if down_flag else 1,
        )
        blocks: list[nn.Module] = []
        cur = in_dim
        for _ in range(mult):
            blocks.append(ResidualBlockTorch(cur, out_dim, dropout))
            cur = out_dim
        if down_flag:
            mode = "downsample3d" if temperal_downsample else "downsample2d"
            blocks.append(ResampleTorch(out_dim, mode))
        self.blocks = nn.ModuleList(blocks)

    def forward(
        self,
        x: torch.Tensor,
        feat_cache: list[Any] | None = None,
        feat_idx: list[int] | None = None,
    ) -> torch.Tensor:
        x_copy = x
        for module in self.blocks:
            if isinstance(module, (ResidualBlockTorch, ResampleTorch)):
                x = module(x, feat_cache, feat_idx)
            else:
                x = module(x)
        return x + self.avg_shortcut(x_copy)


class UpResidualBlockTorch(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dropout: float,
        mult: int,
        *,
        temperal_upsample: bool = False,
        up_flag: bool = False,
    ):
        super().__init__()
        self.avg_shortcut = (
            DupUp3DTorch(in_dim, out_dim, factor_t=2 if temperal_upsample else 1, factor_s=2 if up_flag else 1)
            if up_flag
            else None
        )
        blocks: list[nn.Module] = []
        cur = in_dim
        for _ in range(mult):
            blocks.append(ResidualBlockTorch(cur, out_dim, dropout))
            cur = out_dim
        if up_flag:
            mode = "upsample3d" if temperal_upsample else "upsample2d"
            blocks.append(ResampleTorch(out_dim, mode))
        self.blocks = nn.ModuleList(blocks)

    def forward(
        self,
        x: torch.Tensor,
        feat_cache: list[Any] | None = None,
        feat_idx: list[int] | None = None,
        first_chunk: bool = False,
    ) -> torch.Tensor:
        x_main = x
        for module in self.blocks:
            if isinstance(module, ResampleTorch):
                x_main = module(x_main, feat_cache, feat_idx, first_chunk)
            elif isinstance(module, ResidualBlockTorch):
                x_main = module(x_main, feat_cache, feat_idx)
            else:
                x_main = module(x_main)
        if self.avg_shortcut is not None:
            return x_main + self.avg_shortcut(x, first_chunk)
        return x_main


# ---------------------------------------------------------------------------
# Encoder3d / Decoder3d (torch)
# ---------------------------------------------------------------------------

class Encoder3dTorch(nn.Module):
    def __init__(
        self,
        dim: int = 128,
        z_dim: int = 4,
        dim_mult: list[int] | None = None,
        num_res_blocks: int = 2,
        temperal_downsample: list[bool] | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        dim_mult = dim_mult or [1, 2, 4, 4]
        temperal_downsample = temperal_downsample or [True, True, False]
        dims = [dim * u for u in [1] + dim_mult]
        self.conv1 = CausalConv3dTorch(12, dims[0], 3, padding=1)
        downsamples: list[DownResidualBlockTorch] = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            t_down = temperal_downsample[i] if i < len(temperal_downsample) else False
            downsamples.append(
                DownResidualBlockTorch(
                    in_dim,
                    out_dim,
                    dropout,
                    num_res_blocks,
                    temperal_downsample=t_down,
                    down_flag=i != len(dim_mult) - 1,
                )
            )
        self.downsamples = nn.ModuleList(downsamples)
        self.mid_res0 = ResidualBlockTorch(out_dim, out_dim, dropout)
        self.mid_attn = AttentionBlockTorch(out_dim)
        self.mid_res1 = ResidualBlockTorch(out_dim, out_dim, dropout)
        self.head_norm = WanVAERMSNormTorch(out_dim, images=False)
        self.head_conv = CausalConv3dTorch(out_dim, z_dim, 3, padding=1)

    def forward(
        self,
        x: torch.Tensor,
        feat_cache: list[Any] | None = None,
        feat_idx: list[int] | None = None,
    ) -> torch.Tensor:
        feat_idx = feat_idx if feat_idx is not None else [0]
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :]
            if int(cache_x.shape[2]) < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat(
                    [feat_cache[idx][:, :, -1:, :, :], cache_x],
                    dim=2,
                )
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        for layer in self.downsamples:
            x = layer(x, feat_cache, feat_idx)

        for layer in (self.mid_res0, self.mid_attn, self.mid_res1):
            if isinstance(layer, ResidualBlockTorch) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        x = self.head_norm(x)
        x = F.silu(x)
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :]
            if int(cache_x.shape[2]) < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat(
                    [feat_cache[idx][:, :, -1:, :, :], cache_x],
                    dim=2,
                )
            x = self.head_conv(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.head_conv(x)
        return x


class Decoder3dTorch(nn.Module):
    def __init__(
        self,
        dim: int = 128,
        z_dim: int = 4,
        dim_mult: list[int] | None = None,
        num_res_blocks: int = 2,
        temperal_upsample: list[bool] | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        dim_mult = dim_mult or [1, 2, 4, 4]
        temperal_upsample = temperal_upsample or [False, True, True]
        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        self.conv1 = CausalConv3dTorch(z_dim, dims[0], 3, padding=1)
        self.mid_res0 = ResidualBlockTorch(dims[0], dims[0], dropout)
        self.mid_attn = AttentionBlockTorch(dims[0])
        self.mid_res1 = ResidualBlockTorch(dims[0], dims[0], dropout)
        upsamples: list[UpResidualBlockTorch] = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            t_up = temperal_upsample[i] if i < len(temperal_upsample) else False
            upsamples.append(
                UpResidualBlockTorch(
                    in_dim,
                    out_dim,
                    dropout,
                    num_res_blocks + 1,
                    temperal_upsample=t_up,
                    up_flag=i != len(dim_mult) - 1,
                )
            )
        self.upsamples = nn.ModuleList(upsamples)
        self.head_norm = WanVAERMSNormTorch(out_dim, images=False)
        self.head_conv = CausalConv3dTorch(out_dim, 12, 3, padding=1)

    def forward(
        self,
        x: torch.Tensor,
        feat_cache: list[Any] | None = None,
        feat_idx: list[int] | None = None,
        first_chunk: bool = False,
    ) -> torch.Tensor:
        feat_idx = feat_idx if feat_idx is not None else [0]
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :]
            if int(cache_x.shape[2]) < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat(
                    [feat_cache[idx][:, :, -1:, :, :], cache_x],
                    dim=2,
                )
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        for layer in (self.mid_res0, self.mid_attn, self.mid_res1):
            if isinstance(layer, ResidualBlockTorch) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        for layer in self.upsamples:
            if isinstance(layer, UpResidualBlockTorch):
                x = layer(x, feat_cache, feat_idx, first_chunk)
            elif isinstance(layer, ResidualBlockTorch):
                x = layer(x, feat_cache, feat_idx) if feat_cache is not None else layer(x)
            else:
                x = layer(x)

        x = self.head_norm(x)
        x = F.silu(x)
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :]
            if int(cache_x.shape[2]) < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat(
                    [feat_cache[idx][:, :, -1:, :, :], cache_x],
                    dim=2,
                )
            x = self.head_conv(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.head_conv(x)
        return x


# ---------------------------------------------------------------------------
# WanVAE (torch)
# ---------------------------------------------------------------------------

def _count_conv3d_torch(model: nn.Module) -> int:
    count = 0
    for mod in model.modules():
        if isinstance(mod, CausalConv3dTorch):
            count += 1
    return count


class WanVAETorch(nn.Module):
    def __init__(
        self,
        dim: int = 160,
        dec_dim: int = 256,
        z_dim: int = 48,
        dim_mult: list[int] | None = None,
        num_res_blocks: int = 2,
        temperal_downsample: list[bool] | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        dim_mult = dim_mult or [1, 2, 4, 4]
        temperal_downsample = temperal_downsample or [False, True, True]
        self.z_dim = z_dim
        self.temperal_upsample = temperal_downsample[::-1]
        self.encoder = Encoder3dTorch(
            dim,
            z_dim * 2,
            dim_mult,
            num_res_blocks,
            temperal_downsample,
            dropout,
        )
        self.conv1 = CausalConv3dTorch(z_dim * 2, z_dim * 2, 1)
        self.conv2 = CausalConv3dTorch(z_dim, z_dim, 1)
        self.decoder = Decoder3dTorch(
            dec_dim,
            z_dim,
            dim_mult,
            num_res_blocks,
            self.temperal_upsample,
            dropout,
        )
        self._conv_num = 0
        self._conv_idx = [0]
        self._feat_map: list[Any] = []
        self._enc_conv_num = 0
        self._enc_conv_idx = [0]
        self._enc_feat_map: list[Any] = []

    def clear_cache(self) -> None:
        self._conv_num = _count_conv3d_torch(self.decoder)
        self._conv_idx = [0]
        self._feat_map = [None] * self._conv_num
        self._enc_conv_num = _count_conv3d_torch(self.encoder)
        self._enc_conv_idx = [0]
        self._enc_feat_map = [None] * self._enc_conv_num

    def encode(self, x: torch.Tensor, scale: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        self.clear_cache()
        x = patchify_torch(x, 2)
        t = int(x.shape[2])
        iter_ = 1 + (t - 1) // 4
        out_parts: list[torch.Tensor] = []
        for i in range(iter_):
            self._enc_conv_idx = [0]
            if i == 0:
                chunk = x[:, :, :1, :, :]
            else:
                chunk = x[:, :, 1 + 4 * (i - 1) : 1 + 4 * i, :, :]
            part = self.encoder(chunk, self._enc_feat_map, self._enc_conv_idx)
            out_parts.append(part)
        out = out_parts[0] if len(out_parts) == 1 else torch.cat(out_parts, dim=2)
        mu, _log_var = torch.chunk(self.conv1(out), 2, dim=1)
        mean, inv_std = scale
        mu = (mu - mean) * inv_std
        self.clear_cache()
        return mu

    def decode(
        self,
        z: torch.Tensor,
        scale: tuple[torch.Tensor, torch.Tensor],
        on_stage: Callable[[float], None] | None = None,
    ) -> torch.Tensor:
        self.clear_cache()
        mean, inv_std = scale
        z = z / inv_std + mean
        if on_stage is not None:
            on_stage(0.05)
        x = self.conv2(z)
        out = self.decoder(x, None, [0], first_chunk=True)
        out = unpatchify_torch(out, 2)
        if on_stage is not None:
            on_stage(1.0)
        self.clear_cache()
        return out


# ---------------------------------------------------------------------------
# Tiled decode helpers (torch)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _WanVaeTileParams:
    tile_sample_min_height: int
    tile_sample_min_width: int
    tile_latent_min_height: int
    tile_latent_min_width: int
    overlap_factor: float


def _wan_vae_tile_params(vae_cfg: dict[str, Any], spatial_scale: int) -> _WanVaeTileParams:
    spatial_up = max(1, int(spatial_scale))
    sample_h = int(vae_cfg.get("tile_sample_min_height", 256))
    sample_w = int(vae_cfg.get("tile_sample_min_width", 256))
    overlap = float(vae_cfg.get("tile_overlap_factor", 0.25))
    return _WanVaeTileParams(
        tile_sample_min_height=sample_h,
        tile_sample_min_width=sample_w,
        tile_latent_min_height=max(1, sample_h // spatial_up),
        tile_latent_min_width=max(1, sample_w // spatial_up),
        overlap_factor=overlap,
    )


def _needs_wan_spatial_tiling(
    latent_h: int,
    latent_w: int,
    params: _WanVaeTileParams,
    *,
    enabled: bool,
) -> bool:
    if not enabled:
        return False
    return (
        latent_h > params.tile_latent_min_height
        or latent_w > params.tile_latent_min_width
    )


def _blend_v_wan_torch(a_ncthw: torch.Tensor, b_ncthw: torch.Tensor, blend_extent: int) -> torch.Tensor:
    be = min(int(a_ncthw.shape[3]), int(b_ncthw.shape[3]), int(blend_extent))
    if be <= 0:
        return b_ncthw
    rows: list[torch.Tensor] = []
    for y in range(int(b_ncthw.shape[3])):
        if y < be:
            alpha = float(y) / float(be)
            merged = (
                a_ncthw[:, :, :, -be + y, :] * (1.0 - alpha)
                + b_ncthw[:, :, :, y, :] * alpha
            )
            rows.append(merged.unsqueeze(3))
        else:
            rows.append(b_ncthw[:, :, :, y, :].unsqueeze(3))
    return torch.cat(rows, dim=3)


def _blend_h_wan_torch(a_ncthw: torch.Tensor, b_ncthw: torch.Tensor, blend_extent: int) -> torch.Tensor:
    be = min(int(a_ncthw.shape[4]), int(b_ncthw.shape[4]), int(blend_extent))
    if be <= 0:
        return b_ncthw
    cols: list[torch.Tensor] = []
    for x in range(int(b_ncthw.shape[4])):
        if x < be:
            alpha = float(x) / float(be)
            merged = (
                a_ncthw[:, :, :, :, -be + x] * (1.0 - alpha)
                + b_ncthw[:, :, :, :, x] * alpha
            )
            cols.append(merged.unsqueeze(4))
        else:
            cols.append(b_ncthw[:, :, :, :, x].unsqueeze(4))
    return torch.cat(cols, dim=4)


def _decode_wan_latent_volume_torch(
    model: WanVAETorch,
    latents: torch.Tensor,
    scale: tuple[torch.Tensor, torch.Tensor],
    on_stage: Callable[[float], None] | None = None,
) -> torch.Tensor:
    model.clear_cache()
    return model.decode(latents, scale, on_stage=on_stage)


def _tiled_spatial_decode_wan_torch(
    model: WanVAETorch,
    latents: torch.Tensor,
    scale: tuple[torch.Tensor, torch.Tensor],
    params: _WanVaeTileParams,
    *,
    on_stage: Callable[[float], None] | None = None,
    on_log: Callable[[str], None] | None = None,
) -> torch.Tensor:
    _, _, _, height, width = latents.shape
    overlap_height = max(1, int(params.tile_latent_min_height * (1.0 - params.overlap_factor)))
    overlap_width = max(1, int(params.tile_latent_min_width * (1.0 - params.overlap_factor)))
    blend_height = int(params.tile_sample_min_height * params.overlap_factor)
    blend_width = int(params.tile_sample_min_width * params.overlap_factor)
    row_limit_height = params.tile_sample_min_height - blend_height
    row_limit_width = params.tile_sample_min_width - blend_width

    row_starts = list(range(0, int(height), overlap_height))
    col_starts = list(range(0, int(width), overlap_width))
    total_tiles = len(row_starts) * len(col_starts)
    tile_idx = 0

    rows: list[list[torch.Tensor]] = []
    for i in row_starts:
        row: list[torch.Tensor] = []
        for j in col_starts:
            tile_idx += 1
            msg = (
                f"Wan VAE spatial tile {tile_idx}/{total_tiles} "
                f"(latent y={i}, x={j})"
            )
            logger.info(msg)
            if on_log is not None:
                on_log(msg)
            if on_stage is not None:
                on_stage(0.05 + 0.9 * ((tile_idx - 1) / max(total_tiles, 1)))

            z_tile = latents[
                :,
                :,
                :,
                i : i + params.tile_latent_min_height,
                j : j + params.tile_latent_min_width,
            ]

            def _tile_stage(frac: float) -> None:
                if on_stage is None:
                    return
                base = 0.05 + 0.9 * ((tile_idx - 1) / max(total_tiles, 1))
                span = 0.9 / max(total_tiles, 1)
                on_stage(min(1.0, base + span * float(frac)))

            decoded = _decode_wan_latent_volume_torch(
                model, z_tile, scale, on_stage=_tile_stage,
            )
            row.append(decoded)
        rows.append(row)

    result_rows: list[torch.Tensor] = []
    for ri, row in enumerate(rows):
        result_row: list[torch.Tensor] = []
        for ci, tile in enumerate(row):
            blended = tile
            if ri > 0:
                blended = _blend_v_wan_torch(rows[ri - 1][ci], blended, blend_height)
            if ci > 0:
                blended = _blend_h_wan_torch(row[ci - 1], blended, blend_width)
            out_h = int(blended.shape[3])
            out_w = int(blended.shape[4])
            rh = min(row_limit_height, out_h)
            rw = min(row_limit_width, out_w)
            result_row.append(blended[:, :, :, :rh, :rw])
        result_rows.append(torch.cat(result_row, dim=4))
    sample = torch.cat(result_rows, dim=3)
    return sample


# ---------------------------------------------------------------------------
# Wrapper + load
# ---------------------------------------------------------------------------

@dataclass
class Wan22VAECUDA:
    model: WanVAETorch
    scale_mean: torch.Tensor
    scale_inv_std: torch.Tensor
    z_dim: int = 48


def _set_causal_conv3d_torch(mod: CausalConv3dTorch, prefix: str, weights: dict[str, torch.Tensor]) -> None:
    mod.conv.weight.data.copy_(weights[f"{prefix}.weight"])
    mod.conv.bias.data.copy_(weights[f"{prefix}.bias"])


def _set_conv2d_torch(mod: nn.Conv2d, prefix: str, weights: dict[str, torch.Tensor]) -> None:
    mod.weight.data.copy_(weights[f"{prefix}.weight"])
    mod.bias.data.copy_(weights[f"{prefix}.bias"])


def _set_rms_norm_torch(mod: WanVAERMSNormTorch, prefix: str, weights: dict[str, torch.Tensor]) -> None:
    mod.gamma.data.copy_(weights[f"{prefix}.gamma"])


def _set_residual_block_torch(mod: ResidualBlockTorch, prefix: str, weights: dict[str, torch.Tensor]) -> None:
    _set_rms_norm_torch(mod.norm1, f"{prefix}.residual.0", weights)
    _set_causal_conv3d_torch(mod.conv1, f"{prefix}.residual.2", weights)
    _set_rms_norm_torch(mod.norm2, f"{prefix}.residual.3", weights)
    _set_causal_conv3d_torch(mod.conv2, f"{prefix}.residual.6", weights)
    if mod.shortcut is not None:
        _set_causal_conv3d_torch(mod.shortcut, f"{prefix}.shortcut", weights)


def _set_attention_block_torch(mod: AttentionBlockTorch, prefix: str, weights: dict[str, torch.Tensor]) -> None:
    _set_rms_norm_torch(mod.norm, f"{prefix}.norm", weights)
    _set_conv2d_torch(mod.to_qkv, f"{prefix}.to_qkv", weights)
    _set_conv2d_torch(mod.proj, f"{prefix}.proj", weights)


def _set_resample_torch(mod: ResampleTorch, prefix: str, weights: dict[str, torch.Tensor]) -> None:
    if mod.mode in ("upsample2d", "upsample3d", "downsample2d", "downsample3d"):
        _set_conv2d_torch(mod.resample_conv, f"{prefix}.resample.1", weights)
    if mod.mode in ("upsample3d", "downsample3d"):
        _set_causal_conv3d_torch(mod.time_conv, f"{prefix}.time_conv", weights)


def _set_down_block_torch(mod: DownResidualBlockTorch, prefix: str, weights: dict[str, torch.Tensor]) -> None:
    idx = 0
    for block in mod.blocks:
        if isinstance(block, ResidualBlockTorch):
            _set_residual_block_torch(block, f"{prefix}.downsamples.{idx}", weights)
            idx += 1
        elif isinstance(block, ResampleTorch):
            _set_resample_torch(block, f"{prefix}.downsamples.{idx}", weights)


def _set_up_block_torch(mod: UpResidualBlockTorch, prefix: str, weights: dict[str, torch.Tensor]) -> None:
    idx = 0
    for block in mod.blocks:
        if isinstance(block, ResidualBlockTorch):
            _set_residual_block_torch(block, f"{prefix}.upsamples.{idx}", weights)
            idx += 1
        elif isinstance(block, ResampleTorch):
            _set_resample_torch(block, f"{prefix}.upsamples.{idx}", weights)


def _assign_wan_vae_weights_torch(model: WanVAETorch, weights: dict[str, torch.Tensor]) -> None:
    _set_causal_conv3d_torch(model.encoder.conv1, "encoder.conv1", weights)
    for i, block in enumerate(model.encoder.downsamples):
        _set_down_block_torch(block, f"encoder.downsamples.{i}", weights)
    _set_residual_block_torch(model.encoder.mid_res0, "encoder.middle.0", weights)
    _set_attention_block_torch(model.encoder.mid_attn, "encoder.middle.1", weights)
    _set_residual_block_torch(model.encoder.mid_res1, "encoder.middle.2", weights)
    _set_rms_norm_torch(model.encoder.head_norm, "encoder.head.0", weights)
    _set_causal_conv3d_torch(model.encoder.head_conv, "encoder.head.2", weights)

    _set_causal_conv3d_torch(model.conv1, "conv1", weights)
    _set_causal_conv3d_torch(model.conv2, "conv2", weights)

    _set_causal_conv3d_torch(model.decoder.conv1, "decoder.conv1", weights)
    _set_residual_block_torch(model.decoder.mid_res0, "decoder.middle.0", weights)
    _set_attention_block_torch(model.decoder.mid_attn, "decoder.middle.1", weights)
    _set_residual_block_torch(model.decoder.mid_res1, "decoder.middle.2", weights)
    for i, block in enumerate(model.decoder.upsamples):
        _set_up_block_torch(block, f"decoder.upsamples.{i}", weights)
    _set_rms_norm_torch(model.decoder.head_norm, "decoder.head.0", weights)
    _set_causal_conv3d_torch(model.decoder.head_conv, "decoder.head.2", weights)


def _is_diffusers_vae_keys(keys: set[str]) -> bool:
    return any(k.startswith("encoder.conv_in.") or k.startswith("encoder.down_blocks.") for k in keys)


def _diffusers_to_official_vae_key(key: str) -> str:
    replacements = {
        "encoder.conv_in.": "encoder.conv1.",
        "decoder.conv_in.": "decoder.conv1.",
        "encoder.norm_out.": "encoder.head.0.",
        "encoder.conv_out.": "encoder.head.2.",
        "decoder.norm_out.": "decoder.head.0.",
        "decoder.conv_out.": "decoder.head.2.",
        "quant_conv.": "conv1.",
        "post_quant_conv.": "conv2.",
    }
    for old, new in replacements.items():
        if key.startswith(old):
            return new + key[len(old):]

    if key.startswith("encoder.down_blocks."):
        key = key.replace("encoder.down_blocks.", "encoder.downsamples.")
        key = re.sub(r"\.resnets\.(\d+)\.norm1\.", r".downsamples.\1.residual.0.", key)
        key = re.sub(r"\.resnets\.(\d+)\.conv1\.", r".downsamples.\1.residual.2.", key)
        key = re.sub(r"\.resnets\.(\d+)\.norm2\.", r".downsamples.\1.residual.3.", key)
        key = re.sub(r"\.resnets\.(\d+)\.conv2\.", r".downsamples.\1.residual.6.", key)
        key = re.sub(r"\.resnets\.(\d+)\.conv_shortcut\.", r".downsamples.\1.shortcut.", key)
        key = key.replace(".downsampler.resample.1.", ".downsamples.2.resample.1.")
        key = key.replace(".downsampler.time_conv.", ".downsamples.2.time_conv.")
    elif key.startswith("decoder.up_blocks."):
        key = key.replace("decoder.up_blocks.", "decoder.upsamples.")
        key = re.sub(r"\.resnets\.(\d+)\.norm1\.", r".upsamples.\1.residual.0.", key)
        key = re.sub(r"\.resnets\.(\d+)\.conv1\.", r".upsamples.\1.residual.2.", key)
        key = re.sub(r"\.resnets\.(\d+)\.norm2\.", r".upsamples.\1.residual.3.", key)
        key = re.sub(r"\.resnets\.(\d+)\.conv2\.", r".upsamples.\1.residual.6.", key)
        key = re.sub(r"\.resnets\.(\d+)\.conv_shortcut\.", r".upsamples.\1.shortcut.", key)
        key = key.replace(".upsampler.resample.1.", ".upsamples.3.resample.1.")
        key = key.replace(".upsampler.time_conv.", ".upsamples.3.time_conv.")

    key = key.replace("encoder.mid_block.resnets.0.", "encoder.middle.0.")
    key = key.replace("encoder.mid_block.resnets.1.", "encoder.middle.2.")
    key = key.replace("encoder.mid_block.attentions.0.", "encoder.middle.1.")
    key = key.replace("decoder.mid_block.resnets.0.", "decoder.middle.0.")
    key = key.replace("decoder.mid_block.resnets.1.", "decoder.middle.2.")
    key = key.replace("decoder.mid_block.attentions.0.", "decoder.middle.1.")

    key = key.replace("encoder.middle.0.norm1.", "encoder.middle.0.residual.0.")
    key = key.replace("encoder.middle.0.conv1.", "encoder.middle.0.residual.2.")
    key = key.replace("encoder.middle.0.norm2.", "encoder.middle.0.residual.3.")
    key = key.replace("encoder.middle.0.conv2.", "encoder.middle.0.residual.6.")
    key = key.replace("encoder.middle.0.conv_shortcut.", "encoder.middle.0.shortcut.")
    key = key.replace("encoder.middle.2.norm1.", "encoder.middle.2.residual.0.")
    key = key.replace("encoder.middle.2.conv1.", "encoder.middle.2.residual.2.")
    key = key.replace("encoder.middle.2.norm2.", "encoder.middle.2.residual.3.")
    key = key.replace("encoder.middle.2.conv2.", "encoder.middle.2.residual.6.")
    key = key.replace("encoder.middle.2.conv_shortcut.", "encoder.middle.2.shortcut.")

    key = key.replace("decoder.middle.0.norm1.", "decoder.middle.0.residual.0.")
    key = key.replace("decoder.middle.0.conv1.", "decoder.middle.0.residual.2.")
    key = key.replace("decoder.middle.0.norm2.", "decoder.middle.0.residual.3.")
    key = key.replace("decoder.middle.0.conv2.", "decoder.middle.0.residual.6.")
    key = key.replace("decoder.middle.0.conv_shortcut.", "decoder.middle.0.shortcut.")
    key = key.replace("decoder.middle.2.norm1.", "decoder.middle.2.residual.0.")
    key = key.replace("decoder.middle.2.conv1.", "decoder.middle.2.residual.2.")
    key = key.replace("decoder.middle.2.norm2.", "decoder.middle.2.residual.3.")
    key = key.replace("decoder.middle.2.conv2.", "decoder.middle.2.residual.6.")
    key = key.replace("decoder.middle.2.conv_shortcut.", "decoder.middle.2.shortcut.")
    return key


def _normalize_vae_state_dict_torch(raw: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if not _is_diffusers_vae_keys(set(raw)):
        return raw
    out: dict[str, torch.Tensor] = {}
    for key, value in raw.items():
        out[_diffusers_to_official_vae_key(key)] = value
    return out


def _read_wan_vae_config(bundle_root: Path) -> dict[str, Any]:
    cfg_path = bundle_root / "vae" / "config.json"
    if not cfg_path.is_file():
        return {}
    try:
        import json
        return json.loads(cfg_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise RuntimeError(f"Wan: cannot read VAE config {cfg_path}: {e}") from e


def _scale_tensors_from_vae_cfg_torch(
    cfg: dict[str, Any], z_dim: int, device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    mean = cfg.get("latents_mean")
    std = cfg.get("latents_std")
    if (
        isinstance(mean, list)
        and isinstance(std, list)
        and len(mean) == z_dim
        and len(std) == z_dim
    ):
        mean_a = torch.tensor(mean, dtype=torch.float32, device=device).reshape(1, z_dim, 1, 1, 1)
        inv_std = torch.tensor([1.0 / float(s) for s in std], dtype=torch.float32, device=device).reshape(
            1, z_dim, 1, 1, 1
        )
        return mean_a, inv_std
    mean_a = torch.tensor(_WAN22_VAE_MEAN, dtype=torch.float32, device=device).reshape(1, z_dim, 1, 1, 1)
    inv_std = torch.tensor([1.0 / s for s in _WAN22_VAE_STD], dtype=torch.float32, device=device).reshape(
        1, z_dim, 1, 1, 1
    )
    return mean_a, inv_std


def _load_vae_state_dict_torch(bundle_root: Path) -> dict[str, torch.Tensor]:
    pth_candidates = [
        bundle_root / "Wan2.2_VAE.pth",
        bundle_root / "Wan2_2_VAE.pth",
        bundle_root / "Wan2.1_VAE.pth",
    ]
    for pth in pth_candidates:
        if pth.is_file():
            logger.info("Loading Wan VAE weights from %s", pth)
            from backend.engine.common.bundle.pytorch_bin_numpy import state_dict_to_numpy
            sd = state_dict_to_numpy(pth)
            out: dict[str, torch.Tensor] = {}
            for k, v in sd.items():
                arr = np.asarray(v, dtype=np.float32)
                out[k] = torch.from_numpy(arr)
            return out

    vae_dir = bundle_root / "vae"
    if vae_dir.is_dir():
        merged: dict[str, torch.Tensor] = {}
        for sf in sorted(vae_dir.glob("*.safetensors")):
            import safetensors.torch
            merged.update(safetensors.torch.load_file(str(sf)))
        if merged:
            logger.info("Loading Wan VAE weights from %s (%d tensors)", vae_dir, len(merged))
            return _normalize_vae_state_dict_torch(merged)

    raise RuntimeError(
        f"Wan VAE weights not found under {bundle_root}: expected Wan2.2_VAE.pth or vae/*.safetensors"
    )


def build_wan22_vae_torch(
    *,
    z_dim: int = 48,
    c_dim: int = 160,
    dim_mult: list[int] | None = None,
    temperal_downsample: list[bool] | None = None,
) -> WanVAETorch:
    return WanVAETorch(
        dim=c_dim,
        dec_dim=256,
        z_dim=z_dim,
        dim_mult=dim_mult or [1, 2, 4, 4],
        temperal_downsample=temperal_downsample or [False, True, True],
        dropout=0.0,
    )


_vae_cache_cuda: dict[str, Wan22VAECUDA] = {}


def load_wan_vae_cuda(ctx: CudaContext, bundle_root: Path | str) -> Wan22VAECUDA:
    root = Path(bundle_root)
    if not root.is_dir():
        raise RuntimeError(f"Wan VAE bundle directory missing: {root}")

    key = str(root.resolve())
    cached = _vae_cache_cuda.get(key)
    if cached is not None:
        return cached

    weights = _load_vae_state_dict_torch(root)
    if "encoder.conv1.weight" not in weights and "encoder.conv_in.weight" in weights:
        weights = _normalize_vae_state_dict_torch(weights)

    vae_cfg = _read_wan_vae_config(root)
    z_dim = int(vae_cfg.get("z_dim", 48))
    model = build_wan22_vae_torch(
        z_dim=z_dim,
        c_dim=int(vae_cfg.get("base_dim", 160)),
        dim_mult=list(vae_cfg.get("dim_mult") or [1, 2, 4, 4]),
        temperal_downsample=list(vae_cfg.get("temperal_downsample") or [False, True, True]),
    )
    _assign_wan_vae_weights_torch(model, weights)
    model = model.to(ctx.device).eval()

    mean, inv_std = _scale_tensors_from_vae_cfg_torch(vae_cfg, z_dim, ctx.device)
    wrapper = Wan22VAECUDA(model=model, scale_mean=mean, scale_inv_std=inv_std, z_dim=model.z_dim)
    _vae_cache_cuda[key] = wrapper
    logger.info("Wan 2.2 VAE loaded from %s (z_dim=%d)", root, model.z_dim)
    return wrapper


def decode_wan_vae_latents_cuda(
    ctx: CudaContext,
    latents_bcthw: torch.Tensor,
    bundle_root: Path | str,
    on_stage: Callable[[float], None] | None = None,
    on_log: Callable[[str], None] | None = None,
    *,
    spatial_tiling: bool = False,
    spatial_scale: int = 16,
) -> torch.Tensor:
    """Decode ``[B,C,T,H,W]`` latents to RGB pixels ``[B,C,T,H,W]`` in ``[-1, 1]``."""
    vae = load_wan_vae_cuda(ctx, bundle_root)
    logger.info("Wan VAE decode: latent shape=%s", tuple(latents_bcthw.shape))
    if latents_bcthw.ndim != 5:
        raise RuntimeError(f"Wan VAE decode expects 5D latents [B,C,T,H,W], got {latents_bcthw.shape}")
    if int(latents_bcthw.shape[0]) != 1:
        raise RuntimeError(f"Wan VAE decode batch size must be 1, got {latents_bcthw.shape[0]}")

    def _stage(frac: float) -> None:
        if on_stage is not None:
            on_stage(min(1.0, max(0.0, float(frac))))

    scale = (vae.scale_mean, vae.scale_inv_std)
    vae_cfg = _read_wan_vae_config(Path(bundle_root))
    tile_params = _wan_vae_tile_params(vae_cfg, spatial_scale)
    _, _, _, lh, lw = latents_bcthw.shape

    _stage(0.02)
    if _needs_wan_spatial_tiling(int(lh), int(lw), tile_params, enabled=spatial_tiling):
        if on_log is not None:
            on_log(
                f"Wan VAE spatial tiling enabled (latent {lh}x{lw}, "
                f"tile {tile_params.tile_latent_min_height}x{tile_params.tile_latent_min_width})"
            )
        sample = _tiled_spatial_decode_wan_torch(
            vae.model,
            latents_bcthw,
            scale,
            tile_params,
            on_stage=_stage,
            on_log=on_log,
        )
    else:
        sample = _decode_wan_latent_volume_torch(
            vae.model, latents_bcthw, scale, on_stage=_stage,
        )
    sample = torch.clamp(sample, -1.0, 1.0)
    _stage(1.0)
    logger.info("Wan VAE decode done: pixel shape=%s", tuple(sample.shape))
    return sample


def encode_wan_vae_image_cuda(
    ctx: CudaContext,
    image_chw: torch.Tensor,
    bundle_root: Path | str,
) -> torch.Tensor:
    """Encode a single RGB frame ``[C,H,W]`` (float ``[-1,1]``) to latents ``[1,C,1,h,w]``."""
    vae = load_wan_vae_cuda(ctx, bundle_root)
    if image_chw.ndim != 3:
        raise RuntimeError(f"Wan VAE encode expects image [C,H,W], got {image_chw.shape}")
    pixels = image_chw.unsqueeze(0).unsqueeze(2)
    logger.info("Wan VAE encode: pixel shape=%s", tuple(pixels.shape))
    latents = vae.model.encode(pixels, (vae.scale_mean, vae.scale_inv_std))
    logger.info("Wan VAE encode done: latent shape=%s", tuple(latents.shape))
    return latents
