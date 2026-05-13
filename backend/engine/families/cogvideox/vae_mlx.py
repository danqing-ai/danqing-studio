"""CogVideoX VAE decoder — MLX port of diffusers ``CogVideoXDecoder3D``.

PyTorch checkpoint tensors use ``[out_c, in_c, kt, kh, kw]`` for Conv3d / ``[out_c, in_c, kh, kw]`` for Conv2d.
MLX Conv3d / Conv2d expect ``[out_c, kt, kh, kw, in_c]`` / ``[out_c, kh, kw, in_c]`` with activations in **NDHWC** / **NHWC**.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from backend.engine.runtime._base import RuntimeContext


def _conv3d_weight_torch_to_mlx(w: mx.array) -> mx.array:
    return mx.transpose(w, (0, 2, 3, 4, 1))


def _conv2d_weight_torch_to_mlx(w: mx.array) -> mx.array:
    return mx.transpose(w, (0, 2, 3, 1))


def _ncthw_to_ndhwc(x: mx.array) -> mx.array:
    return mx.transpose(x, (0, 2, 3, 4, 1))


def _ndhwc_to_ncthw(x: mx.array) -> mx.array:
    return mx.transpose(x, (0, 4, 1, 2, 3))


def _group_norm_ncthw(norm: nn.GroupNorm, x_ncthw: mx.array) -> mx.array:
    b, c, t, h, w = x_ncthw.shape
    x = mx.transpose(x_ncthw, (0, 2, 3, 4, 1))
    x = mx.reshape(x, (b * t, h, w, c))
    x = norm(x)
    x = mx.reshape(x, (b, t, h, w, c))
    return mx.transpose(x, (0, 4, 1, 2, 3))


def _resize_nearest_1d(x: mx.array, axis: int, new_size: int) -> mx.array:
    old = int(x.shape[axis])
    if new_size == old:
        return x
    if new_size <= 0:
        raise RuntimeError(f"invalid resize target size {new_size}")
    idx = mx.minimum(
        (mx.arange(new_size, dtype=mx.float32) * float(old - 1) / float(max(new_size - 1, 1))).astype(mx.int32),
        old - 1,
    )
    return mx.take(x, idx, axis=axis)


def _interpolate_nearest_ncthw(z: mx.array, target_t: int, target_h: int, target_w: int) -> mx.array:
    """5D nearest resize along (T,H,W) to match ``target_*`` (diffusers ``F.interpolate`` semantic)."""
    z = _resize_nearest_1d(z, 2, target_t)
    z = _resize_nearest_1d(z, 3, target_h)
    z = _resize_nearest_1d(z, 4, target_w)
    return z


def _swish(x: mx.array) -> mx.array:
    return nn.silu(x)


class SafeConv3d(nn.Module):
    """Maps to ``CogVideoXSafeConv3d`` — Conv3d on NDHWC without temporal chunking (MLX)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int],
        stride: int | tuple[int, int, int] = 1,
        padding: int | tuple[int, int, int] = 0,
        dilation: int | tuple[int, int, int] = 1,
        bias: bool = True,
    ):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3
        if isinstance(stride, int):
            stride = (stride, stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding, padding)
        if isinstance(dilation, int):
            dilation = (dilation, dilation, dilation)
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

    def __call__(self, x_ndhwc: mx.array) -> mx.array:
        return self.conv(x_ndhwc)


class CausalConv3d(nn.Module):
    """Maps to ``CogVideoXCausalConv3d`` (``pad_mode`` = ``first`` / non-replicate)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int],
        stride: int = 1,
        dilation: int = 1,
        pad_mode: str = "first",
    ):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3
        time_kernel_size, height_kernel_size, width_kernel_size = kernel_size
        self.pad_mode = pad_mode
        self.time_kernel_size = time_kernel_size
        height_pad = (height_kernel_size - 1) // 2
        width_pad = (width_kernel_size - 1) // 2
        self.time_pad = time_kernel_size - 1
        self.time_causal_padding = (width_pad, width_pad, height_pad, height_pad, self.time_pad, 0)
        self.const_padding_conv3d = (0, width_pad, height_pad)

        stride_tup = stride if isinstance(stride, tuple) else (stride, 1, 1)
        dilation_tup = (dilation, 1, 1)
        padding_mode = "zeros"
        self.conv = SafeConv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride_tup,
            padding=0 if pad_mode == "replicate" else self.const_padding_conv3d,
            dilation=dilation_tup,
            bias=True,
        )

    def __call__(self, inputs_ndhwc: mx.array, conv_cache: mx.array | None = None) -> tuple[mx.array, mx.array | None]:
        if self.pad_mode == "replicate":
            inputs_ndhwc = mx.pad(inputs_ndhwc, self.time_causal_padding)
            conv_cache_out = None
        else:
            k = self.time_kernel_size
            if k > 1:
                first = inputs_ndhwc[:, :1, :, :, :]
                if conv_cache is not None:
                    prepend = conv_cache
                else:
                    prepend = mx.repeat(first, k - 1, axis=1)
                inputs_ndhwc = mx.concatenate([prepend, inputs_ndhwc], axis=1)
            conv_cache_out = (
                None if self.pad_mode == "replicate" else inputs_ndhwc[:, -(self.time_kernel_size - 1) :, :, :, :]
            )

        out = self.conv(inputs_ndhwc)
        return out, conv_cache_out


class SpatialNorm3D(nn.Module):
    """Maps to ``CogVideoXSpatialNorm3D``."""

    def __init__(self, f_channels: int, zq_channels: int, groups: int = 32):
        super().__init__()
        self.norm_layer = nn.GroupNorm(num_groups=groups, dims=f_channels, eps=1e-6, pytorch_compatible=True)
        self.conv_y = CausalConv3d(zq_channels, f_channels, kernel_size=1, stride=1)
        self.conv_b = CausalConv3d(zq_channels, f_channels, kernel_size=1, stride=1)

    def __call__(
        self,
        f_ncthw: mx.array,
        zq_ncthw: mx.array,
        conv_cache: dict[str, mx.array | None] | None = None,
    ) -> tuple[mx.array, dict[str, mx.array | None]]:
        conv_cache = conv_cache or {}
        new_cache: dict[str, mx.array | None] = {}

        if f_ncthw.shape[2] > 1 and int(f_ncthw.shape[2]) % 2 == 1:
            f_first = f_ncthw[:, :, :1, :, :]
            f_rest = f_ncthw[:, :, 1:, :, :]
            z_first = zq_ncthw[:, :, :1, :, :]
            z_rest = zq_ncthw[:, :, 1:, :, :]
            _, _, _, hf, wf = f_first.shape
            _, _, _, hr, wr = f_rest.shape
            z_first = _interpolate_nearest_ncthw(z_first, 1, int(hf), int(wf))
            z_rest = _interpolate_nearest_ncthw(z_rest, int(z_rest.shape[2]), int(hr), int(wr))
            zq_ncthw = mx.concatenate([z_first, z_rest], axis=2)
        else:
            tt, th, tw = int(f_ncthw.shape[2]), int(f_ncthw.shape[3]), int(f_ncthw.shape[4])
            zq_ncthw = _interpolate_nearest_ncthw(zq_ncthw, tt, th, tw)

        fy_ndhwc = _ncthw_to_ndhwc(f_ncthw)
        z_ndhwc = _ncthw_to_ndhwc(zq_ncthw)

        cy, new_cache["conv_y"] = self.conv_y(z_ndhwc, conv_cache.get("conv_y"))
        cb, new_cache["conv_b"] = self.conv_b(z_ndhwc, conv_cache.get("conv_b"))

        fy_ncthw = _ndhwc_to_ncthw(fy_ndhwc)
        cy_ncthw = _ndhwc_to_ncthw(cy)
        cb_ncthw = _ndhwc_to_ncthw(cb)

        norm_f = _group_norm_ncthw(self.norm_layer, fy_ncthw)
        out = norm_f * cy_ncthw + cb_ncthw
        return out, new_cache


class GroupNormSpatialFree(nn.Module):
    """Plain ``GroupNorm`` over NCTHW (no spatial conditioning)."""

    def __init__(self, channels: int, groups: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=groups, dims=channels, eps=eps, pytorch_compatible=True)

    def __call__(self, x_ncthw: mx.array) -> mx.array:
        return _group_norm_ncthw(self.norm, x_ncthw)


class ResnetBlock3D(nn.Module):
    """Maps to ``CogVideoXResnetBlock3D``."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int | None,
        dropout: float,
        temb_channels: int,
        groups: int,
        eps: float,
        conv_shortcut: bool,
        spatial_norm_dim: int | None,
        pad_mode: str,
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.spatial_norm_dim = spatial_norm_dim

        if spatial_norm_dim is None:
            self.norm1 = GroupNormSpatialFree(in_channels, groups, eps)
            self.norm2 = GroupNormSpatialFree(out_channels, groups, eps)
        else:
            self.norm1 = SpatialNorm3D(in_channels, spatial_norm_dim, groups)
            self.norm2 = SpatialNorm3D(out_channels, spatial_norm_dim, groups)

        self.conv1 = CausalConv3d(in_channels, out_channels, kernel_size=3, pad_mode=pad_mode)
        self.temb_proj = nn.Linear(temb_channels, out_channels) if temb_channels > 0 else None
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else None
        self.conv2 = CausalConv3d(out_channels, out_channels, kernel_size=3, pad_mode=pad_mode)

        if in_channels != out_channels:
            if conv_shortcut:
                self.conv_shortcut = CausalConv3d(in_channels, out_channels, kernel_size=3, pad_mode=pad_mode)
            else:
                self.conv_shortcut = SafeConv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.conv_shortcut = None

    def __call__(
        self,
        inputs_ncthw: mx.array,
        temb: mx.array | None,
        zq: mx.array | None,
        conv_cache: dict[str, mx.array | None] | None = None,
    ) -> tuple[mx.array, dict[str, mx.array | None]]:
        conv_cache = conv_cache or {}
        new_cache: dict[str, mx.array | None] = {}

        hidden = inputs_ncthw
        if isinstance(self.norm1, SpatialNorm3D):
            hidden, new_cache["norm1"] = self.norm1(hidden, zq, conv_cache.get("norm1"))
        else:
            hidden = self.norm1(hidden)

        hidden = _swish(hidden)
        h_ndhwc = _ncthw_to_ndhwc(hidden)
        h2, c1 = self.conv1(h_ndhwc, conv_cache.get("conv1"))
        hidden = _ndhwc_to_ncthw(h2)
        new_cache["conv1"] = c1

        if temb is not None and self.temb_proj is not None:
            temb_act = _swish(temb)
            temb_spatial = self.temb_proj(temb_act)
            hidden = hidden + mx.expand_dims(mx.expand_dims(mx.expand_dims(temb_spatial, -1), -1), -1)

        if isinstance(self.norm2, SpatialNorm3D):
            hidden, new_cache["norm2"] = self.norm2(hidden, zq, conv_cache.get("norm2"))
        else:
            hidden = self.norm2(hidden)

        hidden = _swish(hidden)
        if self.dropout is not None:
            hidden = self.dropout(hidden)

        h_ndhwc = _ncthw_to_ndhwc(hidden)
        h3, c2 = self.conv2(h_ndhwc, conv_cache.get("conv2"))
        hidden = _ndhwc_to_ncthw(h3)
        new_cache["conv2"] = c2

        if self.in_channels != self.out_channels:
            assert self.conv_shortcut is not None
            if isinstance(self.conv_shortcut, CausalConv3d):
                inp_ndhwc = _ncthw_to_ndhwc(inputs_ncthw)
                sc, csc = self.conv_shortcut(inp_ndhwc, conv_cache.get("conv_shortcut"))
                shortcut = _ndhwc_to_ncthw(sc)
                new_cache["conv_shortcut"] = csc
            else:
                shortcut_ndhwc = self.conv_shortcut(_ncthw_to_ndhwc(inputs_ncthw))
                shortcut = _ndhwc_to_ncthw(shortcut_ndhwc)
        else:
            shortcut = inputs_ncthw

        return hidden + shortcut, new_cache


class MidBlock3D(nn.Module):
    """Maps to ``CogVideoXMidBlock3D``."""

    def __init__(
        self,
        in_channels: int,
        temb_channels: int,
        num_layers: int,
        dropout: float,
        resnet_eps: float,
        resnet_act_fn: str,
        resnet_groups: int,
        spatial_norm_dim: int | None,
        pad_mode: str,
    ):
        super().__init__()
        del resnet_act_fn
        for i in range(num_layers):
            setattr(
                self,
                f"resnet_{i}",
                ResnetBlock3D(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    dropout=dropout,
                    temb_channels=temb_channels,
                    groups=resnet_groups,
                    eps=resnet_eps,
                    conv_shortcut=False,
                    spatial_norm_dim=spatial_norm_dim,
                    pad_mode=pad_mode,
                ),
            )
        self._num_resnet_layers = num_layers

    def __call__(
        self,
        hidden_states_ncthw: mx.array,
        temb: mx.array | None,
        zq: mx.array | None,
        conv_cache: dict[str, mx.array | None] | None = None,
    ) -> tuple[mx.array, dict[str, mx.array | None]]:
        conv_cache = conv_cache or {}
        new_cache: dict[str, mx.array | None] = {}
        h = hidden_states_ncthw
        for i in range(self._num_resnet_layers):
            key = f"resnet_{i}"
            resnet = getattr(self, key)
            h, new_cache[key] = resnet(h, temb, zq, conv_cache.get(key))
        return h, new_cache


class Upsample3D(nn.Module):
    """Maps to ``CogVideoXUpsample3D`` (nearest *2 + Conv2d on NHWC slices)."""

    def __init__(self, in_channels: int, out_channels: int, padding: int, compress_time: bool):
        super().__init__()
        self.compress_time = compress_time
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=padding)

    def __call__(self, inputs_ncthw: mx.array) -> mx.array:
        x = inputs_ncthw
        if self.compress_time:
            if x.shape[2] > 1 and int(x.shape[2]) % 2 == 1:
                x_first = x[:, :, :1, :, :]
                x_rest = x[:, :, 1:, :, :]
                x_first = mx.repeat(mx.repeat(x_first, 2, axis=3), 2, axis=4)
                x_rest = mx.repeat(mx.repeat(mx.repeat(x_rest, 2, axis=2), 2, axis=3), 2, axis=4)
                x = mx.concatenate([x_first, x_rest], axis=2)
            elif x.shape[2] > 1:
                x = mx.repeat(mx.repeat(mx.repeat(x, 2, axis=2), 2, axis=3), 2, axis=4)
            else:
                x = x[:, :, 0, :, :]
                x = mx.repeat(mx.repeat(x, 2, axis=2), 2, axis=3)
                x = x[:, :, None, :, :]
        else:
            b, c, t, h, w = x.shape
            x_btchw = mx.transpose(x, (0, 2, 1, 3, 4))
            x_flat = mx.reshape(x_btchw, (b * t, c, h, w))
            x_flat_ndhwc = mx.transpose(x_flat, (0, 2, 3, 1))
            x_flat_ndhwc = mx.repeat(mx.repeat(x_flat_ndhwc, 2, axis=1), 2, axis=2)
            x_flat = mx.transpose(x_flat_ndhwc, (0, 3, 1, 2))
            x = mx.reshape(x_flat, (b, t, c, h * 2, w * 2))
            x = mx.transpose(x, (0, 2, 1, 3, 4))

        b, c, t, h, w = x.shape
        x_ndhwc = _ncthw_to_ndhwc(x)
        x_flat = mx.reshape(x_ndhwc, (b * t, h, w, c))
        x_flat = self.conv(x_flat)
        x_ndhwc = mx.reshape(x_flat, (b, t, h, w, -1))
        return _ndhwc_to_ncthw(x_ndhwc)


class UpBlock3D(nn.Module):
    """Maps to ``CogVideoXUpBlock3D``."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float,
        num_layers: int,
        resnet_eps: float,
        resnet_act_fn: str,
        resnet_groups: int,
        spatial_norm_dim: int,
        add_upsample: bool,
        upsample_padding: int,
        compress_time: bool,
        pad_mode: str,
    ):
        super().__init__()
        for i in range(num_layers):
            inch = in_channels if i == 0 else out_channels
            setattr(
                self,
                f"resnet_{i}",
                ResnetBlock3D(
                    in_channels=inch,
                    out_channels=out_channels,
                    dropout=dropout,
                    temb_channels=temb_channels,
                    groups=resnet_groups,
                    eps=resnet_eps,
                    conv_shortcut=False,
                    spatial_norm_dim=spatial_norm_dim,
                    pad_mode=pad_mode,
                ),
            )
        self._num_resnet_layers = num_layers
        if add_upsample:
            self.upsampler_0 = Upsample3D(out_channels, out_channels, upsample_padding, compress_time)
        else:
            self.upsampler_0 = None

    def __call__(
        self,
        hidden_states: mx.array,
        temb: mx.array | None,
        zq: mx.array | None,
        conv_cache: dict[str, mx.array | None] | None = None,
    ) -> tuple[mx.array, dict[str, mx.array | None]]:
        conv_cache = conv_cache or {}
        new_cache: dict[str, mx.array | None] = {}
        h = hidden_states
        for i in range(self._num_resnet_layers):
            key = f"resnet_{i}"
            resnet = getattr(self, key)
            h, new_cache[key] = resnet(h, temb, zq, conv_cache.get(key))
        if self.upsampler_0 is not None:
            h = self.upsampler_0(h)
        return h, new_cache


class CogVideoXDecoder3DMlx(nn.Module):
    """MLX clone of ``CogVideoXDecoder3D`` (defaults match diffusers / THUDM CogVideoX-5b)."""

    def __init__(
        self,
        in_channels: int = 16,
        out_channels: int = 3,
        block_out_channels: tuple[int, ...] = (128, 256, 256, 512),
        layers_per_block: int = 3,
        norm_eps: float = 1e-6,
        norm_num_groups: int = 32,
        dropout: float = 0.0,
        pad_mode: str = "first",
        temporal_compression_ratio: float = 4,
    ):
        super().__init__()
        reversed_block_out_channels = list(reversed(block_out_channels))
        self.conv_in = CausalConv3d(
            in_channels, reversed_block_out_channels[0], kernel_size=3, pad_mode=pad_mode
        )
        self.mid_block = MidBlock3D(
            in_channels=reversed_block_out_channels[0],
            temb_channels=0,
            num_layers=2,
            dropout=dropout,
            resnet_eps=norm_eps,
            resnet_act_fn="swish",
            resnet_groups=norm_num_groups,
            spatial_norm_dim=in_channels,
            pad_mode=pad_mode,
        )

        import numpy as _np

        temporal_compress_level = int(round(_np.log2(temporal_compression_ratio)))

        output_channel = reversed_block_out_channels[0]
        up_types = ("CogVideoXUpBlock3D",) * len(block_out_channels)
        for i, _upt in enumerate(up_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1
            compress_time = i < temporal_compress_level
            setattr(
                self,
                f"up_block_{i}",
                UpBlock3D(
                    in_channels=prev_output_channel,
                    out_channels=output_channel,
                    temb_channels=0,
                    dropout=dropout,
                    num_layers=layers_per_block + 1,
                    resnet_eps=norm_eps,
                    resnet_act_fn="swish",
                    resnet_groups=norm_num_groups,
                    spatial_norm_dim=in_channels,
                    add_upsample=not is_final_block,
                    upsample_padding=1,
                    compress_time=compress_time,
                    pad_mode=pad_mode,
                ),
            )
        self._num_up_blocks = len(block_out_channels)

        self.norm_out = SpatialNorm3D(reversed_block_out_channels[-1], in_channels, groups=norm_num_groups)
        self.conv_out = CausalConv3d(
            reversed_block_out_channels[-1], out_channels, kernel_size=3, pad_mode=pad_mode
        )

    def __call__(
        self,
        sample_ncthw: mx.array,
        conv_cache: dict[str, mx.array | None] | None = None,
    ) -> tuple[mx.array, dict[str, mx.array | None]]:
        conv_cache = conv_cache or {}
        new_cache: dict[str, mx.array | None] = {}

        zq = sample_ncthw
        x_ndhwc = _ncthw_to_ndhwc(sample_ncthw)
        h, new_cache["conv_in"] = self.conv_in(x_ndhwc, conv_cache.get("conv_in"))
        hidden = _ndhwc_to_ncthw(h)

        hidden, new_cache["mid_block"] = self.mid_block(hidden, None, zq, conv_cache.get("mid_block"))

        for i in range(self._num_up_blocks):
            key = f"up_block_{i}"
            ub = getattr(self, key)
            hidden, new_cache[key] = ub(hidden, None, zq, conv_cache.get(key))

        hidden, new_cache["norm_out"] = self.norm_out(hidden, zq, conv_cache.get("norm_out"))
        hidden = nn.silu(hidden)
        ho_ndhwc = _ncthw_to_ndhwc(hidden)
        ho2, new_cache["conv_out"] = self.conv_out(ho_ndhwc, conv_cache.get("conv_out"))
        out = _ndhwc_to_ncthw(ho2)
        return out, new_cache


def _assign_decoder_weights(dec: CogVideoXDecoder3DMlx, weights_torch_layout: dict[str, mx.array]) -> None:
    """Assign flattened ``decoder.*`` tensors (PyTorch Conv weight layout) to MLX modules."""

    def set_safe_flat(mod: SafeConv3d, prefix: str) -> None:
        w = weights_torch_layout[f"{prefix}.weight"]
        b = weights_torch_layout[f"{prefix}.bias"]
        mod.conv.weight = _conv3d_weight_torch_to_mlx(w)
        mod.conv.bias = b

    def set_conv3d_causal(mod: CausalConv3d, prefix: str) -> None:
        set_safe_flat(mod.conv, f"{prefix}.conv")

    def set_conv2d(mod: nn.Conv2d, prefix: str) -> None:
        w = weights_torch_layout[f"{prefix}.weight"]
        b = weights_torch_layout[f"{prefix}.bias"]
        mod.weight = _conv2d_weight_torch_to_mlx(w)
        mod.bias = b

    def set_linear(mod: nn.Linear | None, prefix: str) -> None:
        if mod is None:
            return
        mod.weight = weights_torch_layout[f"{prefix}.weight"]
        mod.bias = weights_torch_layout[f"{prefix}.bias"]

    def set_group_norm_components(norm: GroupNormSpatialFree | SpatialNorm3D, prefix: str) -> None:
        if isinstance(norm, GroupNormSpatialFree):
            norm.norm.weight = weights_torch_layout[f"{prefix}.weight"]
            norm.norm.bias = weights_torch_layout[f"{prefix}.bias"]
            return
        nl = f"{prefix}.norm_layer"
        norm.norm_layer.weight = weights_torch_layout[f"{nl}.weight"]
        norm.norm_layer.bias = weights_torch_layout[f"{nl}.bias"]
        set_conv3d_causal(norm.conv_y, f"{prefix}.conv_y")
        set_conv3d_causal(norm.conv_b, f"{prefix}.conv_b")

    def set_resnet(rb: ResnetBlock3D, prefix: str) -> None:
        set_group_norm_components(rb.norm1, f"{prefix}.norm1")
        set_group_norm_components(rb.norm2, f"{prefix}.norm2")
        set_conv3d_causal(rb.conv1, f"{prefix}.conv1")
        set_linear(rb.temb_proj, f"{prefix}.temb_proj")
        set_conv3d_causal(rb.conv2, f"{prefix}.conv2")
        if rb.conv_shortcut is not None:
            if isinstance(rb.conv_shortcut, CausalConv3d):
                set_conv3d_causal(rb.conv_shortcut, f"{prefix}.conv_shortcut")
            else:
                set_safe_flat(rb.conv_shortcut, f"{prefix}.conv_shortcut")

    set_conv3d_causal(dec.conv_in, "conv_in")
    for li in range(2):
        set_resnet(getattr(dec.mid_block, f"resnet_{li}"), f"mid_block.resnets.{li}")

    for bi in range(dec._num_up_blocks):
        ub = getattr(dec, f"up_block_{bi}")
        for ri in range(ub._num_resnet_layers):
            set_resnet(getattr(ub, f"resnet_{ri}"), f"up_blocks.{bi}.resnets.{ri}")
        if ub.upsampler_0 is not None:
            set_conv2d(ub.upsampler_0.conv, f"up_blocks.{bi}.upsamplers.0.conv")

    set_group_norm_components(dec.norm_out, "norm_out")
    set_conv3d_causal(dec.conv_out, "conv_out")


def build_cogvideox_decoder_mlx(vae_cfg: dict[str, Any]) -> CogVideoXDecoder3DMlx:
    dec = CogVideoXDecoder3DMlx(
        in_channels=int(vae_cfg.get("latent_channels", 16)),
        out_channels=int(vae_cfg.get("out_channels", 3)),
        block_out_channels=tuple(vae_cfg.get("block_out_channels", (128, 256, 256, 512))),
        layers_per_block=int(vae_cfg.get("layers_per_block", 3)),
        norm_eps=float(vae_cfg.get("norm_eps", 1e-6)),
        norm_num_groups=int(vae_cfg.get("norm_num_groups", 32)),
        dropout=float(vae_cfg.get("dropout", 0.0)),
        pad_mode="first",
        temporal_compression_ratio=float(vae_cfg.get("temporal_compression_ratio", 4)),
    )
    return dec


def load_vae_bundle_weights(bundle_root: Path) -> tuple[dict[str, mx.array], dict[str, Any]]:
    vae_dir = bundle_root / "vae"
    if not vae_dir.is_dir():
        raise RuntimeError(f"CogVideoX VAE directory missing: {vae_dir}")
    cfg_path = vae_dir / "config.json"
    if not cfg_path.is_file():
        raise RuntimeError(f"CogVideoX VAE config missing: {cfg_path}")
    with open(cfg_path, encoding="utf-8") as f:
        vae_cfg = json.load(f)

    merged: dict[str, mx.array] = {}
    for sf in sorted(vae_dir.glob("*.safetensors")):
        merged.update(dict(mx.load(str(sf))))

    dec_weights: dict[str, mx.array] = {}
    for k, v in merged.items():
        if k.startswith("decoder."):
            dec_weights[k[len("decoder.") :]] = v

    if not dec_weights:
        raise RuntimeError(
            "No tensors with prefix `decoder.` found in VAE safetensors — cannot load CogVideoX decoder."
        )

    return dec_weights, vae_cfg


def decode_latents_ncthw(
    ctx: RuntimeContext,
    latents_bcthw: mx.array,
    bundle_root: Path,
) -> mx.array:
    """Decode ``latents_bcthw`` (Pipeline layout ``[B,C,T,H,W]``) to pixels ``[B,C,T,H,W]`` float."""
    if getattr(ctx, "backend", None) != "mlx":
        raise RuntimeError(
            f"CogVideoX 3D VAE decode is implemented for MLX only; got backend={getattr(ctx, 'backend', None)!r}."
        )

    weights, vae_cfg = load_vae_bundle_weights(bundle_root)
    scaling_factor = float(vae_cfg.get("scaling_factor", 1.15258426))
    shift_factor = vae_cfg.get("shift_factor", None)
    latents = latents_bcthw
    latents = latents / scaling_factor
    if shift_factor is not None:
        latents = latents + float(shift_factor)

    dec = build_cogvideox_decoder_mlx(vae_cfg)
    _assign_decoder_weights(dec, weights)
    mx.eval(dec.parameters())

    # [B,C,T,H,W] — decoder expects NCTHW same as diffusers.
    sample, _ = dec(latents)
    return mx.clip(sample, -1.0, 1.0)
