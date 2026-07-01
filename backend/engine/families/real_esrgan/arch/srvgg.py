"""SRVGGNetCompact — MLX port (Real-ESRGAN general / anime video variants)."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from backend.engine.families.real_esrgan.utils.spatial import (
    interpolate_nearest_nhwc,
    pixel_shuffle_nhwc,
)


def _conv(in_ch: int, out_ch: int, k: int = 3) -> nn.Conv2d:
    return nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=1, padding=(k - 1) // 2)


class _PReLU(nn.Module):
    def __init__(self, num_parameters: int = 1, init: float = 0.25) -> None:
        super().__init__()
        self.weight = mx.full((num_parameters,), init)

    def __call__(self, x: mx.array) -> mx.array:
        return mx.maximum(x, 0) + self.weight * mx.minimum(x, 0)


def _make_activation(act_type: str, num_feat: int) -> nn.Module:
    if act_type == "relu":
        return nn.ReLU()
    if act_type == "prelu":
        return _PReLU(num_parameters=num_feat)
    if act_type == "leakyrelu":
        return nn.LeakyReLU(0.1)
    raise ValueError(f"unsupported act_type: {act_type}")


class SRVGGNetCompact(nn.Module):
    """NHWC in/out, pixel values in [0, 1]."""

    def __init__(
        self,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        num_feat: int = 64,
        num_conv: int = 16,
        upscale: int = 4,
        act_type: str = "prelu",
    ) -> None:
        super().__init__()
        self.num_out_ch = num_out_ch
        self.upscale = upscale

        body: list[nn.Module] = []
        body.append(_conv(num_in_ch, num_feat))
        body.append(_make_activation(act_type, num_feat))
        for _ in range(num_conv):
            body.append(_conv(num_feat, num_feat))
            body.append(_make_activation(act_type, num_feat))
        body.append(_conv(num_feat, num_out_ch * upscale * upscale))
        self.body = body

    def __call__(self, x: mx.array) -> mx.array:
        out = x
        for layer in self.body:
            out = layer(out)
        out = pixel_shuffle_nhwc(out, self.upscale)
        base = interpolate_nearest_nhwc(x, self.upscale)
        return out + base
