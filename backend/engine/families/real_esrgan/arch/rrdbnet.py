"""RRDBNet — MLX port (Real-ESRGAN x4plus / x2plus / anime variants)."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from backend.engine.families.real_esrgan.utils.spatial import (
    interpolate_nearest_nhwc,
    pixel_unshuffle_nhwc,
)


def _conv(in_ch: int, out_ch: int, k: int = 3) -> nn.Conv2d:
    return nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=1, padding=(k - 1) // 2)


def _lrelu(x: mx.array) -> mx.array:
    return nn.leaky_relu(x, 0.2)


class ResidualDenseBlock(nn.Module):
    def __init__(self, num_feat: int = 64, num_grow_ch: int = 32) -> None:
        super().__init__()
        self.conv1 = _conv(num_feat, num_grow_ch)
        self.conv2 = _conv(num_feat + num_grow_ch, num_grow_ch)
        self.conv3 = _conv(num_feat + 2 * num_grow_ch, num_grow_ch)
        self.conv4 = _conv(num_feat + 3 * num_grow_ch, num_grow_ch)
        self.conv5 = _conv(num_feat + 4 * num_grow_ch, num_feat)

    def __call__(self, x: mx.array) -> mx.array:
        x1 = _lrelu(self.conv1(x))
        x2 = _lrelu(self.conv2(mx.concatenate([x, x1], axis=-1)))
        x3 = _lrelu(self.conv3(mx.concatenate([x, x1, x2], axis=-1)))
        x4 = _lrelu(self.conv4(mx.concatenate([x, x1, x2, x3], axis=-1)))
        x5 = self.conv5(mx.concatenate([x, x1, x2, x3, x4], axis=-1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, num_feat: int, num_grow_ch: int = 32) -> None:
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def __call__(self, x: mx.array) -> mx.array:
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    """NHWC in/out, pixel values in [0, 1]."""

    def __init__(
        self,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        scale: int = 4,
        num_feat: int = 64,
        num_block: int = 23,
        num_grow_ch: int = 32,
    ) -> None:
        super().__init__()
        self.scale = scale
        if scale == 2:
            num_in_ch = num_in_ch * 4
        elif scale == 1:
            num_in_ch = num_in_ch * 16

        self.conv_first = _conv(num_in_ch, num_feat)
        self.body = [RRDB(num_feat, num_grow_ch) for _ in range(num_block)]
        self.conv_body = _conv(num_feat, num_feat)
        self.conv_up1 = _conv(num_feat, num_feat)
        self.conv_up2 = _conv(num_feat, num_feat)
        self.conv_hr = _conv(num_feat, num_feat)
        self.conv_last = _conv(num_feat, num_out_ch)

    def __call__(self, x: mx.array) -> mx.array:
        if self.scale == 2:
            feat = pixel_unshuffle_nhwc(x, r=2)
        elif self.scale == 1:
            feat = pixel_unshuffle_nhwc(x, r=4)
        else:
            feat = x

        feat = self.conv_first(feat)
        body_feat = feat
        for blk in self.body:
            body_feat = blk(body_feat)
        body_feat = self.conv_body(body_feat)
        feat = feat + body_feat

        feat = _lrelu(self.conv_up1(interpolate_nearest_nhwc(feat, 2)))
        feat = _lrelu(self.conv_up2(interpolate_nearest_nhwc(feat, 2)))
        out = self.conv_last(_lrelu(self.conv_hr(feat)))
        return out
