"""NHWC spatial ops for Real-ESRGAN RRDBNet (MLX)."""
from __future__ import annotations

import mlx.core as mx


def pixel_shuffle_nhwc(x: mx.array, r: int) -> mx.array:
    n, h, w, c_in = x.shape
    if c_in % (r * r) != 0:
        raise ValueError(f"pixel_shuffle: channels {c_in} not divisible by r*r={r * r}")
    c = c_in // (r * r)
    x = x.reshape((n, h, w, c, r, r))
    x = mx.transpose(x, (0, 1, 4, 2, 5, 3))
    return x.reshape((n, h * r, w * r, c))


def pixel_unshuffle_nhwc(x: mx.array, r: int) -> mx.array:
    n, h, w, c = x.shape
    if h % r != 0 or w % r != 0:
        raise ValueError(f"pixel_unshuffle: H={h}, W={w} not divisible by r={r}")
    oh, ow = h // r, w // r
    x = x.reshape((n, oh, r, ow, r, c))
    x = mx.transpose(x, (0, 1, 3, 5, 2, 4))
    return x.reshape((n, oh, ow, c * r * r))


def interpolate_nearest_nhwc(x: mx.array, scale: int) -> mx.array:
    n, h, w, c = x.shape
    x = x.reshape((n, h, 1, w, 1, c))
    x = mx.broadcast_to(x, (n, h, scale, w, scale, c))
    return x.reshape((n, h * scale, w * scale, c))
