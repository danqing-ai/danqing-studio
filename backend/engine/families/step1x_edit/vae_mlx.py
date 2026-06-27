"""Step1X-Edit Flux-style VAE (MLX) — weight keys match ``modules/autoencoder.py``."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from backend.engine.common.ops.attention import scaled_dot_product_attention_bhsd_mx


def _swish(x: mx.array) -> mx.array:
    return x * mx.sigmoid(x)


def _nhwc(x: mx.array) -> mx.array:
    return mx.transpose(x, (0, 2, 3, 1))


def _nchw(x: mx.array) -> mx.array:
    return mx.transpose(x, (0, 3, 1, 2))


class _AttnBlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=32, dims=in_channels, eps=1e-6, pytorch_compatible=True)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def __call__(self, x: mx.array) -> mx.array:
        h = _nhwc(x)
        b, hh, ww, c = h.shape
        normed = self.norm(h.astype(mx.float32)).astype(x.dtype)
        q = self.q(normed).reshape(b, hh * ww, 1, c)
        k = self.k(normed).reshape(b, hh * ww, 1, c)
        v = self.v(normed).reshape(b, hh * ww, 1, c)
        q = mx.transpose(q, (0, 2, 1, 3))
        k = mx.transpose(k, (0, 2, 1, 3))
        v = mx.transpose(v, (0, 2, 1, 3))
        attn = scaled_dot_product_attention_bhsd_mx(
            mx, q, k, v, scale=float(1.0 / mx.sqrt(q.shape[-1].astype(mx.float32)))
        )
        attn = mx.transpose(attn, (0, 2, 1, 3)).reshape(b, hh, ww, c)
        out = self.proj_out(attn)
        return _nchw(h + out)


class _ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int | None = None):
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm1 = nn.GroupNorm(num_groups=32, dims=in_channels, eps=1e-6, pytorch_compatible=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=32, dims=out_channels, eps=1e-6, pytorch_compatible=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.nin_shortcut = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
            if in_channels != out_channels
            else None
        )

    def __call__(self, x: mx.array) -> mx.array:
        residual = x
        h = self.norm1(_nhwc(x).astype(mx.float32)).astype(x.dtype)
        h = _swish(h)
        h = self.conv1(h)
        h = self.norm2(h.astype(mx.float32)).astype(x.dtype)
        h = _swish(h)
        h = self.conv2(h)
        h = _nchw(h)
        if self.nin_shortcut is not None:
            residual = _nchw(self.nin_shortcut(_nhwc(x)))
        return h + residual


class _Downsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def __call__(self, x: mx.array) -> mx.array:
        h = _nhwc(x)
        h = mx.pad(h, [(0, 0), (0, 1), (0, 1), (0, 0)])
        h = self.conv(h)
        return _nchw(h)


class _Upsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        h = _nchw(x)
        h = mx.repeat(h, 2, axis=2)
        h = mx.repeat(h, 2, axis=3)
        h = self.conv(_nhwc(h))
        return _nchw(h)


class _Encoder(nn.Module):
    def __init__(
        self,
        resolution: int,
        in_channels: int,
        ch: int,
        ch_mult: list[int],
        num_res_blocks: int,
        z_channels: int,
    ):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, ch, kernel_size=3, stride=1, padding=1)
        in_ch_mult = (1, *tuple(ch_mult))
        self.down = []
        block_in = ch
        for i_level in range(len(ch_mult)):
            block = []
            attn = []
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for _ in range(num_res_blocks):
                block.append(_ResnetBlock(block_in, block_out))
                block_in = block_out
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != len(ch_mult) - 1:
                down.downsample = _Downsample(block_in)
            self.down.append(down)
        self.mid = nn.Module()
        self.mid.block_1 = _ResnetBlock(block_in, block_in)
        self.mid.attn_1 = _AttnBlock(block_in)
        self.mid.block_2 = _ResnetBlock(block_in, block_in)
        self.norm_out = nn.GroupNorm(num_groups=32, dims=block_in, eps=1e-6, pytorch_compatible=True)
        self.conv_out = nn.Conv2d(block_in, 2 * z_channels, kernel_size=3, stride=1, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        hs = [_nchw(self.conv_in(_nhwc(x)))]
        for level in self.down:
            for res in level.block:
                hs.append(res(hs[-1]))
            if hasattr(level, "downsample"):
                hs.append(level.downsample(hs[-1]))
        h = hs[-1]
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        h = self.norm_out(_nhwc(h).astype(mx.float32)).astype(x.dtype)
        h = _swish(h)
        h = self.conv_out(h)
        return _nchw(h)


class _Decoder(nn.Module):
    def __init__(
        self,
        ch: int,
        out_ch: int,
        ch_mult: list[int],
        num_res_blocks: int,
        z_channels: int,
    ):
        super().__init__()
        block_in = ch * ch_mult[-1]
        self.conv_in = nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)
        self.mid = nn.Module()
        self.mid.block_1 = _ResnetBlock(block_in, block_in)
        self.mid.attn_1 = _AttnBlock(block_in)
        self.mid.block_2 = _ResnetBlock(block_in, block_in)
        self.up = []
        for i_level in reversed(range(len(ch_mult))):
            block = []
            attn = []
            block_out = ch * ch_mult[i_level]
            for _ in range(num_res_blocks + 1):
                block.append(_ResnetBlock(block_in, block_out))
                block_in = block_out
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = _Upsample(block_in)
            self.up.insert(0, up)
        self.norm_out = nn.GroupNorm(num_groups=32, dims=block_in, eps=1e-6, pytorch_compatible=True)
        self.conv_out = nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def __call__(self, z: mx.array) -> mx.array:
        h = _nchw(self.conv_in(_nhwc(z)))
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        for level in reversed(self.up):
            for res in level.block:
                h = res(h)
            if hasattr(level, "upsample"):
                h = level.upsample(h)
        h = self.norm_out(_nhwc(h).astype(mx.float32)).astype(z.dtype)
        h = _swish(h)
        h = self.conv_out(h)
        return _nchw(h)


class Step1XAutoEncoderMLX(nn.Module):
    """Keys: ``encoder.*``, ``decoder.*`` (flat safetensors from StepFun bundle)."""

    def __init__(
        self,
        *,
        resolution: int = 256,
        in_channels: int = 3,
        ch: int = 128,
        out_ch: int = 3,
        ch_mult: list[int] | None = None,
        num_res_blocks: int = 2,
        z_channels: int = 16,
        scale_factor: float = 0.3611,
        shift_factor: float = 0.1159,
    ):
        super().__init__()
        ch_mult = ch_mult or [1, 2, 4, 4]
        self.encoder = _Encoder(
            resolution=resolution,
            in_channels=in_channels,
            ch=ch,
            ch_mult=ch_mult,
            num_res_blocks=num_res_blocks,
            z_channels=z_channels,
        )
        self.decoder = _Decoder(
            ch=ch,
            out_ch=out_ch,
            ch_mult=ch_mult,
            num_res_blocks=num_res_blocks,
            z_channels=z_channels,
        )
        self.scale_factor = scale_factor
        self.shift_factor = shift_factor

    def encode(self, x: mx.array) -> mx.array:
        z = self.encoder(x)
        mean, _logvar = mx.split(z, 2, axis=1)
        z = self.scale_factor * (mean - self.shift_factor)
        return z

    def decode(self, z: mx.array) -> mx.array:
        z = z / self.scale_factor + self.shift_factor
        return self.decoder(z)
