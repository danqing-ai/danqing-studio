"""Wan 2.2 VAE — diffusers / mflux layout (``quant_conv``, ``down_blocks``, …).

FIBO ``vae/*.safetensors`` use ``FiboVaeWeightMapping`` + this module tree.
Official ``.pth`` + temporal-cache path: ``wan.vae_mlx.WanVAE``.
"""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from backend.engine.common.attention import scaled_dot_product_attention_bhsd_mx
from backend.engine.families.wan.vae_mlx import (
    _WAN22_VAE_MEAN,
    _WAN22_VAE_STD,
    patchify,
    unpatchify,
)

class Wan2_2_RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-12, images: bool = True):
        super().__init__()
        self.eps = eps
        self.scale = float(dim) ** 0.5
        self.images = images
        if images:
            self.weight = mx.ones((dim, 1, 1))
        else:
            self.weight = mx.ones((dim, 1, 1, 1))

    def __call__(self, x: mx.array) -> mx.array:
        sum_sq = mx.sum(x * x, axis=1, keepdims=True)
        l2_norm = mx.sqrt(sum_sq)
        denom = mx.maximum(l2_norm, mx.array(self.eps, dtype=l2_norm.dtype))
        x_normalized = x / denom
        if x.ndim == 5 and not self.images:
            weight = self.weight.reshape(1, -1, 1, 1, 1)
        elif x.ndim == 4 and self.images:
            weight = self.weight.reshape(1, -1, 1, 1)
        else:
            if x.ndim == 5:
                weight = self.weight.reshape(1, -1, 1, 1, 1)
            elif x.ndim == 4:
                weight = self.weight.reshape(1, -1, 1, 1)
            else:
                weight = self.weight
        return x_normalized * self.scale * weight



class Wan2_2_CausalConv3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        name: str | None = None,
    ):
        super().__init__()
        self.conv3d = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
        )
        self.padding = padding
        self.stride = stride
        self.kernel_size = kernel_size
        self.name = name or f"conv3d_{in_channels}to{out_channels}"

    def __call__(self, x: mx.array) -> mx.array:
        pad_t = pad_h = pad_w = self.padding
        if pad_t > 0 or pad_h > 0 or pad_w > 0:
            pad_spec = [
                (0, 0),
                (0, 0),
                (2 * pad_t, 0),
                (pad_h, pad_h),
                (pad_w, pad_w),
            ]
            x = mx.pad(x, pad_spec)
        x = mx.transpose(x, (0, 2, 3, 4, 1))
        x = self.conv3d(x)
        x = mx.transpose(x, (0, 4, 1, 2, 3))
        return x




class Wan2_2_ResidualBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        non_linearity: str = "silu",
    ):
        super().__init__()
        self.norm1 = Wan2_2_RMSNorm(in_dim, images=False)
        self.conv1 = Wan2_2_CausalConv3d(in_dim, out_dim, 3, padding=1)
        self.norm2 = Wan2_2_RMSNorm(out_dim, images=False)
        self.conv2 = Wan2_2_CausalConv3d(out_dim, out_dim, 3, padding=1)

        if in_dim != out_dim:
            self.conv_shortcut = Wan2_2_CausalConv3d(in_dim, out_dim, 1, padding=0)
        else:
            self.conv_shortcut = None

    def __call__(self, x: mx.array, resnet_idx: int | None = None, block_idx: int | None = None) -> mx.array:
        h = self.conv_shortcut(x) if self.conv_shortcut is not None else x
        x = self.norm1(x)
        x = nn.silu(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = nn.silu(x)
        x = self.conv2(x)
        result = x + h
        return result




class Wan2_2_Resample(nn.Module):
    def __init__(self, dim: int, mode: str, upsample_out_dim: int = None):
        super().__init__()
        self.dim = dim
        self.mode = mode

        if upsample_out_dim is None:
            upsample_out_dim = dim // 2

        if mode == "upsample3d":
            self.time_conv = Wan2_2_CausalConv3d(dim, dim * 2, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0))
            self.resample_conv = nn.Conv2d(dim, upsample_out_dim, kernel_size=3, stride=1, padding=1)
        elif mode == "upsample2d":
            self.resample_conv = nn.Conv2d(dim, upsample_out_dim, kernel_size=3, stride=1, padding=1)
            self.time_conv = None
        elif mode == "downsample2d":
            self.resample_conv = nn.Conv2d(dim, dim, kernel_size=3, stride=2, padding=0)
            self.time_conv = None
        elif mode == "downsample3d":
            self.resample_conv = nn.Conv2d(dim, dim, kernel_size=3, stride=2, padding=0)
            self.time_conv = None
        else:
            raise ValueError(f"Unsupported resample mode: {mode}")

    def __call__(self, x: mx.array, block_idx: int | None = None) -> mx.array:
        b, c, t, h, w = x.shape
        if self.mode in ("upsample2d", "upsample3d"):
            if self.mode == "upsample3d" and self.time_conv is not None:
                x = self.time_conv(x)
                x = mx.reshape(x, (b, 2, c, t, h, w))
                x = mx.transpose(x, (0, 2, 3, 1, 4, 5))
                x = mx.reshape(x, (b, c, t * 2, h, w))
                t = t * 2
            x = mx.transpose(x, (0, 2, 1, 3, 4))
            x = mx.reshape(x, (b * t, c, h, w))
            x = mx.transpose(x, (0, 2, 3, 1))
            x = mx.repeat(x, 2, axis=1)
            x = mx.repeat(x, 2, axis=2)
            x = self.resample_conv(x)
            x = mx.transpose(x, (0, 3, 1, 2))
            new_c = x.shape[1]
            new_h, new_w = x.shape[2], x.shape[3]
            x = mx.reshape(x, (b, t, new_c, new_h, new_w))
            x = mx.transpose(x, (0, 2, 1, 3, 4))
            return x

        # downsample modes
        x = mx.transpose(x, (0, 2, 1, 3, 4))
        x = mx.reshape(x, (b * t, c, h, w))
        x = mx.transpose(x, (0, 2, 3, 1))
        x = mx.pad(x, [(0, 0), (0, 1), (0, 1), (0, 0)])
        x = self.resample_conv(x)
        x = mx.transpose(x, (0, 3, 1, 2))
        new_c = x.shape[1]
        new_h, new_w = x.shape[2], x.shape[3]
        x = mx.reshape(x, (b, t, new_c, new_h, new_w))
        x = mx.transpose(x, (0, 2, 1, 3, 4))
        return x

class Wan2_2_AttentionBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.norm = Wan2_2_RMSNorm(dim, images=True)
        self.to_qkv = nn.Conv2d(dim, dim * 3, kernel_size=1)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)

    def __call__(self, x: mx.array) -> mx.array:
        identity = x
        batch_size, channels, time, height, width = x.shape
        x = mx.transpose(x, (0, 2, 1, 3, 4))
        x = mx.reshape(x, (batch_size * time, channels, height, width))

        x = self.norm(x)
        x = mx.transpose(x, (0, 2, 3, 1))

        qkv = self.to_qkv(x)
        qkv = mx.transpose(qkv, (0, 3, 1, 2))
        qkv = mx.reshape(qkv, (batch_size * time, 1, channels * 3, height * width))
        qkv = mx.transpose(qkv, (0, 1, 3, 2))
        q, k, v = mx.split(qkv, 3, axis=3)

        x = scaled_dot_product_attention_bhsd_mx(mx, q, k, v, scale=(channels ** -0.5))
        x = mx.reshape(x, (batch_size * time, height * width, channels))
        x = mx.transpose(x, (0, 2, 1))
        x = mx.reshape(x, (batch_size * time, channels, height, width))
        x = mx.transpose(x, (0, 2, 3, 1))

        x = self.proj(x)
        x = mx.transpose(x, (0, 3, 1, 2))
        x = mx.reshape(x, (batch_size, time, channels, height, width))
        x = mx.transpose(x, (0, 2, 1, 3, 4))
        return x + identity




class Wan2_2_MidBlock(nn.Module):
    def __init__(self, dim: int, non_linearity: str = "silu", num_layers: int = 1):
        super().__init__()
        self.resnets = [Wan2_2_ResidualBlock(dim, dim, non_linearity)]
        self.attentions = []
        for _ in range(num_layers):
            self.attentions.append(Wan2_2_AttentionBlock(dim))
            self.resnets.append(Wan2_2_ResidualBlock(dim, dim, non_linearity))

    def __call__(self, x: mx.array) -> mx.array:
        x = self.resnets[0](x)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            x = attn(x)
            x = resnet(x)
        return x




class Wan2_2_DownBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_res_blocks: int,
        attn_scales: list[float] | None = None,
        scale: float = 1.0,
        temporal_downsample: bool = False,
        non_linearity: str = "silu",
        is_last: bool = False,
    ):
        super().__init__()
        if attn_scales is None:
            attn_scales = []

        resnets: list[nn.Module] = []
        current_dim = in_dim
        for _ in range(num_res_blocks):
            resnets.append(Wan2_2_ResidualBlock(current_dim, out_dim, non_linearity))
            if scale in attn_scales:
                resnets.append(Wan2_2_AttentionBlock(out_dim))
            current_dim = out_dim

        self.resnets = resnets

        # Shortcut path with downsample (mirrors AvgDown3D in diffusers)
        self.avg_shortcut = Wan2_2_AvgDown3D(
            in_dim,
            out_dim,
            factor_t=2 if temporal_downsample else 1,
            factor_s=2 if not is_last else 1,
        )

        # Main path downsampler
        if not is_last:
            mode = "downsample3d" if temporal_downsample else "downsample2d"
            self.downsampler = Wan2_2_Resample(out_dim, mode=mode)
        else:
            self.downsampler = None

    def __call__(self, x: mx.array) -> mx.array:
        x_copy = x
        for layer in self.resnets:
            x = layer(x)
        if self.downsampler is not None:
            x = self.downsampler(x)
        return x + self.avg_shortcut(x_copy)



class Wan2_2_AvgDown3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        factor_t: int,
        factor_s: int = 1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor_t = factor_t
        self.factor_s = factor_s
        self.factor = self.factor_t * self.factor_s * self.factor_s

        assert in_channels * self.factor % out_channels == 0
        self.group_size = in_channels * self.factor // out_channels

    def __call__(self, x: mx.array) -> mx.array:
        pad_t = (self.factor_t - x.shape[2] % self.factor_t) % self.factor_t
        if pad_t > 0:
            x = mx.pad(
                x,
                [
                    (0, 0),  # batch
                    (0, 0),  # channels
                    (pad_t, 0),  # time
                    (0, 0),  # height
                    (0, 0),  # width
                ],
            )

        b, c, t, h, w = x.shape

        x = mx.reshape(
            x,
            (
                b,
                c,
                t // self.factor_t,
                self.factor_t,
                h // self.factor_s,
                self.factor_s,
                w // self.factor_s,
                self.factor_s,
            ),
        )
        x = mx.transpose(x, (0, 1, 3, 5, 7, 2, 4, 6))
        x = mx.reshape(
            x,
            (
                b,
                c * self.factor,
                t // self.factor_t,
                h // self.factor_s,
                w // self.factor_s,
            ),
        )
        x = mx.reshape(
            x,
            (
                b,
                self.out_channels,
                self.group_size,
                t // self.factor_t,
                h // self.factor_s,
                w // self.factor_s,
            ),
        )
        x = mx.mean(x, axis=2)
        return x




class Wan2_2_Encoder3d(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        dim: int = 128,
        z_dim: int = 4,
        dim_mult: list[int] | None = None,
        num_res_blocks: int = 2,
        attn_scales: list[float] | None = None,
        temporal_downsample: list[bool] | None = None,
        non_linearity: str = "silu",
        is_residual: bool = False,
    ):
        super().__init__()

        if dim_mult is None:
            dim_mult = [1, 2, 4, 4]
        if attn_scales is None:
            attn_scales = []
        if temporal_downsample is None:
            temporal_downsample = [False, True, True]

        if is_residual:
            raise NotImplementedError("Residual down blocks are not implemented for Wan2_2_Encoder3d in MLX.")

        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        dims = [dim * u for u in [1] + dim_mult]
        self.temporal_downsample = temporal_downsample

        self.conv_in = Wan2_2_CausalConv3d(in_channels, dims[0], 3, padding=1)
        scale = 1.0

        self.down_blocks: list[Wan2_2_DownBlock] = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            block = Wan2_2_DownBlock(
                in_dim=in_dim,
                out_dim=out_dim,
                num_res_blocks=num_res_blocks,
                attn_scales=attn_scales,
                scale=scale,
                temporal_downsample=temporal_downsample[i] if i < len(temporal_downsample) else False,
                non_linearity=non_linearity,
                is_last=i == len(dim_mult) - 1,
            )
            self.down_blocks.append(block)
            if i != len(dim_mult) - 1:
                scale /= 2.0

        self.mid_block = Wan2_2_MidBlock(out_dim, non_linearity, num_layers=1)

        self.norm_out = Wan2_2_RMSNorm(out_dim, images=False)
        self.conv_out = Wan2_2_CausalConv3d(out_dim, z_dim, 3, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.conv_in(x)
        for block in self.down_blocks:
            x = block(x)
        x = self.mid_block(x)
        x = self.norm_out(x)
        x = nn.silu(x)
        x = self.conv_out(x)
        return x



class Wan2_2_DupUp3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        factor_t: int,
        factor_s: int = 1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor_t = factor_t
        self.factor_s = factor_s
        self.factor = self.factor_t * self.factor_s * self.factor_s
        self.repeats = out_channels * self.factor // in_channels

    def __call__(self, x: mx.array, first_chunk: bool = False) -> mx.array:
        b, c, t, h, w = x.shape
        x = mx.repeat(x, self.repeats, axis=1)
        x = mx.reshape(
            x,
            (
                b,
                self.out_channels,
                self.factor_t,
                self.factor_s,
                self.factor_s,
                t,
                h,
                w,
            ),
        )

        x = mx.transpose(x, (0, 1, 5, 2, 6, 3, 7, 4))
        x = mx.reshape(
            x,
            (
                b,
                self.out_channels,
                t * self.factor_t,
                h * self.factor_s,
                w * self.factor_s,
            ),
        )

        if first_chunk and self.factor_t > 1:
            x = x[:, :, self.factor_t - 1 :, :, :]

        return x




class Wan2_2_ResidualUpBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_res_blocks: int,
        temporal_upsample: bool = False,
        up_flag: bool = False,
        non_linearity: str = "silu",
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        if up_flag:
            self.avg_shortcut: Wan2_2_DupUp3D | None = Wan2_2_DupUp3D(
                in_channels=in_dim,
                out_channels=out_dim,
                factor_t=2 if temporal_upsample else 1,
                factor_s=2,
            )
        else:
            self.avg_shortcut = None

        resnets: list[Wan2_2_ResidualBlock] = []
        current_dim = in_dim
        for _ in range(num_res_blocks + 1):
            resnets.append(Wan2_2_ResidualBlock(current_dim, out_dim, non_linearity))
            current_dim = out_dim
        self.resnets = resnets

        if up_flag:
            upsample_mode = "upsample3d" if temporal_upsample else "upsample2d"
            self.upsampler: Wan2_2_Resample | None = Wan2_2_Resample(
                out_dim, mode=upsample_mode, upsample_out_dim=out_dim
            )
        else:
            self.upsampler = None

    def __call__(self, x: mx.array, block_idx: int | None = None, first_chunk: bool = False) -> mx.array:
        x_copy = x

        for i, resnet in enumerate(self.resnets):
            x = resnet(x, resnet_idx=i, block_idx=block_idx)

        if self.upsampler is not None:
            x = self.upsampler(x, block_idx=block_idx)

        if self.avg_shortcut is not None:
            x = x + self.avg_shortcut(x_copy, first_chunk=first_chunk)

        return x




class Wan2_2_UpBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_res_blocks: int,
        upsample_mode: str | None = None,
        non_linearity: str = "silu",
    ):
        super().__init__()
        self.resnets: list[Wan2_2_ResidualBlock] = []
        current_dim = in_dim
        for _ in range(num_res_blocks + 1):
            self.resnets.append(Wan2_2_ResidualBlock(current_dim, out_dim, non_linearity))
            current_dim = out_dim
        self.upsampler: Wan2_2_Resample | None = None
        if upsample_mode is not None:
            self.upsampler = Wan2_2_Resample(out_dim, mode=upsample_mode, upsample_out_dim=out_dim)

    def __call__(self, x: mx.array, block_idx: int | None = None) -> mx.array:
        for i, resnet in enumerate(self.resnets):
            x = resnet(x, resnet_idx=i, block_idx=block_idx)

        if self.upsampler is not None:
            x = self.upsampler(x, block_idx=block_idx)

        return x




class Wan2_2_Decoder3d(nn.Module):
    def __init__(
        self,
        dim: int = 256,
        z_dim: int = 48,
        dim_mult: list[int] = [1, 2, 4, 4],
        num_res_blocks: int = 2,
        temporal_upsample: list[bool] | None = None,
        non_linearity: str = "silu",
        out_channels: int = 12,
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.temporal_upsample = temporal_upsample or []
        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        self.conv_in = Wan2_2_CausalConv3d(z_dim, dims[0], 3, padding=1, name="decoder_conv_in")
        self.mid_block = Wan2_2_MidBlock(dims[0], non_linearity, num_layers=1)
        self.up_blocks: list[Wan2_2_ResidualUpBlock] = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            up_flag = i != len(dim_mult) - 1
            temporal_flag = (
                bool(self.temporal_upsample and i < len(self.temporal_upsample) and self.temporal_upsample[i])
                if up_flag
                else False
            )
            up_block = Wan2_2_ResidualUpBlock(
                in_dim=in_dim,
                out_dim=out_dim,
                num_res_blocks=num_res_blocks,
                temporal_upsample=temporal_flag,
                up_flag=up_flag,
                non_linearity=non_linearity,
            )
            self.up_blocks.append(up_block)
        self.norm_out = Wan2_2_RMSNorm(out_dim, images=False)
        self.conv_out = Wan2_2_CausalConv3d(out_dim, out_channels, 3, padding=1, name="decoder_conv_out")

    def __call__(self, x: mx.array) -> mx.array:
        x = self.conv_in(x)
        x = self.mid_block(x)
        for i, up_block in enumerate(self.up_blocks):
            x = up_block(x, block_idx=i, first_chunk=True)
        x = self.norm_out(x)
        x = nn.silu(x)
        x = self.conv_out(x)
        return x




class Wan2_2_VAE(nn.Module):
    Z_DIM = 48
    ENCODER_BASE_DIM = 160
    DECODER_BASE_DIM = 256
    DIM_MULT = [1, 2, 4, 4]
    NUM_RES_BLOCKS = 2
    OUT_CHANNELS = 12
    spatial_scale = 16
    latent_channels = 48

    def __init__(self):
        super().__init__()

        self.encoder = Wan2_2_Encoder3d(
            in_channels=12,
            dim=self.ENCODER_BASE_DIM,
            z_dim=self.Z_DIM * 2,
            dim_mult=self.DIM_MULT,
            num_res_blocks=self.NUM_RES_BLOCKS,
            attn_scales=[],
            temporal_downsample=[False, True, True],
        )
        self.quant_conv = Wan2_2_CausalConv3d(self.Z_DIM * 2, self.Z_DIM * 2, 1, padding=0, name="quant_conv")

        self.decoder = Wan2_2_Decoder3d(
            dim=self.DECODER_BASE_DIM,
            z_dim=self.Z_DIM,
            dim_mult=self.DIM_MULT,
            num_res_blocks=self.NUM_RES_BLOCKS,
            temporal_upsample=None,
            out_channels=self.OUT_CHANNELS,
        )
        self.post_quant_conv = Wan2_2_CausalConv3d(self.Z_DIM, self.Z_DIM, 1, padding=0, name="post_quant_conv")

    def encode(self, images: mx.array) -> mx.array:
        if images.ndim == 4:
            x = images.reshape(images.shape[0], images.shape[1], 1, images.shape[2], images.shape[3])
        elif images.ndim == 5:
            x = images
        else:
            raise ValueError(f"Expected 4D or 5D input for VAE.encode, got shape {images.shape}")

        patch_size = 2
        x = patchify(x, patch_size=patch_size)
        h = self.encoder(x)
        h = self.quant_conv(h)
        mean = h[:, : self.Z_DIM, :, :, :]
        latents_mean = mx.array(_WAN22_VAE_MEAN, dtype=mx.float32).reshape(1, self.Z_DIM, 1, 1, 1)
        latents_std = mx.array(_WAN22_VAE_STD, dtype=mx.float32).reshape(1, self.Z_DIM, 1, 1, 1)
        encoded = (mean - latents_mean) / latents_std
        return encoded

    def decode(self, latents: mx.array) -> mx.array:
        if latents.ndim == 4:
            latents = latents.reshape(latents.shape[0], latents.shape[1], 1, latents.shape[2], latents.shape[3])

        latents_mean = mx.array(_WAN22_VAE_MEAN, dtype=mx.float32).reshape(1, self.Z_DIM, 1, 1, 1)
        latents_std = mx.array(_WAN22_VAE_STD, dtype=mx.float32).reshape(1, self.Z_DIM, 1, 1, 1)
        latents = latents * latents_std + latents_mean
        latents = self.post_quant_conv(latents)
        decoded = self.decoder(latents)
        patch_size = 2
        return unpatchify(decoded, patch_size=patch_size)
        return decoded

    @staticmethod
    def _patchify(x: mx.array, patch_size: int) -> mx.array:
        if patch_size == 1:
            return x
        batch_size, channels, frames, height, width = x.shape
        x = mx.reshape(
            x,
            (
                batch_size,
                channels,
                frames,
                height // patch_size,
                patch_size,
                width // patch_size,
                patch_size,
            ),
        )
        x = mx.transpose(x, (0, 1, 6, 4, 2, 3, 5))
        x = mx.reshape(
            x,
            (batch_size, channels * patch_size * patch_size, frames, height // patch_size, width // patch_size),
        )
        return x

    @staticmethod
    def _unpatchify(x: mx.array, patch_size: int) -> mx.array:
        if patch_size == 1:
            return x
        batch_size, c_patches, frames, height, width = x.shape
        channels = c_patches // (patch_size * patch_size)
        x = mx.reshape(x, (batch_size, channels, patch_size, patch_size, frames, height, width))
        x = mx.transpose(x, (0, 1, 4, 5, 3, 6, 2))
        x = mx.reshape(x, (batch_size, channels, frames, height * patch_size, width * patch_size))
