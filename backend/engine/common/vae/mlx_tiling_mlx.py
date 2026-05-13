from __future__ import annotations

"""MLX 空间域 VAE 分块编解码（任意 ``nn.Module`` 提供 ``encode`` / ``decode``）。

与 ``ImagePipeline`` 的通用 2D VAE 编码器路径独立；适用于 5D latent（B,C,T,H,W）及
单帧 ``T=1`` 的 3D VAE。重叠融合为余弦 ramp，与具体 DiT 族无关。
"""

from dataclasses import dataclass
from typing import Callable

import mlx.core as mx
import numpy as np
from mlx import nn


@dataclass(frozen=True, slots=True)
class TilingConfig:
    vae_decode_tiles_per_dim: int | None = 8
    vae_decode_overlap: int = 8
    vae_encode_tiled: bool = True
    vae_encode_tile_size: int = 512
    vae_encode_tile_overlap: int = 64


class VAETiler:
    @staticmethod
    def encode_image_tiled(
        *,
        image: mx.array,
        encode_fn: Callable[[mx.array], mx.array],
        latent_channels: int,
        tile_size: tuple[int, int] = (512, 512),
        tile_overlap: tuple[int, int] = (64, 64),
        spatial_scale: int = 8,
    ) -> mx.array:
        B, _, H, W = image.shape

        if B != 1:
            tile_5d = image[:, :, None, :, :]
            return encode_fn(tile_5d)

        tile_h, tile_w = tile_size
        overlap_h, overlap_w = tile_overlap

        if H <= tile_h and W <= tile_w:
            tile_5d = image[:, :, None, :, :]
            return encode_fn(tile_5d)

        scale = int(spatial_scale)
        latent_tile_h = max(1, tile_h // scale)
        latent_tile_w = max(1, tile_w // scale)
        latent_overlap_h = max(0, min(overlap_h // scale, latent_tile_h - 1))
        latent_overlap_w = max(0, min(overlap_w // scale, latent_tile_w - 1))

        stride_h = max(1, latent_tile_h - latent_overlap_h)
        stride_w = max(1, latent_tile_w - latent_overlap_w)

        H_lat_total = (H + scale - 1) // scale
        W_lat_total = (W + scale - 1) // scale

        ramp_h = VAETiler._cos_ramp(latent_overlap_h)
        ramp_w = VAETiler._cos_ramp(latent_overlap_w)

        out_np = np.zeros((H_lat_total, W_lat_total, latent_channels), dtype=np.float32)
        count_np = np.zeros((H_lat_total, W_lat_total, 1), dtype=np.float32)

        for y_lat in range(0, H_lat_total, stride_h):
            y_lat_end = min(y_lat + latent_tile_h, H_lat_total)
            for x_lat in range(0, W_lat_total, stride_w):
                x_lat_end = min(x_lat + latent_tile_w, W_lat_total)

                if (y_lat > 0 and (y_lat_end - y_lat) <= latent_overlap_h) or (
                    x_lat > 0 and (x_lat_end - x_lat) <= latent_overlap_w
                ):
                    continue

                y_out = y_lat * scale
                x_out = x_lat * scale
                y_out_end = min(y_lat_end * scale, H)
                x_out_end = min(x_lat_end * scale, W)

                tile_sample = image[:, :, y_out:y_out_end, x_out:x_out_end]
                tile_5d = tile_sample[:, :, None, :, :]
                enc = encode_fn(tile_5d)

                if enc.shape[2] == 1:
                    enc = enc[:, :, 0, :, :]

                enc_np = np.array(enc.astype(mx.float32))[0].transpose(1, 2, 0)

                eff_h_lat = min(y_lat_end - y_lat, enc_np.shape[0], H_lat_total - y_lat)
                eff_w_lat = min(x_lat_end - x_lat, enc_np.shape[1], W_lat_total - x_lat)
                enc_np = enc_np[:eff_h_lat, :eff_w_lat, :]

                ov_h = max(0, min(latent_overlap_h, eff_h_lat - 1))
                ov_w = max(0, min(latent_overlap_w, eff_w_lat - 1))

                wh = np.ones((eff_h_lat,), dtype=np.float32)
                ww = np.ones((eff_w_lat,), dtype=np.float32)

                if ov_h > 0:
                    if y_lat > 0:
                        wh[:ov_h] = ramp_h[:ov_h]
                    if y_lat_end < H_lat_total:
                        wh[-ov_h:] = 1.0 - ramp_h[:ov_h]
                if ov_w > 0:
                    if x_lat > 0:
                        ww[:ov_w] = ramp_w[:ov_w]
                    if x_lat_end < W_lat_total:
                        ww[-ov_w:] = 1.0 - ramp_w[:ov_w]

                w2d = wh[:, None] * ww[None, :]
                out_np[y_lat : y_lat + eff_h_lat, x_lat : x_lat + eff_w_lat, :] += enc_np * w2d[:, :, None]
                count_np[y_lat : y_lat + eff_h_lat, x_lat : x_lat + eff_w_lat, :] += w2d[:, :, None]

        out_np = out_np / np.clip(count_np, 1e-6, None)
        out_chw = out_np.transpose(2, 0, 1)
        return mx.array(out_chw[None, :, None, :, :])

    @staticmethod
    def decode_image_tiled(
        *,
        latent: mx.array,
        decode_fn: Callable[[mx.array], mx.array],
        tile_size: tuple[int, int] = (512, 512),
        tile_overlap: tuple[int, int] = (64, 64),
        spatial_scale: int = 8,
    ) -> mx.array:
        B, _, T, H_lat, W_lat = latent.shape
        if B != 1 or T != 1:
            decoded = decode_fn(latent)
            if decoded.shape[2] == 1:
                decoded = decoded[:, :, 0, :, :]
            return decoded

        scale = int(spatial_scale)
        H_out = H_lat * scale
        W_out = W_lat * scale

        tile_h, tile_w = tile_size
        overlap_h, overlap_w = tile_overlap

        latent_tile_h = max(1, tile_h // scale)
        latent_tile_w = max(1, tile_w // scale)

        if H_lat <= latent_tile_h and W_lat <= latent_tile_w:
            decoded = decode_fn(latent)
            if decoded.shape[2] == 1:
                decoded = decoded[:, :, 0, :, :]
            return decoded

        latent_overlap_h = max(0, min(overlap_h // scale, latent_tile_h - 1))
        latent_overlap_w = max(0, min(overlap_w // scale, latent_tile_w - 1))

        stride_h = max(1, latent_tile_h - latent_overlap_h)
        stride_w = max(1, latent_tile_w - latent_overlap_w)

        ramp_h = VAETiler._cos_ramp(overlap_h)
        ramp_w = VAETiler._cos_ramp(overlap_w)

        out_np: np.ndarray | None = None
        count_np: np.ndarray | None = None

        for y_lat in range(0, H_lat, stride_h):
            y_lat_end = min(y_lat + latent_tile_h, H_lat)
            for x_lat in range(0, W_lat, stride_w):
                x_lat_end = min(x_lat + latent_tile_w, W_lat)

                if (y_lat > 0 and (y_lat_end - y_lat) <= latent_overlap_h) or (
                    x_lat > 0 and (x_lat_end - x_lat) <= latent_overlap_w
                ):
                    continue

                tile_latent = latent[:, :, :, y_lat:y_lat_end, x_lat:x_lat_end]
                decoded_tile = decode_fn(tile_latent)
                if decoded_tile.shape[2] == 1:
                    decoded_tile = decoded_tile[:, :, 0, :, :]

                tile_np = np.array(decoded_tile.astype(mx.float32))[0].transpose(1, 2, 0)

                y_out = y_lat * scale
                x_out = x_lat * scale
                y_out_end = y_lat_end * scale
                x_out_end = x_lat_end * scale

                h_out = y_out_end - y_out
                w_out = x_out_end - x_out

                eff_h = min(h_out, tile_np.shape[0], H_out - y_out)
                eff_w = min(w_out, tile_np.shape[1], W_out - x_out)
                tile_np = tile_np[:eff_h, :eff_w, :]

                if out_np is None:
                    out_np = np.zeros((H_out, W_out, tile_np.shape[2]), dtype=np.float32)
                    count_np = np.zeros((H_out, W_out, 1), dtype=np.float32)

                ov_h_out = max(0, min(overlap_h, eff_h - 1))
                ov_w_out = max(0, min(overlap_w, eff_w - 1))

                wh = np.ones((eff_h,), dtype=np.float32)
                ww = np.ones((eff_w,), dtype=np.float32)

                if ov_h_out > 0:
                    if y_lat > 0:
                        wh[:ov_h_out] = ramp_h[:ov_h_out]
                    if y_lat_end < H_lat:
                        wh[-ov_h_out:] = 1.0 - ramp_h[:ov_h_out]
                if ov_w_out > 0:
                    if x_lat > 0:
                        ww[:ov_w_out] = ramp_w[:ov_w_out]
                    if x_lat_end < W_lat:
                        ww[-ov_w_out:] = 1.0 - ramp_w[:ov_w_out]

                w2d = wh[:, None] * ww[None, :]
                out_np[y_out : y_out + eff_h, x_out : x_out + eff_w, :] += tile_np * w2d[:, :, None]
                count_np[y_out : y_out + eff_h, x_out : x_out + eff_w, :] += w2d[:, :, None]

        assert out_np is not None
        assert count_np is not None
        out_np = out_np / np.clip(count_np, 1e-6, None)
        out_chw = out_np.transpose(2, 0, 1)
        return mx.array(out_chw[None, ...])

    @staticmethod
    def _cos_ramp(n: int) -> np.ndarray:
        if n <= 0:
            return np.zeros((0,), dtype=np.float32)
        t = np.linspace(0.0, 1.0, num=n, dtype=np.float32)
        return 0.5 - 0.5 * np.cos(t * np.pi)


class VAEUtil:
    @staticmethod
    def encode(
        vae: nn.Module,
        image: mx.array,
        tiling_config: TilingConfig | None = None,
    ) -> mx.array:
        if tiling_config is not None and tiling_config.vae_encode_tiled:
            return VAETiler.encode_image_tiled(
                image=image,
                encode_fn=vae.encode,
                latent_channels=getattr(vae, "latent_channels", 16),
                tile_size=(tiling_config.vae_encode_tile_size, tiling_config.vae_encode_tile_size),
                tile_overlap=(tiling_config.vae_encode_tile_overlap, tiling_config.vae_encode_tile_overlap),
                spatial_scale=getattr(vae, "spatial_scale", 8),
            )

        encoded = vae.encode(image)

        if encoded.ndim == 5 and encoded.shape[2] == 1:
            encoded = encoded[:, :, 0, :, :]

        return encoded

    @staticmethod
    def decode(
        vae: nn.Module,
        latent: mx.array,
        tiling_config: TilingConfig | None = None,
    ) -> mx.array:
        if (
            tiling_config is not None
            and tiling_config.vae_decode_tiles_per_dim
            and tiling_config.vae_decode_tiles_per_dim > 1
        ):
            if latent.ndim == 4:
                latent = latent[:, :, None, :, :]

            spatial_scale = getattr(vae, "spatial_scale", 8)
            overlap_px = int(tiling_config.vae_decode_overlap) * spatial_scale
            return VAETiler.decode_image_tiled(
                latent=latent,
                decode_fn=vae.decode,
                tile_size=(512, 512),
                tile_overlap=(overlap_px, overlap_px),
                spatial_scale=spatial_scale,
            )

        decoded = vae.decode(latent)

        if decoded.ndim == 5 and decoded.shape[2] == 1:
            decoded = decoded[:, :, 0, :, :]

        return decoded
