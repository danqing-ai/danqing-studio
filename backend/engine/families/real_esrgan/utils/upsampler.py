"""RealESRGANer — tiled full-image inference (NHWC / MLX)."""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
from PIL import Image


class RealESRGANer:
    def __init__(
        self,
        model,
        netscale: int,
        tile: int = 0,
        tile_pad: int = 10,
        pre_pad: int = 10,
    ) -> None:
        self.model = model
        self.scale = netscale
        self.tile_size = tile
        self.tile_pad = tile_pad
        self.pre_pad = pre_pad
        self.mod_scale = {2: 2, 1: 4}.get(netscale)

    def _infer(self, nhwc: np.ndarray) -> np.ndarray:
        y = self.model(mx.array(nhwc.astype(np.float32)))
        mx.eval(y)
        return np.array(y).astype(np.float32)

    def _pre_pad(self, img: np.ndarray) -> tuple[np.ndarray, int, int]:
        if self.pre_pad:
            img = np.pad(
                img,
                ((0, 0), (0, self.pre_pad), (0, self.pre_pad), (0, 0)),
                mode="reflect",
            )
        mph = mpw = 0
        if self.mod_scale is not None:
            h, w = img.shape[1], img.shape[2]
            if h % self.mod_scale:
                mph = self.mod_scale - h % self.mod_scale
            if w % self.mod_scale:
                mpw = self.mod_scale - w % self.mod_scale
            img = np.pad(img, ((0, 0), (0, mph), (0, mpw), (0, 0)), mode="reflect")
        return img, mph, mpw

    def _post_crop(self, out: np.ndarray, mph: int, mpw: int) -> np.ndarray:
        h, w = out.shape[1], out.shape[2]
        if self.mod_scale is not None:
            out = out[:, 0 : h - mph * self.scale, 0 : w - mpw * self.scale, :]
        if self.pre_pad:
            h, w = out.shape[1], out.shape[2]
            out = out[:, 0 : h - self.pre_pad * self.scale, 0 : w - self.pre_pad * self.scale, :]
        return out

    def _tile_process(self, img: np.ndarray) -> np.ndarray:
        _, height, width, channel = img.shape
        s = self.scale
        out = np.zeros((1, height * s, width * s, channel), dtype=np.float32)
        tiles_x = math.ceil(width / self.tile_size)
        tiles_y = math.ceil(height / self.tile_size)

        for y in range(tiles_y):
            for x in range(tiles_x):
                ofs_x, ofs_y = x * self.tile_size, y * self.tile_size
                isx, iex = ofs_x, min(ofs_x + self.tile_size, width)
                isy, iey = ofs_y, min(ofs_y + self.tile_size, height)
                th, tw = iex - isx, iey - isy

                isxp, iexp = max(isx - self.tile_pad, 0), min(iex + self.tile_pad, width)
                isyp, ieyp = max(isy - self.tile_pad, 0), min(iey + self.tile_pad, height)

                tile = img[:, isyp:ieyp, isxp:iexp, :]
                ot = self._infer(tile)

                osxt = (isx - isxp) * s
                osyt = (isy - isyp) * s
                out[:, isy * s : iey * s, isx * s : iex * s, :] = ot[
                    :, osyt : osyt + th * s, osxt : osxt + tw * s, :
                ]
        return out

    def enhance(self, img: np.ndarray, outscale: float | None = None):
        h_in, w_in = img.shape[:2]
        max_range = 65535 if img.dtype == np.uint16 else 255
        arr = img.astype(np.float32) / max_range

        if arr.ndim == 2:
            mode = "L"
            arr = np.repeat(arr[:, :, None], 3, axis=2)
            alpha = None
        elif arr.shape[2] == 4:
            mode = "RGBA"
            alpha = arr[:, :, 3]
            arr = arr[:, :, 0:3]
        else:
            mode = "RGB"
            alpha = None

        nhwc = arr[None, ...]
        padded, mph, mpw = self._pre_pad(nhwc)
        out = self._tile_process(padded) if self.tile_size > 0 else self._infer(padded)
        out = self._post_crop(out, mph, mpw)
        out_rgb = np.clip(out[0], 0.0, 1.0)

        if mode == "RGBA":
            ah, aw = out_rgb.shape[:2]
            a_img = Image.fromarray((np.clip(alpha, 0, 1) * 255).astype(np.uint8), mode="L")
            a_up = np.asarray(a_img.resize((aw, ah), Image.BICUBIC)).astype(np.float32) / 255.0
            out_rgb = np.concatenate([out_rgb, a_up[:, :, None]], axis=2)
        elif mode == "L":
            out_rgb = out_rgb[:, :, 0:1]

        if outscale is not None and float(outscale) != float(self.scale):
            target_w = int(round(w_in * float(outscale)))
            target_h = int(round(h_in * float(outscale)))
            pil_mode = {"L": "L", "RGB": "RGB", "RGBA": "RGBA"}[mode]
            im = Image.fromarray(
                (out_rgb * max_range).astype(np.uint16 if max_range == 65535 else np.uint8).squeeze(),
                mode=pil_mode,
            )
            im = im.resize((target_w, target_h), Image.LANCZOS)
            out_rgb = np.asarray(im).astype(np.float32) / max_range
            if out_rgb.ndim == 2:
                out_rgb = out_rgb[:, :, None]

        out_u = (out_rgb * max_range).round().astype(np.uint16 if max_range == 65535 else np.uint8)
        if mode == "L":
            out_u = out_u[:, :, 0]
        return out_u, mode
