"""Shared video VAE pixel tensor → PIL frame list conversion."""
from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from backend.engine.common.codecs.vae.decoder import vae_output_to_uint8_hwc


def ncthw_pixels_to_pil_frames(
    ctx: Any,
    pixels_ncthw: Any,
    *,
    model_label: str = "video",
) -> list[Image.Image]:
    """Convert ``[B,C,T,H,W]`` decoded pixels to RGB PIL frames (batch size 1)."""
    if getattr(ctx, "is_tensor", lambda _x: False)(pixels_ncthw):
        ctx.eval(pixels_ncthw)

    arr = np.asarray(pixels_ncthw)
    if arr.ndim != 5:
        raise RuntimeError(f"{model_label} decoder expected 5D output, got shape {arr.shape}")

    b, _c, t, _h, _w = arr.shape
    if b != 1:
        raise RuntimeError(f"{model_label} VAE batch size must be 1 for decode, got {b}")

    frames: list[Image.Image] = []
    for ti in range(t):
        frame_chw = arr[0, :, ti, :, :]
        hwc = vae_output_to_uint8_hwc(frame_chw, ctx)
        frames.append(Image.fromarray(hwc))
    return frames
