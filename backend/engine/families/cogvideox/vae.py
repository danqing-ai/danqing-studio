"""CogVideoX 3D causal VAE — 解码入口（潜变量 → PIL 帧，与 ``VideoPipeline`` 约定一致）。"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

from backend.engine.common.vae.decoder import vae_output_to_uint8_hwc

from .vae_mlx import decode_latents_ncthw


def decode_cogvideox_latents_to_pil_frames(
    ctx: Any,
    latents_bcthw: Any,
    bundle_root: Path | None,
    on_stage: Callable[[float], None] | None = None,
    on_log: Callable[[str], None] | None = None,
    frame_batch_size: int = 2,
) -> list[Image.Image]:
    """将 ``[B,C,T,H,W]`` 潜变量解码为 RGB ``PIL.Image`` 帧列表。"""
    if bundle_root is None:
        raise RuntimeError("CogVideoX VAE decode requires a local model bundle path.")

    pixels_ncthw = decode_latents_ncthw(
        ctx, latents_bcthw, bundle_root,
        on_stage=on_stage, on_log=on_log, frame_batch_size=frame_batch_size,
    )
    if getattr(ctx, "is_tensor", lambda _x: False)(pixels_ncthw):
        ctx.eval(pixels_ncthw)

    arr = np.asarray(pixels_ncthw)
    if arr.ndim != 5:
        raise RuntimeError(f"CogVideoX decoder expected 5D output, got shape {arr.shape}")

    b, c, t, h, w = arr.shape
    if b != 1:
        raise RuntimeError(f"CogVideoX VAE batch size must be 1 for decode, got {b}")

    frames: list[Image.Image] = []
    for ti in range(t):
        frame_chw = arr[0, :, ti, :, :]
        hwc = vae_output_to_uint8_hwc(frame_chw, ctx)
        frames.append(Image.fromarray(hwc))

    return frames
