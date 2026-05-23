"""Wan 2.2 3D causal VAE — decode/encode entry points (``VideoPipeline`` contract)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

from backend.engine.common.vae.decoder import vae_output_to_uint8_hwc

from .vae_mlx import decode_wan_vae_latents, encode_wan_vae_image


def decode_wan_latents_to_pil_frames(
    ctx: Any,
    latents_bcthw: Any,
    bundle_root: Path | None,
    on_stage: Callable[[float], None] | None = None,
    on_log: Callable[[str], None] | None = None,
    *,
    spatial_tiling: bool = False,
    spatial_scale: int = 16,
) -> list[Image.Image]:
    """Decode ``[B,C,T,H,W]`` latents to RGB ``PIL.Image`` frame list."""
    if bundle_root is None:
        raise RuntimeError("Wan VAE decode requires a local model bundle path.")

    if on_log is not None:
        on_log(f"Wan VAE decode start (latent shape {tuple(getattr(latents_bcthw, 'shape', ()))})")

    pixels_ncthw = decode_wan_vae_latents(
        ctx,
        latents_bcthw,
        bundle_root,
        on_stage=on_stage,
        on_log=on_log,
        spatial_tiling=spatial_tiling,
        spatial_scale=spatial_scale,
    )
    if getattr(ctx, "is_tensor", lambda _x: False)(pixels_ncthw):
        ctx.eval(pixels_ncthw)

    arr = np.asarray(pixels_ncthw)
    if arr.ndim != 5:
        raise RuntimeError(f"Wan decoder expected 5D output, got shape {arr.shape}")

    b, c, t, h, w = arr.shape
    if b != 1:
        raise RuntimeError(f"Wan VAE batch size must be 1 for decode, got {b}")

    frames: list[Image.Image] = []
    for ti in range(t):
        frame_chw = arr[0, :, ti, :, :]
        hwc = vae_output_to_uint8_hwc(frame_chw, ctx)
        frames.append(Image.fromarray(hwc))

    if on_log is not None:
        on_log(f"Wan VAE decode done ({len(frames)} frames, {w}x{h})")

    return frames


def encode_wan_image_to_latent(
    ctx: Any,
    image_chw: Any,
    bundle_root: Path | None,
) -> Any:
    """Encode a single RGB frame ``[C,H,W]`` (float ``[-1,1]``) to latents for I2V conditioning."""
    if bundle_root is None:
        raise RuntimeError("Wan VAE encode requires a local model bundle path.")

    latents = encode_wan_vae_image(ctx, image_chw, bundle_root)
    if getattr(ctx, "is_tensor", lambda _x: False)(latents):
        ctx.eval(latents)
    return latents
