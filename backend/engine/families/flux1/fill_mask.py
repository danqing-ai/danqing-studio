"""FLUX.1 Fill — mask reshape + static packed context (``MaskUtil`` parity)."""
from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

# ``ModelConfig.x_embedder_input_dim`` for Fill: 64 noise + 64 masked VAE + 256 mask pack.
FILL_PATCH_TOKEN_DIM = 384
FILL_STATIC_TOKEN_DIM = 320


def mask_pil_to_weight(mask: Image.Image) -> np.ndarray:
    """White (high) = inpaint region → weight 1.0; returns ``[H, W]`` float32."""
    if mask.mode != "RGB":
        mask = mask.convert("RGB")
    arr = np.asarray(mask, dtype=np.float32) / 255.0
    return (arr[..., 0] > 0.5).astype(np.float32)


def apply_inpaint_mask_rgb(rgb: np.ndarray, mask_hw: np.ndarray) -> np.ndarray:
    """Zero pixels to repaint before VAE encode (``image * (1 - mask)``)."""
    m = mask_hw[..., None] if mask_hw.ndim == 2 else mask_hw[:, :, :1]
    return rgb * (1.0 - m)


def reshape_mask_latent_channels(mask_hw: np.ndarray, height: int, width: int) -> np.ndarray:
    """``[H,W]`` mask → ``[1, 64, H//8, W//8]`` (``MaskUtil.reshape_mask``)."""
    if mask_hw.ndim != 2:
        raise RuntimeError(f"reshape_mask_latent_channels expects 2D mask, got {mask_hw.shape}")
    h, w = int(height), int(width)
    if mask_hw.shape != (h, w):
        raise RuntimeError(
            f"mask shape {mask_hw.shape} does not match image {h}x{w}"
        )
    m = mask_hw.astype(np.float32)
    m = np.reshape(m, (1, h // 8, 8, w // 8, 8))
    m = np.transpose(m, (0, 2, 4, 1, 3))
    return np.reshape(m, (1, 64, h // 8, w // 8)).astype(np.float32)


def build_outpaint_image_and_mask(
    pil: Image.Image,
    directions: list[str],
    pixels: int,
) -> tuple[Image.Image, Image.Image]:
    """Expand canvas; white mask on new border (regenerate), black on original."""
    w, h = pil.size
    pad = {"top": 0, "bottom": 0, "left": 0, "right": 0}
    px = max(64, min(2048, int(pixels)))
    for d in directions:
        if d in pad:
            pad[d] = px
    new_w = w + pad["left"] + pad["right"]
    new_h = h + pad["top"] + pad["bottom"]
    canvas = Image.new("RGB", (new_w, new_h), (0, 0, 0))
    mask = Image.new("RGB", (new_w, new_h), (255, 255, 255))
    canvas.paste(pil, (pad["left"], pad["top"]))
    preserve = Image.new("RGB", (w, h), (0, 0, 0))
    mask.paste(preserve, (pad["left"], pad["top"]))
    return canvas, mask


def create_fill_static_packed(
    ctx: Any,
    *,
    masked_latents_nchw: Any,
    mask_hw: np.ndarray,
    height: int,
    width: int,
    pack_latents_fn: Any,
    pack_mask_latents_fn: Any,
) -> Any:
    """Return ``[1, seq, 320]`` masked VAE pack (64) + mask pack (256)."""
    masked_packed = pack_latents_fn(ctx, masked_latents_nchw)
    mask_spatial = reshape_mask_latent_channels(mask_hw, height, width)
    mask_tensor = ctx.array(mask_spatial)
    mask_packed = pack_mask_latents_fn(ctx, mask_tensor)
    static = ctx.concat([masked_packed, mask_packed], axis=-1)
    expected = int(static.shape[-1])
    if expected != FILL_STATIC_TOKEN_DIM:
        raise RuntimeError(
            f"FLUX Fill static context last dim {expected} (expected {FILL_STATIC_TOKEN_DIM})"
        )
    return static
