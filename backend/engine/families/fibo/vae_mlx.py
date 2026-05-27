"""FIBO image VAE — thin adapter over Wan 2.2 VAE (same Wan2_2 architecture, ``vae/`` bundle layout)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import mlx.core as mx


def pack_latents(latents: mx.array, height: int, width: int) -> mx.array:
    if latents.ndim == 5:
        latents = latents[:, :, 0, :, :]
    b, c, h, w = latents.shape
    latents = mx.transpose(latents, (0, 2, 3, 1))
    return mx.reshape(latents, (b, h * w, c))


def unpack_latents(latents: mx.array, height: int, width: int) -> mx.array:
    b, seq_len, c = latents.shape
    vae_scale = 16
    lh, lw = height // vae_scale, width // vae_scale
    latents = mx.reshape(latents, (b, lh, lw, c))
    return mx.transpose(latents, (0, 3, 1, 2))


def _wan_scale(vae: Any) -> tuple[mx.array, mx.array]:
    return (vae.scale_mean, vae.scale_inv_std)


def encode_image_n11(ctx: Any, image_n11: mx.array, bundle_root: Path | str) -> mx.array:
    """``image_n11``: [1,3,H,W] in [-1,1] → latents [1,48,h,w]."""
    from backend.engine.families.wan.vae_mlx import load_wan_vae

    vae = load_wan_vae(ctx, Path(bundle_root))
    img = image_n11
    if img.ndim == 4:
        img = img[:, :, None, :, :]
    enc = vae.model.encode(img, _wan_scale(vae))
    if enc.ndim == 5 and enc.shape[2] == 1:
        enc = enc[:, :, 0, :, :]
    ctx.eval(enc)
    return enc


def decode_latents_nchw(ctx: Any, latents: mx.array, bundle_root: Path | str) -> mx.array:
    from backend.engine.families.wan.vae_mlx import load_wan_vae

    vae = load_wan_vae(ctx, Path(bundle_root))
    z = latents
    if z.ndim == 4:
        z = z[:, :, None, :, :]
    out = vae.model.decode(z, _wan_scale(vae))
    if out.ndim == 5:
        out = out[:, :, 0, :, :]
    ctx.eval(out)
    return out
