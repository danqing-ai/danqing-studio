"""FIBO Wan2.2 VAE — FIBO weight map + mflux-layout ``wan.vae_diffusers_mlx``."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import mlx.core as mx

from backend.engine.families.fibo.vae_weights import (
    FiboVaeWeightMapping,
    _FiboVaeWeightDefinition,
)
from backend.engine.families.wan.vae_diffusers_mlx import Wan2_2_VAE

_vae_cache: dict[str, Wan2_2_VAE] = {}


def _load_vae_model(bundle_root: str) -> Wan2_2_VAE:
    key = str(Path(bundle_root).resolve())
    cached = _vae_cache.get(key)
    if cached is not None:
        return cached
    from backend.engine.common.bundle_weights.loader_mlx import WeightLoader

    bundle = Path(bundle_root).parent
    loaded = WeightLoader.load(_FiboVaeWeightDefinition, model_path=str(bundle))
    model = Wan2_2_VAE()
    model.update(loaded.components["vae"], strict=False)
    _vae_cache[key] = model
    return model


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
    latents = mx.transpose(latents, (0, 3, 1, 2))
    return latents


def encode_image_n11(ctx: Any, image_n11: mx.array, bundle_root: Path | str) -> mx.array:
    """``image_n11``: [1,3,H,W] in [-1,1] → latents [1,48,h,w]."""
    vae = _load_vae_model(str(Path(bundle_root) / "vae"))
    img = image_n11
    if img.ndim == 4:
        img = img[:, :, None, :, :]
    enc = vae.encode(img)
    if enc.ndim == 5 and enc.shape[2] == 1:
        enc = enc[:, :, 0, :, :]
    ctx.eval(enc)
    return enc


def decode_latents_nchw(ctx: Any, latents: mx.array, bundle_root: Path | str) -> mx.array:
    vae = _load_vae_model(str(Path(bundle_root) / "vae"))
    z = latents
    if z.ndim == 4:
        z = z[:, :, None, :, :]
    out = vae.decode(z)
    if out.ndim == 5:
        out = out[:, :, 0, :, :]
    ctx.eval(out)
    return out


__all__ = [
    "FiboVaeWeightMapping",
    "decode_latents_nchw",
    "encode_image_n11",
    "pack_latents",
    "unpack_latents",
]
