"""Qwen-Image-Edit 尺寸与 VAE 条件 latent（对齐 mflux ``QwenEditUtil`` / ``_compute_dimensions``）。"""
from __future__ import annotations

import math
from typing import Any

from PIL import Image


def compute_qwen_edit_dimensions(
    source: Image.Image,
    *,
    width: int | None = None,
    height: int | None = None,
) -> tuple[int, int, int, int, int, int]:
    """返回 ``(out_w, out_h, vl_w, vl_h, vae_w, vae_h)``，均为 32 对齐。"""
    image_size = source.size
    ratio = image_size[0] / max(image_size[1], 1)
    target_area = 1024 * 1024
    calculated_width = math.sqrt(target_area * ratio)
    calculated_height = calculated_width / ratio
    calculated_width = round(calculated_width / 32) * 32
    calculated_height = round(calculated_height / 32) * 32

    use_height = int(height or calculated_height)
    use_width = int(width or calculated_width)

    multiple_of = 16
    use_width = max(multiple_of, use_width // multiple_of * multiple_of)
    use_height = max(multiple_of, use_height // multiple_of * multiple_of)

    condition_area = 384 * 384
    condition_ratio = image_size[0] / max(image_size[1], 1)
    vl_width = round(math.sqrt(condition_area * condition_ratio) / 32) * 32
    vl_height = round((vl_width / condition_ratio) / 32) * 32

    vae_area = 1024 * 1024
    vae_ratio = image_size[0] / max(image_size[1], 1)
    vae_width = round(math.sqrt(vae_area * vae_ratio) / 32) * 32
    vae_height = round((vae_width / vae_ratio) / 32) * 32

    return (
        int(use_width),
        int(use_height),
        int(vl_width),
        int(vl_height),
        int(vae_width),
        int(vae_height),
    )


def pack_qwen_latents_to_sequence(ctx: Any, latents_nchw: Any) -> Any:
    """``[B,64,H,W]`` → ``[B, H*W, 64]``。"""
    b = int(latents_nchw.shape[0])
    c = int(latents_nchw.shape[1])
    h = int(latents_nchw.shape[2])
    w = int(latents_nchw.shape[3])
    x = ctx.permute(latents_nchw, (0, 2, 3, 1))
    return ctx.reshape(x, (b, h * w, c))


def unpack_qwen_sequence_to_nchw(ctx: Any, seq_bsc: Any, height_px: int, width_px: int) -> Any:
    """``[B, seq, 64]`` → ``[B,64,H/16,W/16]``。"""
    b = int(seq_bsc.shape[0])
    h_lat = height_px // 16
    w_lat = width_px // 16
    x = ctx.reshape(seq_bsc, (b, h_lat, w_lat, 64))
    return ctx.permute(x, (0, 3, 1, 2))


def create_qwen_edit_conditioning_latents(
    ctx: Any,
    *,
    vae_encode_fn,
    source: Image.Image,
    vae_width: int,
    vae_height: int,
    on_log: Any | None = None,
) -> tuple[Any, tuple[int, int, int]]:
    """VAE 编码参考图并 pack；返回 ``(packed_nchw, cond_image_grid)``。"""
    from backend.engine.vae_codec_registry import qwen_pack_latents_nchw

    src = source.convert("RGB").resize((vae_width, vae_height), Image.BICUBIC)
    import numpy as np

    arr = np.asarray(src, dtype=np.float32) / 255.0
    arr = arr * 2.0 - 1.0
    img_n11 = ctx.array(arr.transpose(2, 0, 1)[None, ...])

    encoded = vae_encode_fn(
        img_n11,
        height_px=vae_height,
        width_px=vae_width,
    )
    packed = qwen_pack_latents_nchw(ctx, encoded, vae_height, vae_width)
    if getattr(ctx, "backend", None) == "mlx":
        ctx.eval(packed)
    cond_h = vae_height // 16
    cond_w = vae_width // 16
    if on_log:
        on_log(
            "info",
            f"qwen_edit conditioning vae={vae_width}x{vae_height} grid=1x{cond_h}x{cond_w} "
            f"packed={tuple(packed.shape)}",
        )
    return packed, (1, cond_h, cond_w)
