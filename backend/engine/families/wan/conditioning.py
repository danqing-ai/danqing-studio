"""Wan TI2V / I2V conditioning helpers (ported from Wan2.2 reference utils)."""
from __future__ import annotations

import math
from typing import Any

from PIL import Image

from backend.engine.runtime._base import RuntimeContext

# Official ``wan/configs/shared_config.py`` — used when the user leaves negative prompt empty.
WAN_SAMPLE_NEG_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
    "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
    "杂乱的背景，三条腿，背景人很多，倒着走"
)


def snap_wan_pixel_dims(
    width: int,
    height: int,
    *,
    vae_scale: int,
    patch_size: tuple[int, int, int],
) -> tuple[int, int]:
    """Snap pixel ``width``×``height`` so latent grid aligns with VAE + DiT patch stride."""
    _, ph, pw = patch_size
    step_w = int(vae_scale) * int(pw)
    step_h = int(vae_scale) * int(ph)
    w = width - (width % step_w)
    h = height - (height % step_h)
    if w <= 0 or h <= 0:
        raise RuntimeError(
            f"Wan video size {width}x{height} is too small for vae_scale={vae_scale} "
            f"and patch_size={patch_size} (need width % {step_w} == 0, height % {step_h} == 0)"
        )
    return w, h


def best_output_size(w: int, h: int, dw: int, dh: int, expected_area: int) -> tuple[int, int]:
    """Pick output (width, height) divisible by ``dw``×``dh`` under ``expected_area``."""
    ratio = w / h
    ow = (expected_area * ratio) ** 0.5
    oh = expected_area / ow

    ow1 = int(ow // dw * dw)
    oh1 = int(expected_area / ow1 // dh * dh)
    assert ow1 % dw == 0 and oh1 % dh == 0 and ow1 * oh1 <= expected_area
    ratio1 = ow1 / oh1

    oh2 = int(oh // dh * dh)
    ow2 = int(expected_area / oh2 // dw * dw)
    assert oh2 % dh == 0 and ow2 % dw == 0 and ow2 * oh2 <= expected_area
    ratio2 = ow2 / oh2

    if max(ratio / ratio1, ratio1 / ratio) < max(ratio / ratio2, ratio2 / ratio):
        return ow1, oh1
    return ow2, oh2


def prepare_wan_reference_image(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    """Scale + center-crop reference RGB to ``target_w``×``target_h`` (official Wan TI2V I2V)."""
    iw, ih = img.size
    if iw <= 0 or ih <= 0:
        raise RuntimeError(f"invalid reference image size {iw}x{ih}")
    scale = max(target_w / iw, target_h / ih)
    resized = img.resize((round(iw * scale), round(ih * scale)), Image.LANCZOS)
    x1 = (resized.width - target_w) // 2
    y1 = (resized.height - target_h) // 2
    return resized.crop((x1, y1, x1 + target_w, y1 + target_h))


def wan_i2v_uses_channel_concat(config: Any) -> bool:
    """True for Wan 14B I2V (``in_dim`` > ``vae_z_dim``): concat noise + side in DiT forward."""
    z = int(getattr(config, "vae_z_dim", 0) or 0)
    din = int(getattr(config, "dim_in", 0) or 0)
    return z > 0 and din > z


def build_wan_i2v_side_channels(
    ctx: RuntimeContext,
    cond_latent: Any,
    num_latent_frames: int,
    latent_h: int,
    latent_w: int,
    *,
    temporal_vae_scale: int = 4,
) -> Any:
    """Build ``y = concat([mask(4), cond(16)], channel)`` for Wan2.2 I2V (official ``image2video.py``)."""
    t_lat = int(num_latent_frames)
    h = int(latent_h)
    w = int(latent_w)
    t_vae = int(temporal_vae_scale)
    if t_lat <= 0 or h <= 0 or w <= 0:
        raise RuntimeError(
            f"Wan I2V side channels need positive latent shape, got T={t_lat} H={h} W={w}"
        )
    if cond_latent.ndim == 5:
        cond_latent = ctx.squeeze(cond_latent, 0)
    if int(cond_latent.shape[0]) != 16:
        raise RuntimeError(
            f"Wan I2V cond latent expects 16 channels, got shape {getattr(cond_latent, 'shape', ())}"
        )
    if int(cond_latent.shape[1]) != t_lat:
        raise RuntimeError(
            f"Wan I2V cond latent temporal dim {cond_latent.shape[1]} != latent frames {t_lat}"
        )

    f_pix = (t_lat - 1) * t_vae + 1
    msk = ctx.ones((1, f_pix, h, w), dtype=cond_latent.dtype)
    if f_pix > 1:
        tail = ctx.zeros((1, f_pix - 1, h, w), dtype=cond_latent.dtype)
        msk = ctx.concat([msk[:, 0:1], tail], axis=1)
    first_rep = ctx.repeat(msk[:, 0:1], t_vae, axis=1)
    if f_pix > 1:
        msk = ctx.concat([first_rep, msk[:, 1:]], axis=1)
    else:
        msk = first_rep
    total = int(msk.shape[1])
    if total % t_vae != 0:
        raise RuntimeError(
            f"Wan I2V mask length {total} not divisible by temporal_vae_scale={t_vae}"
        )
    msk = ctx.reshape(msk, (1, total // t_vae, t_vae, h, w))
    msk = ctx.permute(msk, (0, 2, 1, 3, 4))
    msk = ctx.squeeze(msk, 0)
    return ctx.concat([msk, cond_latent], axis=0)


def expand_wan_cond_latent(ctx: RuntimeContext, cond: Any, target_t: int) -> Any:
    """Expand ``[C,1,H,W]`` VAE encode to ``[C,T,H,W]`` for I2V blending."""
    t = int(target_t)
    if int(cond.shape[1]) == t:
        return cond
    if int(cond.shape[1]) != 1:
        raise RuntimeError(
            f"Wan I2V cond latent temporal dim must be 1 or {t}, got {cond.shape[1]}"
        )
    c, _, h, w = cond.shape
    tail = ctx.zeros((c, t - 1, h, w), dtype=cond.dtype)
    return ctx.concat([cond, tail], axis=1)


def masks_like(ctx: RuntimeContext, tensors: list[Any], *, zero: bool = False, p: float = 0.2):
    """Return (mask1, mask2) with the same shape as each tensor entry.

    When ``zero=True`` (I2V), temporal index 0 is cleared in both masks — matching
    ``wan/utils/utils.py`` inference behaviour (deterministic; no random dropout).
    """
    del p
    out1 = [ctx.ones_like(u) for u in tensors]
    out2 = [ctx.ones_like(u) for u in tensors]
    if not zero:
        return out1, out2

    for i, u in enumerate(out1):
        v = out2[i]
        if u.ndim == 4:
            z = ctx.zeros_like(u[:, 0:1, :, :])
            out1[i] = ctx.concat([z, u[:, 1:, :, :]], axis=1) if int(u.shape[1]) > 1 else z
            out2[i] = ctx.concat([z, v[:, 1:, :, :]], axis=1) if int(v.shape[1]) > 1 else z
        elif u.ndim == 5:
            z = ctx.zeros_like(u[:, :, 0:1, :, :])
            out1[i] = ctx.concat([z, u[:, :, 1:, :, :]], axis=2) if int(u.shape[2]) > 1 else z
            out2[i] = ctx.concat([z, v[:, :, 1:, :, :]], axis=2) if int(v.shape[2]) > 1 else z
        else:
            raise RuntimeError(f"masks_like expects 4D/5D tensors, got shape {getattr(u, 'shape', ())}")
    return out1, out2


def prepare_ti2v_i2v_latents(
    ctx: RuntimeContext,
    noise: Any,
    cond_latent: Any,
    mask2: Any,
) -> Any:
    """Blend encoded first-frame conditioning with noise (TI2V I2V)."""
    batched = False
    if noise.ndim == 5:
        batched = True
        noise = ctx.squeeze(noise, 0)
    if mask2.ndim == 5:
        mask2 = ctx.squeeze(mask2, 0)
    if cond_latent.ndim == 5:
        cond_latent = ctx.squeeze(cond_latent, 0)
    blended = (1.0 - mask2) * cond_latent + mask2 * noise
    return ctx.expand_dims(blended, 0) if batched else blended


def wan_seq_len(
    num_latent_frames: int,
    latent_h: int,
    latent_w: int,
    patch_size: tuple[int, int, int],
) -> int:
    """Maximum DiT sequence length for Wan patch grid."""
    pt, ph, pw = patch_size
    del pt
    return math.ceil((latent_h * latent_w) / (ph * pw) * num_latent_frames)
