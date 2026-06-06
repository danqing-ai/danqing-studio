"""Video VAE encode/decode dispatch — keyed by ``ModelConfig.video_vae_backend``."""
from __future__ import annotations

from typing import Any, Callable


def _decode_hunyuan(
    *,
    ctx: Any,
    latents: Any,
    entry: Any,
    version_key: str | None,
    local_bundle_root: Callable[[Any, str | None], Any],
    registry_scalar_default: Callable[[Any, str, Any], Any],
    on_post_progress: Callable[[float], None] | None,
    on_post_log: Callable[[str], None] | None,
) -> list:
    from backend.engine.families.hunyuan.vae import decode_hunyuan_latents_to_pil_frames

    bundle_root = local_bundle_root(entry, version_key)
    chunk = int(registry_scalar_default(entry, "vae_temporal_chunk_size", 8) or 0)
    if chunk <= 0:
        temporal_chunk = 0
    elif getattr(latents, "ndim", None) == 5:
        t = int(latents.shape[2])
        temporal_chunk = 0 if t <= chunk else chunk
    else:
        temporal_chunk = chunk
    spatial = bool(registry_scalar_default(entry, "vae_spatial_tiling", False))
    return decode_hunyuan_latents_to_pil_frames(
        ctx,
        latents,
        bundle_root,
        on_stage=on_post_progress,
        on_log=on_post_log,
        temporal_chunk_size=temporal_chunk,
        spatial_tiling=spatial,
    )


def _encode_hunyuan(
    *,
    ctx: Any,
    image_tensor: Any,
    entry: Any,
    version_key: str | None,
    local_bundle_root: Callable[[Any, str | None], Any],
    registry_scalar_default: Callable[[Any, str, Any], Any],
    on_post_progress: Callable[[float], None] | None = None,
    on_post_log: Callable[[str], None] | None = None,
) -> Any:
    del registry_scalar_default, on_post_progress, on_post_log
    from backend.engine.families.hunyuan.vae import encode_hunyuan_rgb_to_latents

    bundle_root = local_bundle_root(entry, version_key)
    if bundle_root is None:
        return None
    if image_tensor.ndim == 4:
        image_tensor = ctx.unsqueeze(image_tensor, axis=2)
    return encode_hunyuan_rgb_to_latents(ctx, image_tensor, bundle_root)


def _decode_ltx(
    *,
    ctx: Any,
    latents: Any,
    entry: Any,
    version_key: str | None,
    local_bundle_root: Callable[[Any, str | None], Any],
    registry_scalar_default: Callable[[Any, str, Any], Any],
    on_post_progress: Callable[[float], None] | None,
    on_post_log: Callable[[str], None] | None,
) -> list:
    del registry_scalar_default
    from backend.engine.families.ltx.vae import decode_ltx_latents_to_pil_frames

    bundle_root = local_bundle_root(entry, version_key)
    return decode_ltx_latents_to_pil_frames(
        ctx,
        latents,
        bundle_root,
        on_stage=on_post_progress,
        on_log=on_post_log,
    )


def _decode_wan(
    *,
    ctx: Any,
    latents: Any,
    entry: Any,
    version_key: str | None,
    local_bundle_root: Callable[[Any, str | None], Any],
    registry_scalar_default: Callable[[Any, str, Any], Any],
    on_post_progress: Callable[[float], None] | None,
    on_post_log: Callable[[str], None] | None,
) -> list:
    from backend.engine.families.wan.vae import decode_wan_latents_to_pil_frames

    bundle_root = local_bundle_root(entry, version_key)
    spatial = bool(registry_scalar_default(entry, "vae_spatial_tiling", False))
    spatial_scale = int(registry_scalar_default(entry, "vae_scale", 16) or 16)
    return decode_wan_latents_to_pil_frames(
        ctx,
        latents,
        bundle_root,
        on_stage=on_post_progress,
        on_log=on_post_log,
        spatial_tiling=spatial,
        spatial_scale=spatial_scale,
    )


def _encode_wan(
    *,
    ctx: Any,
    image_tensor: Any,
    entry: Any,
    version_key: str | None,
    local_bundle_root: Callable[[Any, str | None], Any],
    registry_scalar_default: Callable[[Any, str, Any], Any],
    on_post_progress: Callable[[float], None] | None = None,
    on_post_log: Callable[[str], None] | None = None,
) -> Any:
    del registry_scalar_default, on_post_progress, on_post_log
    from backend.engine.families.wan.vae import encode_wan_image_to_latent

    bundle_root = local_bundle_root(entry, version_key)
    if bundle_root is None:
        return None
    if image_tensor.ndim == 4:
        chw = image_tensor[0]
    else:
        chw = image_tensor
    if int(chw.shape[0]) != 3:
        chw = ctx.permute(chw, (2, 0, 1))
    return encode_wan_image_to_latent(ctx, chw, bundle_root)


_VIDEO_DECODE: dict[str, Callable[..., Any]] = {
    "hunyuan": _decode_hunyuan,
    "ltx": _decode_ltx,
    "wan": _decode_wan,
}

_VIDEO_ENCODE: dict[str, Callable[..., Any]] = {
    "hunyuan": _encode_hunyuan,
    "wan": _encode_wan,
}


def get_video_decode_handler(video_vae_backend: str) -> Callable[..., Any] | None:
    return _VIDEO_DECODE.get(video_vae_backend)


def get_video_encode_handler(video_vae_backend: str) -> Callable[..., Any] | None:
    return _VIDEO_ENCODE.get(video_vae_backend)
