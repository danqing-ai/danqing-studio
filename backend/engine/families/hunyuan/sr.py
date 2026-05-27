"""HunyuanVideo-1.5 SR pipeline helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from backend.engine.families.hunyuan.sr_mlx import load_hunyuan_sr_transformer, upscale_latents_to_1080p
from backend.engine.families.hunyuan.vae import decode_hunyuan_latents_to_pil_frames


def run_hunyuan_video_sr(
    ctx: Any,
    config: Any,
    bundle_root: Path,
    low_res_latents: Any,
    *,
    txt_embeds: Any,
    txt_attn_mask: Any,
    txt_embeds_2: Any,
    txt_attn_mask_2: Any,
    vae_bundle_root: Path | None = None,
    steps: int = 6,
    temporal_chunk_size: int = 0,
    spatial_tiling: bool = True,
    model_cache: Any | None = None,
    cache_key: str | None = None,
    cache_size_gb: float = 10.0,
) -> tuple[Any, list]:
    """SR latents then decode to PIL frames."""
    sr_model = load_hunyuan_sr_transformer(
        ctx,
        config,
        bundle_root,
        model_cache=model_cache,
        cache_key=cache_key,
        cache_size_gb=cache_size_gb,
    )
    hr_latents = upscale_latents_to_1080p(
        ctx,
        low_res_latents,
        sr_model,
        txt_embeds=txt_embeds,
        txt_attn_mask=txt_attn_mask,
        txt_embeds_2=txt_embeds_2,
        txt_attn_mask_2=txt_attn_mask_2,
        steps=steps,
    )
    del sr_model
    if hasattr(ctx, "clear_cache"):
        ctx.clear_cache()
    decode_root = vae_bundle_root or bundle_root
    frames = decode_hunyuan_latents_to_pil_frames(
        ctx,
        hr_latents,
        decode_root,
        temporal_chunk_size=temporal_chunk_size,
        spatial_tiling=spatial_tiling,
    )
    return hr_latents, frames
