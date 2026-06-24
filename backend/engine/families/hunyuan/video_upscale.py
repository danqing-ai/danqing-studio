"""HunyuanVideo 1.5 1080p SR — registered video upscale runner."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

from backend.engine.cache import ModelCache
from backend.engine.common.bundle.weights import parse_size_gb
from backend.engine.config.model_configs import get_config_class
from backend.engine.contracts import (
    inject_hunyuan_text_encoder_paths,
    local_bundle_root,
    registry_scalar_default,
    require_entry_family,
    resolve_version_block,
)
from backend.engine.families.hunyuan.sr import run_hunyuan_video_sr
from backend.engine.families.hunyuan.text_encoder import get_hunyuan_text_encoder
from backend.engine.families.hunyuan.vae import encode_hunyuan_rgb_to_latents
from backend.engine.common.codecs.vae.video_io import save_pil_frames_to_mp4
from backend.engine.runtime._base import RuntimeContext
from backend.engine.video_codec_registry import (
    resolve_hunyuan_vae_spatial_tiling,
    resolve_hunyuan_vae_temporal_chunk,
)


def run_hunyuan_1080p_sr(
    *,
    ctx: RuntimeContext,
    request: Any,
    ctx_exec: Any,
    entry: Any,
    version_key: str | None,
    model_key: str,
    asset_store: Any,
    model_registry: Any,
    model_cache: ModelCache | None,
    project_root: Path,
    on_progress: Callable | None = None,
    on_log: Callable | None = None,
) -> tuple[str, dict[str, Any]] | None:
    family = require_entry_family(entry, model_id=model_key)
    bundle_root = local_bundle_root(project_root, entry, version_key)
    if bundle_root is None:
        raise RuntimeError(f"HunyuanVideo SR bundle not installed for {model_key!r}")

    config_cls = get_config_class(family)
    config = config_cls()
    object.__setattr__(config, "use_meanflow", True)
    vst = registry_scalar_default(entry, "vae_spatial_tiling", None)
    if vst is not None:
        config.vae_spatial_tiling = bool(vst)
    inject_hunyuan_text_encoder_paths(entry, config, project_root)

    src_id = request.source_asset_id
    if not src_id:
        raise RuntimeError("Video upscale requires source_asset_id (low-res video latents or asset).")

    src_path = asset_store.get_file_path(src_id)
    if src_path is None or not src_path.exists():
        raise RuntimeError(f"Source asset not found: {src_id!r}")

    prompt = request.prompt or ""
    enc = get_hunyuan_text_encoder(ctx, bundle_root, config)
    txt_embeds, txt_mask, txt_embeds_2, txt_mask_2 = enc.encode([prompt])
    if getattr(ctx, "backend", None) == "mlx":
        ctx.clear_cache()

    if src_path.suffix.lower() in (".mp4", ".webm", ".mov", ".mkv", ".m4v"):
        raise RuntimeError(
            "HunyuanVideo 1080p SR expects a latent or still-image asset, not a container video file. "
            "Use seedvr2-video-7b (or seedvr2-video-3b) for MP4/WebM restoration."
        )

    img = Image.open(src_path).convert("RGB")
    arr = np.array(img).astype(np.float32) / 127.5 - 1.0
    pixels = ctx.array(arr[np.newaxis, np.newaxis, ...])

    vae_root = Path(getattr(request, "vae_bundle", "") or str(bundle_root))
    low_latents = encode_hunyuan_rgb_to_latents(ctx, pixels, vae_root)

    sr_steps = int(getattr(request, "steps", None) or 6)
    chunk = resolve_hunyuan_vae_temporal_chunk(entry, low_latents, registry_scalar_default)
    spatial = resolve_hunyuan_vae_spatial_tiling(entry, registry_scalar_default)
    ver = resolve_version_block(entry, version_key) or {}
    raw = getattr(entry, "raw", {}) or {}
    size_str = str((ver or {}).get("size") or raw.get("size") or "10GB")
    sr_cache_key = f"upscale:video:{entry.id}:{version_key or 'default'}"
    _, frames = run_hunyuan_video_sr(
        ctx,
        config,
        bundle_root,
        low_latents,
        txt_embeds=txt_embeds,
        txt_attn_mask=txt_mask,
        txt_embeds_2=txt_embeds_2,
        txt_attn_mask_2=txt_mask_2,
        vae_bundle_root=vae_root,
        steps=sr_steps,
        temporal_chunk_size=chunk,
        spatial_tiling=spatial or getattr(config, "vae_spatial_tiling", True),
        model_cache=model_cache,
        cache_key=sr_cache_key,
        cache_size_gb=parse_size_gb(size_str),
    )

    work = Path(ctx_exec.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    out_path = str(work / f"{model_key}_sr_{src_id}.mp4")
    fps = int(getattr(request, "fps", None) or 24)
    save_pil_frames_to_mp4(frames, out_path, fps=fps)

    if on_log:
        on_log("info", f"HunyuanVideo SR saved {out_path}")
    if on_progress:
        on_progress(1.0, 1, 1, None, "complete")

    return out_path, {
        "model": request.model,
        "source_asset_id": src_id,
        "mime_type": "video/mp4",
        "sr_steps": sr_steps,
    }
