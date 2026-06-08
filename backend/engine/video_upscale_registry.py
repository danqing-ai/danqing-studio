"""Video upscale dispatch — registry contract → family runner (no pipeline ``family ==``)."""
from __future__ import annotations

import importlib
from typing import Any, Callable

from backend.engine.contracts.pipeline_registry import resolve_version_block


_VIDEO_UPSCALE: dict[str, tuple[str, str]] = {
    "hunyuan_1080p_sr": (
        "backend.engine.families.hunyuan.video_upscale",
        "run_hunyuan_1080p_sr",
    ),
    "seedvr2_spatiotemporal": (
        "backend.engine.families.seedvr2.video_upscale",
        "run_seedvr2_video_upscale",
    ),
}


def resolve_video_upscale_kind(entry: Any, version_key: str | None) -> str:
    model_id = str(getattr(entry, "id", "") or "")
    family = str(getattr(entry, "family", "") or "")
    media = str(getattr(entry, "media", "") or "")
    ver = resolve_version_block(entry, version_key) or {}
    explicit = str(ver.get("video_upscale_kind") or "")
    if explicit:
        if explicit not in _VIDEO_UPSCALE:
            supported = ", ".join(sorted(_VIDEO_UPSCALE.keys()))
            raise RuntimeError(
                f"Unknown video_upscale_kind {explicit!r} on {model_id!r}; supported: {supported}"
            )
        return explicit
    if family == "seedvr2" and media == "video":
        return "seedvr2_spatiotemporal"
    sr_variant = str(ver.get("hunyuan_ms_variant") or "")
    if sr_variant == "1080p_sr_distilled" or "1080p-sr" in model_id:
        return "hunyuan_1080p_sr"
    raise RuntimeError(
        f"Video upscale is not configured for model {model_id!r} "
        f"(set video_upscale_kind on version block, or use seedvr2 video / hunyuan 1080p SR)."
    )


def get_video_upscale_runner(kind: str) -> Callable[..., Any]:
    entry = _VIDEO_UPSCALE.get(kind)
    if entry is None:
        supported = ", ".join(sorted(_VIDEO_UPSCALE.keys()))
        raise RuntimeError(
            f"Unknown video upscale kind {kind!r}; supported: {supported}"
        )
    mod = importlib.import_module(entry[0])
    fn = getattr(mod, entry[1])
    if fn is None:
        raise RuntimeError(f"Video upscale runner {entry[0]}.{entry[1]} is missing")
    return fn
