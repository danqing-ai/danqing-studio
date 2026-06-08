"""Shared upscale job pipeline load — UpscaleSession and v3 FamilyPlugin backbones."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from backend.engine.contracts import resolve_version_block
from backend.engine.upscale_job_registry import get_upscale_pipeline_loader


def upscale_pipeline_cache_key(entry: Any, version_key: str | None) -> str:
    return f"upscale:image:{entry.id}:{version_key or 'default'}"


def load_upscale_pipeline(
    *,
    family: str,
    bundle_path: Path,
    model_key: str,
    entry: Any,
    version_key: str | None,
    model_cache: Any | None = None,
    on_log: Callable[[str, str], None] | None = None,
) -> Any | None:
    """Load family upscale pipeline via registry (Shape B job paradigm)."""
    load_fn = get_upscale_pipeline_loader(family)
    if load_fn is None:
        return None

    from backend.engine.common.bundle.weights import parse_size_gb

    cache_key = upscale_pipeline_cache_key(entry, version_key)
    ver = resolve_version_block(entry, version_key)
    size_str = str(
        (ver or {}).get("size") or (getattr(entry, "raw", {}) or {}).get("size") or "8GB"
    )
    return load_fn(
        bundle_path=bundle_path,
        model_key=model_key,
        model_cache=model_cache,
        cache_key=cache_key,
        cache_size_gb=parse_size_gb(size_str),
        on_log=on_log,
    )
