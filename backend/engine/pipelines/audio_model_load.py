"""Shared audio generator load path — AudioSession and v3 FamilyPlugin backbones."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from backend.engine._transformer_registry import get_audio_generation_factory
from backend.engine.contracts import resolve_version_block
from backend.engine.cache import ModelCache


def audio_generator_cache_key(entry: Any, version_key: str | None, family: str) -> str:
    return f"audio:{entry.id}:{version_key or 'default'}:{family}"


def load_audio_generator(
    *,
    ctx: Any,
    family: str,
    bundle_root: Path,
    entry: Any,
    version_key: str | None,
    model_cache: ModelCache | None = None,
) -> Any:
    cache_key = audio_generator_cache_key(entry, version_key, family)
    if model_cache is not None:
        cached = model_cache.get(cache_key)
        if cached is not None:
            return cached

    factory = get_audio_generation_factory(family)
    gen = factory(ctx, bundle_root)
    gen.load()

    if model_cache is not None:
        from backend.engine.common.bundle.weights import parse_size_gb

        ver = resolve_version_block(entry, version_key)
        size_str = ""
        if ver:
            size_str = str(ver.get("size") or "")
        if not size_str:
            raw = getattr(entry, "raw", {}) or {}
            size_str = str(raw.get("size") or "10GB")
        model_cache.put(cache_key, gen, parse_size_gb(size_str))
    return gen
