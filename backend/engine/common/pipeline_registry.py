"""Shared registry / path helpers for ImagePipeline and VideoPipeline.

Keeps bundle resolution and parameter defaults identical across media pipelines.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from backend.core.bundle_repos import version_primary_local_path


def resolve_project_path(project_root: Path, local_path: str) -> Path:
    p = Path(local_path)
    if p.is_absolute():
        return p
    return (project_root / local_path).resolve()


def registry_scalar_default(entry: Any, key: str, fallback: Any) -> Any:
    spec = (entry.parameters or {}).get(key)
    if spec is None:
        return fallback
    if isinstance(spec, dict):
        return spec.get("default", fallback)
    return spec  # direct value (list / int / str etc.), not wrapped in dict


def resolve_version_block(entry: Any, version_key: str | None) -> dict | None:
    raw = getattr(entry, "raw", {}) or {}
    versions = raw.get("versions") or {}
    if version_key and version_key in versions and isinstance(versions[version_key], dict):
        return versions[version_key]
    for vinfo in versions.values():
        if isinstance(vinfo, dict) and vinfo.get("default"):
            return vinfo
    return None


def local_bundle_root(project_root: Path, entry: Any, version_key: str | None) -> Path | None:
    block = resolve_version_block(entry, version_key)
    if not block:
        return None
    lp = (block.get("local_path") or "").strip()
    if not lp:
        try:
            lp = version_primary_local_path(block)
        except ValueError:
            return None
    path = resolve_project_path(project_root, lp)
    return path if path.exists() else None
