"""Asset group helpers for engine asset creation."""
from __future__ import annotations

from typing import Any, Optional


def _group_id_from_metadata(metadata: Optional[dict[str, Any]]) -> Optional[str]:
    if not metadata:
        return None
    direct = str(metadata.get("group_id") or metadata.get("asset_group_id") or "").strip()
    if direct:
        return direct
    project_id = str(metadata.get("long_video_project_id") or "").strip()
    return project_id or None


def resolve_asset_group_id(
    metadata: Optional[dict[str, Any]],
    asset_store: Any,
) -> Optional[str]:
    """Resolve and ensure an asset group from request metadata.

    Supports ``group_id`` / ``long_video_project_id`` -> auto-create a group named after the project.
    Returns ``None`` when no grouping metadata is present.
    """
    group_id = _group_id_from_metadata(metadata)
    if not group_id:
        return None
    ensure = getattr(asset_store, "ensure_group", None)
    if ensure is None:
        return group_id
    title = str((metadata or {}).get("long_video_project_title") or "").strip() or "Long video project"
    ensure(
        group_id,
        title=title,
        kind="mixed",
        metadata={"long_video_project_id": group_id},
    )
    return group_id


def resolve_group_id_from_asset(asset_store: Any, asset_id: str) -> Optional[str]:
    """Return ``group_id`` already assigned to an asset, if any."""
    getter = getattr(asset_store, "get_asset_record", None)
    if getter is None:
        return None
    try:
        row = getter(asset_id)
    except Exception:
        return None
    if not row:
        return None
    gid = row.get("group_id")
    return str(gid).strip() if gid else None
