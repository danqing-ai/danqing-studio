"""Asset lineage helpers shared by image/video engines."""

from __future__ import annotations

from typing import Any, Optional, Tuple


def resolve_lineage(
    metadata: Optional[dict[str, Any]],
    *,
    parent_asset_id: Optional[str] = None,
    relation_type: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str]]:
    meta = metadata or {}
    parent = parent_asset_id or meta.get("parent_asset_id")
    relation = relation_type or meta.get("relation_type")
    if isinstance(parent, str):
        parent = parent.strip() or None
    else:
        parent = None
    if parent and not relation:
        relation = "create"
    if parent:
        return parent, str(relation) if relation else "create"
    return None, None


def image_edit_relation_type(operation: str, *, rewrite_mode: str = "") -> str:
    if operation == "rewrite" and rewrite_mode == "reference":
        return "img2img"
    return operation


def video_edit_relation_type(operation: str) -> str:
    if operation == "animate":
        return "animate"
    return operation or "edit"
