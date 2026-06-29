"""Layer 2 — spatial layout per scene location (LLM + minimal fallback)."""
from __future__ import annotations

from typing import Any, Callable

from pydantic import ValidationError

from backend.engine.llm.chapter_segment_plan import _split_beat_fields
from backend.engine.llm.chat_invoke import invoke_text_chat
from backend.engine.llm.json_output import extract_json_object
from backend.engine.llm.prompts.system import CHAPTER_SPATIAL_LAYOUT_SYSTEM
from backend.engine.llm.schemas.long_video import SpatialLayoutBatchSchema

SpatialLayoutByKey = dict[str, dict[str, Any]]


def _location_keys(beat_sheet: list[str]) -> list[tuple[str, str]]:
    seen: dict[str, str] = {}
    for beat_raw in beat_sheet:
        _title, _shot, location, narrative = _split_beat_fields(beat_raw)
        loc = (location or "scene").strip() or "scene"
        key = loc.replace("·", "_").replace(" ", "_")[:48]
        if key not in seen:
            seen[key] = loc
    return list(seen.items())


def _rule_spatial_layout(beat_sheet: list[str]) -> SpatialLayoutByKey:
    out: SpatialLayoutByKey = {}
    for key, loc in _location_keys(beat_sheet):
        out[key] = {
            "scene_key": key,
            "location": loc,
            "dimensions": "medium interior",
            "objects": [],
            "camera_zones": [
                {"id": f"{key}_wide", "description": "establishing wide", "visible_area": "full room"},
                {"id": f"{key}_entry", "description": "door/entry side", "visible_area": "entry + subject path"},
            ],
        }
    return out


def run_spatial_layout_pass(
    *,
    beat_sheet: list[str],
    synopsis: str,
    locale_block: str,
    chat_fn: Callable[..., Any],
    think_apply: Callable[[str], str],
    max_tokens: int,
) -> tuple[SpatialLayoutByKey, int]:
    keys = _location_keys(beat_sheet)
    if not keys:
        return {}, 0
    rows = [f"scene_key={k} location={loc}" for k, loc in keys]
    user = (
        f"Synopsis:\n{synopsis.strip()}\n\nScenes:\n"
        + "\n".join(rows)
        + "\n\n"
        + locale_block
    )
    try:
        resp = invoke_text_chat(
            chat_fn,
            system=CHAPTER_SPATIAL_LAYOUT_SYSTEM,
            user=user,
            max_tokens=max_tokens,
            think_apply=think_apply,
        )
        payload = SpatialLayoutBatchSchema.model_validate(extract_json_object(resp))
        out: SpatialLayoutByKey = {}
        for scene in payload.scenes:
            key = scene.scene_key.strip()
            out[key] = {
                "scene_key": key,
                "location": scene.location.strip(),
                "dimensions": scene.dimensions.strip(),
                "objects": list(scene.objects),
                "camera_zones": [
                    {"id": z.id, "description": z.description, "visible_area": z.visible_area}
                    for z in scene.camera_zones
                ],
            }
        for key, loc in keys:
            out.setdefault(key, _rule_spatial_layout(beat_sheet)[key])
        return out, 1
    except (ValueError, ValidationError):
        return _rule_spatial_layout(beat_sheet), 0


def attach_spatial_layout_to_scenes(
    scene_dtos: list[dict[str, Any]],
    beat_sheet: list[str],
    layouts: SpatialLayoutByKey,
) -> list[dict[str, Any]]:
    """Match scene entities to layout by location name heuristics."""
    if not layouts:
        return scene_dtos
    loc_to_key = {loc: key for key, loc in _location_keys(beat_sheet)}
    out: list[dict[str, Any]] = []
    for row in scene_dtos:
        copy = dict(row)
        name = str(copy.get("name") or "").strip()
        key = None
        for loc, k in loc_to_key.items():
            if loc in name or name in loc:
                key = k
                break
        if key and key in layouts:
            copy["spatial_layout_json"] = layouts[key]
        out.append(copy)
    return out


def ensure_scenes_from_beats(
    scene_dtos: list[dict[str, Any]],
    beat_sheet: list[str],
    layouts: SpatialLayoutByKey,
) -> list[dict[str, Any]]:
    """Ensure every beat location has a scene row with spatial layout."""
    import hashlib

    by_name: dict[str, dict[str, Any]] = {}
    for row in scene_dtos:
        name = str(row.get("name") or "").strip()
        if name:
            by_name[name] = dict(row)

    out: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for _i, beat_raw in enumerate(beat_sheet):
        _title, _shot, location, _narr = _split_beat_fields(beat_raw)
        loc = (location or "scene").strip() or "scene"
        if loc in seen_names:
            continue
        seen_names.add(loc)
        key = loc.replace("·", "_").replace(" ", "_")[:48]
        layout = layouts.get(key) or layouts.get(loc) or {
            "scene_key": key,
            "location": loc,
            "camera_zones": [{"id": f"{key}_wide", "description": "establishing wide"}],
        }
        if loc in by_name:
            row = dict(by_name[loc])
            if not row.get("spatial_layout_json"):
                row["spatial_layout_json"] = layout
            out.append(row)
        else:
            digest = hashlib.sha1(loc.encode("utf-8")).hexdigest()[:8]
            out.append(
                {
                    "id": f"scene_{digest}",
                    "name": loc,
                    "default_look_id": "default",
                    "looks": [{"id": "default", "label": "default", "body": loc}],
                    "spatial_layout_json": layout,
                }
            )
    if not out:
        return scene_dtos
    return out
