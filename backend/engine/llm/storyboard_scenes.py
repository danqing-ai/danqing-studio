"""Scene / location roster + per-shot scene look binding for long-video keyframes."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

from backend.engine.llm.storyboard import KEYFRAME_REF_DIVIDER, normalize_storyboard_locale, prompt_locale
from backend.engine.llm.storyboard_cast import _split_anchor_raw_blocks

_SCENE_TRIPLE_ZH = re.compile(r"^【场景·([^·】]+)·([^】]+)】\s*(.+)", re.S)
_SCENE_TRIPLE_EN = re.compile(r"^\[Scene:\s*([^|]+)\|\s*([^\]]+)\]\s*(.+)", re.I | re.S)
_SCENE_LABELED_ZH = re.compile(r"^【([^】]+)】\s*(.+)", re.S)
_VARIANT_TAG_ZH = re.compile(r"([^（(]+)[（(]([^）)]+)[）)]")
_VARIANT_TAG_EN = re.compile(r"([A-Za-z\u4e00-\u9fff][^\s,]+)\s*\(([^)]+)\)")


@dataclass
class SceneLook:
    id: str
    label: str
    body: str


@dataclass
class StoryboardScene:
    id: str
    name: str
    looks: list[SceneLook] = field(default_factory=list)
    default_look_id: str = ""


@dataclass
class ShotSceneLook:
    scene_id: str
    look_id: str


def _stable_id(prefix: str, key: str) -> str:
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:10]
    return f"{prefix}_{digest}"


def _default_look_label(locale: str) -> str:
    return "默认" if normalize_storyboard_locale(locale) == "zh" else "default"


def parse_scene_beat_location(beat: str) -> str:
    """Extract location/time segment from a storyboard beat string."""
    raw = (beat or "").strip()
    if not raw:
        return ""
    shot_match = re.match(r"^【([^】]+)】([\s\S]*)$", raw)
    if not shot_match:
        return ""
    rest = shot_match.group(2).strip()
    comma_idx = re.search(r"[，,]", rest)
    if comma_idx:
        return rest[: comma_idx.start()].strip()
    return ""


def _normalize_location_key(text: str) -> str:
    key = re.sub(r"\s+", "", (text or "").strip().lower())
    key = re.sub(r"[·/\\|｜\-—–]", "", key)
    return key


def _locations_similar(a: str, b: str) -> bool:
    ka, kb = _normalize_location_key(a), _normalize_location_key(b)
    if not ka or not kb:
        return False
    if ka == kb:
        return True
    short, long = (ka, kb) if len(ka) <= len(kb) else (kb, ka)
    if len(short) >= 3 and short in long:
        return True
    if (
        len(ka) >= 3
        and len(kb) >= 3
        and ka[0] == kb[0]
        and ka[-2:] == kb[-2:]
    ):
        return True
    return len(short) >= 4 and short in long


def _look_environment_snippet(body: str, *, max_len: int = 10) -> str:
    raw = (body or "").strip()
    for prefix in ("环境：", "Environment:"):
        if raw.startswith(prefix):
            segment = raw.split("|", 1)[0][len(prefix) :].strip()
            segment = re.sub(r"[，,；;。\.!\?！？]", " ", segment)
            segment = re.sub(r"\s+", " ", segment).strip()
            if segment:
                return segment[:max_len].strip()
    return ""


def _disambiguate_scene_looks(sc: StoryboardScene) -> None:
    used: set[str] = set()
    next_looks: list[SceneLook] = []
    for index, lk in enumerate(sc.looks):
        label = (lk.label or "").strip() or "默认"
        if label in used:
            hint = _look_environment_snippet(lk.body) or str(index + 1)
            candidate = f"{label}·{hint}"
            suffix = 2
            while candidate in used:
                candidate = f"{label}·{hint}{suffix}"
                suffix += 1
            label = candidate
        used.add(label)
        next_looks.append(
            SceneLook(
                id=_stable_id("slook", f"{sc.name}|{label}|{index}|{lk.body[:64]}"),
                label=label,
                body=lk.body,
            )
        )
    sc.looks = next_looks
    if sc.looks and not any(lk.id == sc.default_look_id for lk in sc.looks):
        sc.default_look_id = sc.looks[0].id


def normalize_storyboard_scene_roster(scenes: list[StoryboardScene]) -> list[StoryboardScene]:
    """Merge near-duplicate locations and ensure unique look labels + ids per scene."""
    merged: list[StoryboardScene] = []
    for sc in scenes:
        name = (sc.name or "").strip()
        if not name:
            continue
        target: StoryboardScene | None = None
        for existing in merged:
            if _locations_similar(existing.name, name):
                target = existing
                break
        if target is None:
            merged.append(
                StoryboardScene(
                    id=_stable_id("scene", name),
                    name=name,
                    looks=list(sc.looks),
                    default_look_id=sc.default_look_id,
                )
            )
            continue
        if len(name) > len(target.name):
            target.name = name
            target.id = _stable_id("scene", name)
        target.looks.extend(sc.looks)
    for sc in merged:
        _disambiguate_scene_looks(sc)
    return merged


def _add_look(
    by_name: dict[str, StoryboardScene],
    name: str,
    look_label: str,
    body: str,
    default_label: str,
) -> None:
    if not name or not body:
        return
    scene_id = _stable_id("scene", name)
    if name not in by_name:
        by_name[name] = StoryboardScene(id=scene_id, name=name, looks=[], default_look_id="")
    sc = by_name[name]
    label = look_label.strip() or default_label
    look_id = _stable_id("slook", f"{name}|{label}|{len(sc.looks)}|{body[:64]}")
    if any(lk.label == label for lk in sc.looks):
        hint = _look_environment_snippet(body) or str(len(sc.looks) + 1)
        label = f"{label}·{hint}"
        look_id = _stable_id("slook", f"{name}|{label}|{len(sc.looks)}|{body[:64]}")
    sc.looks.append(SceneLook(id=look_id, label=label, body=body.strip()))
    if not sc.default_look_id:
        sc.default_look_id = look_id


def parse_scene_roster(
    scene_anchor: str,
    *,
    locale: str | None = None,
) -> list[StoryboardScene]:
    anchor = (scene_anchor or "").strip()
    if not anchor:
        return []
    loc = normalize_storyboard_locale(locale) if locale else (
        "zh" if prompt_locale(anchor) == "zh" else "en"
    )
    default_label = _default_look_label(loc)
    by_name: dict[str, StoryboardScene] = {}

    for raw in _split_anchor_raw_blocks(anchor):
        block = raw.strip()
        if not block:
            continue
        m = _SCENE_TRIPLE_ZH.match(block)
        if m:
            name, look_label, body = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
            _add_look(by_name, name, look_label, body, default_label)
            continue
        m = _SCENE_TRIPLE_EN.match(block)
        if m:
            name, look_label, body = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
            _add_look(by_name, name, look_label, body, default_label)
            continue
        m = _SCENE_LABELED_ZH.match(block)
        if m:
            label, body = m.group(1).strip(), m.group(2).strip()
            if label.startswith("场景·"):
                rest = label[3:]
                if "·" in rest:
                    name, look_label = rest.split("·", 1)
                    _add_look(by_name, name.strip(), look_label.strip(), body, default_label)
                else:
                    _add_look(by_name, rest.strip(), default_label, body, default_label)
            continue

    return list(by_name.values())


def format_scene_roster(
    scenes: list[StoryboardScene],
    *,
    locale: str | None = None,
) -> str:
    loc = normalize_storyboard_locale(locale) if locale else "zh"
    lines: list[str] = []
    for sc in scenes:
        for lk in sc.looks:
            if loc == "zh":
                lines.append(f"【场景·{sc.name}·{lk.label}】{lk.body}")
            else:
                lines.append(f"[Scene: {sc.name} | {lk.label}] {lk.body}")
    return f"\n{KEYFRAME_REF_DIVIDER}\n".join(lines)


def roster_to_dtos(scenes: list[StoryboardScene]) -> list[dict]:
    out: list[dict] = []
    for sc in scenes:
        out.append(
            {
                "id": sc.id,
                "name": sc.name,
                "default_look_id": sc.default_look_id,
                "looks": [{"id": lk.id, "label": lk.label, "body": lk.body} for lk in sc.looks],
            }
        )
    return out


def dtos_to_roster(items: list[dict]) -> list[StoryboardScene]:
    scenes: list[StoryboardScene] = []
    for row in items or []:
        looks = [
            SceneLook(id=str(lk.get("id", "")), label=str(lk.get("label", "")), body=str(lk.get("body", "")))
            for lk in (row.get("looks") or [])
            if lk.get("body")
        ]
        if not looks:
            continue
        scenes.append(
            StoryboardScene(
                id=str(row.get("id", "")),
                name=str(row.get("name", "")),
                looks=looks,
                default_look_id=str(row.get("default_look_id") or looks[0].id),
            )
        )
    return scenes


def _match_look_by_hint(sc: StoryboardScene, hint: str) -> SceneLook | None:
    hint = (hint or "").strip()
    if not hint:
        return None
    for lk in sc.looks:
        if hint == lk.label or hint in lk.label or lk.label in hint:
            return lk
    for lk in sc.looks:
        if hint in lk.body:
            return lk
    return None


def _match_scene_by_location(scenes: list[StoryboardScene], location: str) -> StoryboardScene | None:
    loc = (location or "").strip()
    if not loc or not scenes:
        return None
    for sc in scenes:
        if _locations_similar(sc.name, loc):
            return sc
    for sc in scenes:
        for lk in sc.looks:
            if _locations_similar(lk.label, loc) or _locations_similar(f"{sc.name}{lk.label}", loc):
                return sc
    loc_key = _normalize_location_key(loc)
    best: StoryboardScene | None = None
    best_score = 0
    for sc in scenes:
        for lk in sc.looks:
            combined = f"{sc.name}{lk.label}"
            ck = _normalize_location_key(combined)
            if not ck:
                continue
            score = 0
            if loc_key in ck or ck in loc_key:
                score = min(len(loc_key), len(ck))
            elif any(part in ck for part in re.split(r"[·/\\|｜，,]", loc) if len(part.strip()) >= 2):
                score = 4
            if score > best_score:
                best_score = score
                best = sc
    return best


def infer_variant_hint_from_location(location: str, scene_name: str) -> str | None:
    loc = (location or "").strip()
    name = (scene_name or "").strip()
    if not loc:
        return None
    if name and loc.startswith(name):
        rest = loc[len(name) :].lstrip("·/\\|｜，, ")
        return rest.strip() or None
    for sep in ("·", "/", "|", "｜"):
        if sep in loc:
            parts = [p.strip() for p in loc.split(sep) if p.strip()]
            if len(parts) >= 2 and _locations_similar(parts[0], name):
                return sep.join(parts[1:])
    return loc if name and not _locations_similar(loc, name) else None


def infer_shot_scene_look(
    *,
    beat: str,
    scenes: list[StoryboardScene],
    prev: ShotSceneLook | None = None,
) -> ShotSceneLook | None:
    location = parse_scene_beat_location(beat)
    if not location and prev:
        return prev
    sc = _match_scene_by_location(scenes, location)
    if not sc:
        if prev:
            return prev
        return None
    hint = infer_variant_hint_from_location(location, sc.name)
    look_id = prev.look_id if prev and prev.scene_id == sc.id else sc.default_look_id
    if hint:
        matched = _match_look_by_hint(sc, hint)
        if matched:
            look_id = matched.id
    if not look_id and sc.looks:
        look_id = sc.looks[0].id
    if not look_id:
        return None
    return ShotSceneLook(scene_id=sc.id, look_id=look_id)


def scene_look_to_dtos(item: ShotSceneLook | None) -> dict | None:
    if not item:
        return None
    return {"scene_id": item.scene_id, "look_id": item.look_id}


def dto_to_scene_look(row: dict | None) -> ShotSceneLook | None:
    if not row:
        return None
    sid = str(row.get("scene_id", "")).strip()
    lid = str(row.get("look_id", "")).strip()
    if sid and lid:
        return ShotSceneLook(scene_id=sid, look_id=lid)
    return None


def scenes_from_beat_locations(
    beat_sheet: list[str],
    *,
    locale: str | None = None,
) -> list[StoryboardScene]:
    """Deterministic fallback: one scene entity per unique beat location."""
    loc = normalize_storyboard_locale(locale) if locale else "zh"
    default_label = _default_look_label(loc)
    by_name: dict[str, StoryboardScene] = {}
    for beat in beat_sheet:
        location = parse_scene_beat_location(beat)
        if not location:
            continue
        name = location.split("·")[0].split("/")[0].split("|")[0].strip()
        if not name:
            continue
        variant = infer_variant_hint_from_location(location, name) or default_label
        body = (
            f"环境：{location} | 置景：与地点名称一致的空间结构与典型光线"
            if loc == "zh"
            else f"Environment: {location} | Set dressing: spatial layout and lighting typical of this place"
        )
        _add_look(by_name, name, variant, body, default_label)
    return list(by_name.values())


def merge_scene_rosters(
    existing: list[StoryboardScene],
    incoming: list[StoryboardScene],
) -> list[StoryboardScene]:
    """Merge incoming LLM roster into existing, preserving reference_asset_id on frontend side."""
    by_id = {sc.id: sc for sc in existing}
    by_name = {sc.name: sc for sc in existing}
    merged: list[StoryboardScene] = list(existing)

    for inc in incoming:
        target = by_id.get(inc.id) or by_name.get(inc.name)
        if not target:
            merged.append(inc)
            by_id[inc.id] = inc
            by_name[inc.name] = inc
            continue
        existing_labels = {lk.label for lk in target.looks}
        for lk in inc.looks:
            if lk.label not in existing_labels:
                target.looks.append(lk)
                existing_labels.add(lk.label)
    return merged
