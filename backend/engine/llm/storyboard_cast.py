"""Character roster + multi-look cast for long-video keyframe prompts."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

from backend.engine.llm.storyboard import (
    KEYFRAME_REF_DIVIDER,
    _LABELED_BLOCK_ZH,
    _STYLE_BLOCK_EN,
    _STYLE_LABELS,
    _split_anchor_raw_blocks,
    extract_keyframe_shot_scene,
    normalize_storyboard_locale,
    prompt_locale,
)

_LOOK_TRIPLE_ZH = re.compile(r"^【角色·([^·】]+)·([^】]+)】\s*(.+)", re.S)
_LOOK_TRIPLE_EN = re.compile(r"^\[Character:\s*([^|]+)\|\s*([^\]]+)\]\s*(.+)", re.I | re.S)
_LOOK_TAG_ZH = re.compile(r"([^（(]+)[（(]([^）)]+)[）)]")
_LOOK_TAG_EN = re.compile(r"([A-Za-z\u4e00-\u9fff][^\s,]+)\s*\(([^)]+)\)")


@dataclass
class CharacterLook:
    id: str
    label: str
    body: str


@dataclass
class StoryboardCharacter:
    id: str
    name: str
    looks: list[CharacterLook] = field(default_factory=list)
    default_look_id: str = ""


@dataclass
class ShotCastLook:
    character_id: str
    look_id: str


def _stable_id(prefix: str, key: str) -> str:
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:10]
    return f"{prefix}_{digest}"


def _default_look_label(locale: str) -> str:
    return "默认" if normalize_storyboard_locale(locale) == "zh" else "default"


def parse_character_roster(
    character_anchor: str,
    *,
    locale: str | None = None,
) -> tuple[list[StoryboardCharacter], str]:
    """Parse [Anchor]/[Cast] text into characters (multi-look) + shared style line."""
    anchor = (character_anchor or "").strip()
    if not anchor:
        return [], ""
    loc = normalize_storyboard_locale(locale) if locale else (
        "zh" if prompt_locale(anchor) == "zh" else "en"
    )
    default_label = _default_look_label(loc)
    style = ""
    by_name: dict[str, StoryboardCharacter] = {}

    for raw in _split_anchor_raw_blocks(anchor):
        block = raw.strip()
        if not block:
            continue
        m = _LOOK_TRIPLE_ZH.match(block)
        if m:
            name, look_label, body = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
            _add_look(by_name, name, look_label, body, default_label)
            continue
        m = _LOOK_TRIPLE_EN.match(block)
        if m:
            name, look_label, body = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
            _add_look(by_name, name, look_label, body, default_label)
            continue
        m = _LABELED_BLOCK_ZH.match(block)
        if m:
            label, body = m.group(1).strip(), m.group(2).strip()
            if label.startswith("角色·"):
                rest = label[3:]
                if "·" in rest:
                    name, look_label = rest.split("·", 1)
                    _add_look(by_name, name.strip(), look_label.strip(), body, default_label)
                else:
                    _add_look(by_name, rest.strip(), default_label, body, default_label)
                continue
            if label.lower() in _STYLE_LABELS or label in ("画风", "风格"):
                style = body
                continue
        m = _STYLE_BLOCK_EN.match(block)
        if m:
            style = m.group(2).strip()
            continue
        name_m = re.match(r"^([^，,：:]{1,16})[，,：:]\s*(.+)", block, re.S)
        if name_m:
            name, body = name_m.group(1).strip(), name_m.group(2).strip()
            if name in ("画风", "风格") or name.lower() in _STYLE_LABELS:
                style = body
            else:
                _add_look(by_name, name, default_label, body, default_label)
            continue
        if not style:
            style = block

    return list(by_name.values()), style


def _add_look(
    by_name: dict[str, StoryboardCharacter],
    name: str,
    look_label: str,
    body: str,
    default_label: str,
) -> None:
    if not name or not body:
        return
    char_id = _stable_id("char", name)
    if name not in by_name:
        by_name[name] = StoryboardCharacter(id=char_id, name=name, looks=[], default_look_id="")
    ch = by_name[name]
    label = look_label.strip() or default_label
    look_id = _stable_id("look", f"{name}|{label}")
    if any(lk.id == look_id for lk in ch.looks):
        return
    ch.looks.append(CharacterLook(id=look_id, label=label, body=body.strip()))
    if not ch.default_look_id:
        ch.default_look_id = look_id


def format_character_roster(
    characters: list[StoryboardCharacter],
    style_anchor: str,
    *,
    locale: str | None = None,
) -> str:
    """Serialize roster to --- separated Anchor/Cast blocks."""
    loc = normalize_storyboard_locale(locale) if locale else "zh"
    lines: list[str] = []
    for ch in characters:
        for lk in ch.looks:
            if loc == "zh":
                lines.append(f"【角色·{ch.name}·{lk.label}】{lk.body}")
            else:
                lines.append(f"[Character: {ch.name} | {lk.label}] {lk.body}")
    if style_anchor.strip():
        lines.append(f"【画风】{style_anchor.strip()}" if loc == "zh" else f"[Style] {style_anchor.strip()}")
    return f"\n{KEYFRAME_REF_DIVIDER}\n".join(lines)


def roster_to_dtos(characters: list[StoryboardCharacter]) -> list[dict]:
    out: list[dict] = []
    for ch in characters:
        out.append(
            {
                "id": ch.id,
                "name": ch.name,
                "default_look_id": ch.default_look_id,
                "looks": [{"id": lk.id, "label": lk.label, "body": lk.body} for lk in ch.looks],
            }
        )
    return out


def dtos_to_roster(items: list[dict]) -> list[StoryboardCharacter]:
    chars: list[StoryboardCharacter] = []
    for row in items or []:
        looks = [
            CharacterLook(id=str(lk.get("id", "")), label=str(lk.get("label", "")), body=str(lk.get("body", "")))
            for lk in (row.get("looks") or [])
            if lk.get("body")
        ]
        if not looks:
            continue
        chars.append(
            StoryboardCharacter(
                id=str(row.get("id", "")),
                name=str(row.get("name", "")),
                looks=looks,
                default_look_id=str(row.get("default_look_id") or looks[0].id),
            )
        )
    return chars


def characters_on_screen(scene: str, characters: list[StoryboardCharacter]) -> list[StoryboardCharacter]:
    text = (scene or "").strip()
    if not text:
        return []
    return [ch for ch in characters if ch.name and ch.name in text]


def _match_look_by_hint(ch: StoryboardCharacter, hint: str) -> CharacterLook | None:
    hint = (hint or "").strip()
    if not hint:
        return None
    for lk in ch.looks:
        if hint == lk.label or hint in lk.label or lk.label in hint:
            return lk
    for lk in ch.looks:
        if hint in lk.body:
            return lk
    return None


def infer_look_label_from_text(text: str, character_name: str) -> str | None:
    raw = text or ""
    for m in _LOOK_TAG_ZH.finditer(raw):
        if m.group(1).strip() == character_name:
            return m.group(2).strip()
    for m in _LOOK_TAG_EN.finditer(raw):
        if m.group(1).strip() == character_name:
            return m.group(2).strip()
    return None


def infer_shot_cast_looks(
    *,
    scene: str,
    beat: str,
    characters: list[StoryboardCharacter],
    prev: list[ShotCastLook] | None = None,
) -> list[ShotCastLook]:
    """Pick outfit per on-screen character from beat tags, else carry previous/default."""
    prev_map = {c.character_id: c.look_id for c in (prev or [])}
    cast: list[ShotCastLook] = []
    for ch in characters_on_screen(scene or beat, characters):
        hint = infer_look_label_from_text(beat, ch.name) or infer_look_label_from_text(scene, ch.name)
        look_id = prev_map.get(ch.id) or ch.default_look_id
        if hint:
            matched = _match_look_by_hint(ch, hint)
            if matched:
                look_id = matched.id
        if not look_id and ch.looks:
            look_id = ch.looks[0].id
        if look_id:
            cast.append(ShotCastLook(character_id=ch.id, look_id=look_id))
    return cast


def cast_looks_to_dtos(cast: list[ShotCastLook]) -> list[dict]:
    return [{"character_id": c.character_id, "look_id": c.look_id} for c in cast]


def dtos_to_cast_looks(items: list[dict]) -> list[ShotCastLook]:
    out: list[ShotCastLook] = []
    for row in items or []:
        cid = str(row.get("character_id", "")).strip()
        lid = str(row.get("look_id", "")).strip()
        if cid and lid:
            out.append(ShotCastLook(character_id=cid, look_id=lid))
    return out


def format_cast_reference_blocks(
    characters: list[StoryboardCharacter],
    cast: list[ShotCastLook],
    style_anchor: str,
    *,
    locale: str | None = None,
) -> str:
    loc = normalize_storyboard_locale(locale) if locale else "zh"
    cast_map = {c.character_id: c.look_id for c in cast}
    lines: list[str] = []
    for ch in characters:
        look_id = cast_map.get(ch.id)
        if cast and ch.id not in cast_map:
            continue
        lk = next((l for l in ch.looks if l.id == look_id), None) if look_id else None
        if not lk and ch.looks:
            lk = next((l for l in ch.looks if l.id == ch.default_look_id), ch.looks[0])
        if not lk:
            continue
        if loc == "zh":
            lines.append(f"【角色·{ch.name}·{lk.label}】{lk.body}")
        else:
            lines.append(f"[Character: {ch.name} | {lk.label}] {lk.body}")
    if style_anchor.strip():
        lines.append(f"【画风】{style_anchor.strip()}" if loc == "zh" else f"[Style] {style_anchor.strip()}")
    return f"\n{KEYFRAME_REF_DIVIDER}\n".join(lines)


def compose_keyframe_with_cast(
    scene: str,
    *,
    characters: list[StoryboardCharacter],
    cast: list[ShotCastLook],
    style_anchor: str = "",
    locale: str | None = None,
    character_anchor: str = "",
) -> str:
    """Build T2I prompt: scene first; cast/style reference appended after ---."""
    scene = (scene or "").strip()
    if not scene and not characters and not character_anchor:
        return ""
    loc = normalize_storyboard_locale(locale) if locale else (
        "zh" if prompt_locale(scene or character_anchor) == "zh" else "en"
    )
    if not characters and character_anchor.strip():
        from backend.engine.llm.storyboard import compose_keyframe_visual_prompt

        return compose_keyframe_visual_prompt(scene or character_anchor, character_anchor, locale=loc)
    screen_chars = characters_on_screen(scene, characters) if scene else list(characters)
    on_screen = {ch.id for ch in screen_chars}
    cast_filtered = [c for c in cast if c.character_id in on_screen] if on_screen else cast
    if not cast_filtered and screen_chars:
        cast_filtered = infer_shot_cast_looks(scene=scene, beat=scene, characters=characters)
    ref_chars = screen_chars if screen_chars else characters
    ref = format_cast_reference_blocks(ref_chars, cast_filtered, style_anchor, locale=loc)
    if not scene:
        return ref
    if not ref:
        return scene
    from backend.engine.llm.storyboard import join_keyframe_prompt

    return join_keyframe_prompt(scene, ref, locale=loc)


def shot_cast_from_visual(
    visual: str,
    characters: list[StoryboardCharacter],
) -> list[ShotCastLook]:
    """Best-effort parse look labels from an already-composed visual prompt."""
    scene = extract_keyframe_shot_scene(visual)
    return infer_shot_cast_looks(scene=scene, beat=visual, characters=characters)
