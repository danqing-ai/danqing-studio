"""Deterministic I2V motion prompts from structured storyboard beats.

Inspired by LumenX camera_movement enums and Toonflow director storyboard tables:
motion = subject action + camera grammar, separate from static keyframe scene text.
"""
from __future__ import annotations

import re

_SHOT_TAG_ZH = re.compile(r"^【([^】]+)】\s*(.*)", re.S)
_SHOT_TAG_EN = re.compile(r"^\[(?:Shot|Frame)\]\s*(.+?)(?:\]\s*|\]\s*$)(.*)", re.I | re.S)
_SHOT_SIZES_ZH = frozenset(
    {"大远景", "远景", "全景", "中景", "近景", "特写", "大特写", "过肩", "主观"}
)
_SHOT_SIZES_EN = frozenset(
    {"extreme wide", "wide", "full", "medium", "close", "extreme close", "over shoulder", "pov"}
)
_NAME_ZH = re.compile(r"[\u4e00-\u9fff]{2,4}")
_PRONOUN_START = re.compile(r"^(她|他|它|they|she|he)\b", re.I)


def _split_visual_parts(text: str) -> tuple[str, str]:
    raw = (text or "").strip()
    if not raw:
        return "", ""
    m = _SHOT_TAG_ZH.match(raw)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    m = _SHOT_TAG_EN.match(raw)
    if m:
        return m.group(1).strip(), (m.group(2) or "").strip()
    if "|" in raw:
        parts = [p.strip() for p in raw.split("|")]
        if len(parts) >= 4:
            return parts[1], "，".join(p for p in parts[2:] if p)
        if len(parts) == 3:
            return parts[1], parts[2]
    return "", raw


def _camera_for_shot(shot_size: str, *, locale: str) -> str:
    s = (shot_size or "").strip().lower()
    zh = locale == "zh"
    if any(k in s for k in ("大远景", "extreme wide")):
        return "大远景缓慢推拉" if zh else "slow dolly on extreme wide shot"
    if any(k in s for k in ("远景", "wide")):
        return "固定机位，轻微缓慢推近" if zh else "locked camera with a gentle slow push-in"
    if any(k in s for k in ("全景", "full")):
        return "固定机位，轻微左右摇镜" if zh else "locked camera with a slight pan"
    if any(k in s for k in ("特写", "大特写", "close", "extreme close")):
        return "固定机位，浅景深，微推近" if zh else "locked camera, shallow depth, subtle push-in"
    if any(k in s for k in ("近景",)):
        return "固定机位，轻微手持感" if zh else "locked camera with subtle handheld drift"
    if any(k in s for k in ("过肩", "over shoulder")):
        return "过肩固定机位，轻微跟随" if zh else "over-shoulder hold with slight follow"
    if any(k in s for k in ("主观", "pov")):
        return "主观视角轻微晃动" if zh else "subjective POV with slight sway"
    return "固定机位" if zh else "locked camera"


def _action_from_visual(body: str, *, locale: str) -> str:
    text = (body or "").strip()
    if not text:
        return ""
    if "，" in text:
        parts = [p.strip() for p in text.split("，") if p.strip()]
        if len(parts) >= 2:
            action = "，".join(parts[1:])
            if len(action) >= 6:
                return action
    if "," in text and locale != "zh":
        parts = [p.strip() for p in text.split(",") if p.strip()]
        if len(parts) >= 2:
            return ", ".join(parts[1:])
    return text


def _names_in_text(text: str, *, locale: str) -> list[str]:
    if locale != "zh":
        return []
    seen: list[str] = []
    for m in _NAME_ZH.finditer(text or ""):
        name = m.group(0)
        if name in seen:
            continue
        seen.append(name)
    return seen[:3]


def _transition_hint(current: str, nxt: str, *, locale: str) -> str:
    cur = (current or "").strip()
    nxt = (nxt or "").strip()
    if not nxt or cur == nxt:
        return ""
    cur_names = set(_names_in_text(cur, locale=locale))
    nxt_names = set(_names_in_text(nxt, locale=locale))
    if locale == "zh":
        if cur_names != nxt_names and nxt_names - cur_names:
            return "画面节奏过渡至下一镜"
        return "动作自然延续"
    if cur_names != nxt_names and nxt_names - cur_names:
        return "transition toward the next beat"
    return "motion continues naturally"


def motion_prompt_from_beat(
    visual: str,
    *,
    beat: str = "",
    next_visual: str = "",
    locale: str | None = None,
) -> str:
    """Build an I2V motion line: named subject action + camera move (no wardrobe repeat)."""
    source = (visual or beat or "").strip()
    if not source:
        return ""
    loc = "zh" if locale != "en" and re.search(r"[\u4e00-\u9fff]", source) else "en"
    shot_size, body = _split_visual_parts(source)
    action = _action_from_visual(body, locale=loc)
    if not action:
        action = body or source
    names = _names_in_text(source, locale=loc)
    camera = _camera_for_shot(shot_size, locale=loc)
    transition = _transition_hint(source, next_visual, locale=loc)

    if loc == "zh":
        if names and not any(n in action for n in names):
            action = f"{'、'.join(names)}{action}"
        parts = [action, f"镜头{camera}"]
        if transition:
            parts.append(transition)
        return "；".join(p for p in parts if p)

    subject = ", ".join(names) if names else "on-screen subject"
    if action and subject.lower() not in action.lower():
        action = f"{subject}: {action}"
    parts = [action, camera]
    if transition:
        parts.append(transition)
    return "; ".join(p for p in parts if p)
