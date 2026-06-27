"""Novel chapter analysis for long-video storyboard (Map-Reduce for long text)."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable

from backend.engine.llm.storyboard import parse_plan_script

MAX_CHAPTER_CHARS = 120_000
CHAPTER_CHUNK_CHARS = 3500
MIN_SCENES = 2
MAX_SCENES = 24

_SYNOPSIS_RE = re.compile(
    r"\[Synopsis\]\s*(.+?)(?=\[Anchor\]|\[Beat|\Z)",
    re.S | re.I,
)

LONG_VIDEO_CHAPTER_ANALYZE_SYSTEM_PROMPT = """You analyze a novel chapter for segmented image-to-video storyboard.
Output format ONLY:
[Synopsis] <2-3 sentences: chapter plot summary>
[Anchor]
<Cast roster — blocks separated by a line containing only --- >
One block per character LOOK:
【角色·<姓名>·<装扮名>】<固定发型、服饰、体型、肤色>
---
【画风】<全片统一的色调、镜头、胶片感>
[Beat 1] <one filmable still moment — composition, pose, who is visible>
[Beat 2] ...
Each [Beat] = ONE keyframe (a photographable instant). No transitions, inner monologue, or dialogue quotes.
Name every on-screen character in each beat (never 她/他/she/he alone).
Extract 2-24 major visual scenes covering the chapter arc in order.
Match chapter language. No markdown outside the format."""

LONG_VIDEO_CHAPTER_CHUNK_SYSTEM_PROMPT = """Extract visual keyframe beats from this novel excerpt.
Output format ONLY — one line per beat:
[Beat 1] <filmable still moment>
[Beat 2] ...
Each beat = one photographable instant. Name visible characters. No inner thoughts or transitions.
Match excerpt language."""

LONG_VIDEO_CHAPTER_REDUCE_SYSTEM_PROMPT = """Merge partial scene beats from a long novel chapter into a final storyboard plan.
Output format ONLY:
[Synopsis] <2-3 sentence chapter summary>
[Anchor]
<Cast blocks with --- separators, same format as analysis>
【角色·<姓名>·<装扮名>】<appearance>
---
【画风】<style>
[Beat 1] ... through [Beat N] — exactly N beats, N between 2 and 24.
Preserve narrative order; merge redundant adjacent beats; drop non-visual lines.
Match chapter language."""


@dataclass(frozen=True)
class ChapterAnalyzeResult:
    chapter_title: str
    synopsis: str
    character_anchor: str
    style_anchor: str
    beat_sheet: list[str]
    llm_calls: int


def validate_chapter_text(text: str) -> str:
    raw = (text or "").strip()
    if not raw:
        raise ValueError("chapter text is empty")
    if len(raw) > MAX_CHAPTER_CHARS:
        raise ValueError(
            f"chapter text exceeds {MAX_CHAPTER_CHARS} characters; split into smaller sections"
        )
    return raw


def split_chapter_chunks(text: str, *, chunk_size: int = CHAPTER_CHUNK_CHARS) -> list[str]:
    raw = text.strip()
    if len(raw) <= chunk_size:
        return [raw]
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", raw) if p.strip()]
    if not paragraphs:
        paragraphs = [raw[i : i + chunk_size] for i in range(0, len(raw), chunk_size)]
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        candidate = f"{current}\n\n{para}".strip() if current else para
        if len(candidate) <= chunk_size:
            current = candidate
            continue
        if current:
            chunks.append(current)
        if len(para) <= chunk_size:
            current = para
        else:
            for i in range(0, len(para), chunk_size):
                chunks.append(para[i : i + chunk_size])
            current = ""
    if current:
        chunks.append(current)
    return chunks or [raw]


def parse_chapter_analyze_script(text: str) -> tuple[str, str, list[str]]:
    raw = (text or "").strip()
    synopsis_m = _SYNOPSIS_RE.search(raw)
    synopsis = synopsis_m.group(1).strip() if synopsis_m else ""
    character_anchor, beats = parse_plan_script(raw, expected_beats=0)
    return synopsis, character_anchor, beats


def merge_partial_beats(chunks: list[list[str]]) -> list[str]:
    merged: list[str] = []
    for partial in chunks:
        for beat in partial:
            line = (beat or "").strip()
            if not line or len(line) < 6:
                continue
            if merged and _beats_similar(merged[-1], line):
                continue
            merged.append(line)
    return merged


def clamp_scene_count(count: int) -> int:
    n = int(count)
    if n < MIN_SCENES:
        raise ValueError(
            f"chapter analysis found {n} visual scene(s); need at least {MIN_SCENES}. "
            "Try a longer chapter or add more descriptive prose."
        )
    if n > MAX_SCENES:
        raise ValueError(
            f"chapter analysis found {n} scenes; maximum is {MAX_SCENES}. "
            "Split the chapter into smaller sections."
        )
    return n


def _beats_similar(a: str, b: str) -> bool:
    na = re.sub(r"\s+", "", a.lower())
    nb = re.sub(r"\s+", "", b.lower())
    if not na or not nb:
        return False
    if na == nb:
        return True
    short, long = (na, nb) if len(na) <= len(nb) else (nb, na)
    return len(short) >= 12 and short in long


def run_chapter_analyze(
    *,
    chapter_text: str,
    chapter_title: str = "",
    locale: str = "zh",
    chat_fn: Callable[..., Any],
    think_apply: Callable[[str], str] | None = None,
    token_budget: Callable[[int], int] | None = None,
) -> ChapterAnalyzeResult:
    from backend.engine.llm.storyboard import (
        normalize_storyboard_locale,
        storyboard_language_rule,
        storyboard_language_user_suffix,
    )
    from backend.engine.llm.storyboard_cast import parse_character_roster

    raw = validate_chapter_text(chapter_text)
    loc = normalize_storyboard_locale(locale)
    lang_rule = storyboard_language_rule(loc)
    lang_suffix = storyboard_language_user_suffix(loc)
    apply_think = think_apply or (lambda t: t)
    budget = token_budget or (lambda b: b)

    chunks = split_chapter_chunks(raw)
    llm_calls = 0
    title = (chapter_title or "").strip()

    if len(chunks) == 1:
        user = f"Chapter title: {title or '(untitled)'}\n\nText:\n{raw[:CHAPTER_CHUNK_CHARS * 2]}"
        if len(raw) > CHAPTER_CHUNK_CHARS * 2:
            user = f"Chapter title: {title or '(untitled)'}\n\nText:\n{raw}"
        user += lang_suffix
        system = f"{LONG_VIDEO_CHAPTER_ANALYZE_SYSTEM_PROMPT}\n\n{lang_rule}"
        resp = chat_fn(system=system, user=apply_think(user), max_tokens=budget(900))
        llm_calls += 1
        synopsis, character_anchor, beat_sheet = parse_chapter_analyze_script(resp)
    else:
        partial_lists: list[list[str]] = []
        chunk_system = f"{LONG_VIDEO_CHAPTER_CHUNK_SYSTEM_PROMPT}\n\n{lang_rule}"
        for idx, chunk in enumerate(chunks):
            user = (
                f"Excerpt {idx + 1}/{len(chunks)} of chapter «{title or 'untitled'}»:\n\n{chunk}"
                + lang_suffix
            )
            resp = chat_fn(system=chunk_system, user=apply_think(user), max_tokens=budget(700))
            llm_calls += 1
            _, _, beats = parse_plan_script(resp, expected_beats=0)
            partial_lists.append(beats)
        merged = merge_partial_beats(partial_lists)
        beat_block = "\n".join(f"- {b}" for b in merged[: MAX_SCENES + 4])
        reduce_user = (
            f"Chapter title: {title or '(untitled)'}\n\n"
            f"Partial beats from {len(chunks)} excerpts ({len(merged)} lines):\n{beat_block}\n\n"
            f"Merge into {MIN_SCENES}-{MAX_SCENES} ordered keyframe beats with [Synopsis] and [Anchor]."
            + lang_suffix
        )
        reduce_system = f"{LONG_VIDEO_CHAPTER_REDUCE_SYSTEM_PROMPT}\n\n{lang_rule}"
        resp = chat_fn(
            system=reduce_system,
            user=apply_think(reduce_user),
            max_tokens=budget(1000),
        )
        llm_calls += 1
        synopsis, character_anchor, beat_sheet = parse_chapter_analyze_script(resp)
        if len(beat_sheet) < MIN_SCENES and merged:
            beat_sheet = merged[:MAX_SCENES]

    scene_count = clamp_scene_count(len(beat_sheet))
    beat_sheet = beat_sheet[:scene_count]

    if len(character_anchor.strip()) < 12:
        character_anchor = synopsis[:240].strip() or raw[:240].strip()

    _, style_anchor = parse_character_roster(character_anchor, locale=loc)
    if not synopsis:
        synopsis = raw[:300].strip()

    return ChapterAnalyzeResult(
        chapter_title=title,
        synopsis=synopsis,
        character_anchor=character_anchor,
        style_anchor=style_anchor,
        beat_sheet=beat_sheet,
        llm_calls=llm_calls,
    )
