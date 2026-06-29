"""Novel chapter / story brief analysis for long-video storyboard (Map-Reduce for long text)."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any, Callable, Literal

from pydantic import ValidationError

from backend.engine.llm.chat_invoke import invoke_text_chat
from backend.engine.llm.json_output import extract_json_object
from backend.engine.llm.prompts.system import (
    CHAPTER_PLAN_SYSTEM,
    CHAPTER_CHUNK_SYSTEM,
    CHAPTER_REDUCE_SYSTEM,
    CHAPTER_ROSTER_SYSTEM,
)
from backend.engine.llm.schemas.long_video import (
    BeatSchema,
    ChapterAnalyzeSchema,
    ChapterChunkSchema,
    ChapterPlanSchema,
    ChapterRosterSchema,
)
from backend.engine.llm.storyboard_cast import (
    CharacterLook,
    StoryboardCharacter,
    format_character_roster,
    normalize_look_label,
    _default_look_label,
)

MAX_CHAPTER_CHARS = 120_000
CHAPTER_CHUNK_CHARS = 3500
MIN_SCENES = 2
MAX_SCENES = 24
SCRIPT_EXPAND_CHAR_THRESHOLD = 500
MIN_SCRIPT_CHARS = 8

LONG_VIDEO_CHAPTER_ANALYZE_SYSTEM_PROMPT = CHAPTER_PLAN_SYSTEM
LONG_VIDEO_CHAPTER_CHUNK_SYSTEM_PROMPT = CHAPTER_CHUNK_SYSTEM
LONG_VIDEO_CHAPTER_REDUCE_SYSTEM_PROMPT = CHAPTER_REDUCE_SYSTEM
LONG_VIDEO_CHAPTER_ROSTER_SYSTEM_PROMPT = CHAPTER_ROSTER_SYSTEM


@dataclass(frozen=True)
class ChapterAnalyzeResult:
    chapter_title: str
    synopsis: str
    mood: str
    character_anchor: str
    style_anchor: str
    beat_sheet: list[str]
    llm_calls: int


def needs_script_expand(text: str) -> bool:
    return len((text or "").strip()) < SCRIPT_EXPAND_CHAR_THRESHOLD


def validate_script_text(text: str) -> str:
    raw = (text or "").strip()
    if not raw:
        raise ValueError("script text is empty")
    if len(raw) > MAX_CHAPTER_CHARS:
        raise ValueError(
            f"script text exceeds {MAX_CHAPTER_CHARS} characters; shorten or split into smaller sections"
        )
    if len(raw) < MIN_SCRIPT_CHARS:
        raise ValueError("script is too short; add a few sentences describing the story")
    return raw


def validate_chapter_text(text: str, *, source_mode: Literal["brief", "chapter"] = "chapter") -> str:
    del source_mode
    return validate_script_text(text)


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


def parse_structured_beat(beat_raw: str) -> tuple[str, str]:
    """Parse ``title | shot | location | visual`` from JSON-derived beat lines."""
    raw = (beat_raw or "").strip()
    if not raw:
        raise ValueError("beat line is empty")
    if "|" not in raw:
        raise ValueError(f"beat line missing pipe fields: {raw[:80]}")
    parts = [p.strip() for p in raw.split("|")]
    if len(parts) >= 4:
        title = parts[0]
        shot = re.sub(r"^(?:景别|shot\s*size)[:：]\s*", "", parts[1], flags=re.I).strip()
        location = re.sub(r"^(?:地点|location|场景)[:：]\s*", "", parts[2], flags=re.I).strip()
        visual = "|".join(parts[3:]).strip()
        beat = f"【{shot}】{location}，{visual}" if location else f"【{shot}】{visual}"
        return title, beat
    if len(parts) == 3:
        title, shot, visual = parts[0], parts[1], parts[2]
        return title, f"【{shot}】{visual}"
    raise ValueError(f"beat line has unexpected field count ({len(parts)}): {raw[:80]}")


def _stable_id(prefix: str, key: str) -> str:
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:10]
    return f"{prefix}_{digest}"


def _format_look_body(*, role: str, appearance: str, wardrobe: str, locale: str) -> str:
    loc = locale if locale in ("zh", "en") else "zh"
    role = (role or "").strip()
    appearance = (appearance or "").strip()
    wardrobe = (wardrobe or "").strip()
    if loc == "zh":
        parts: list[str] = []
        if role:
            parts.append(f"定位：{role}")
        if appearance:
            parts.append(f"外貌：{appearance}")
        if wardrobe:
            parts.append(f"服装：{wardrobe}")
        return " | ".join(parts) if parts else appearance or wardrobe
    parts = []
    if role:
        parts.append(f"Role: {role}")
    if appearance:
        parts.append(f"Appearance: {appearance}")
    if wardrobe:
        parts.append(f"Wardrobe: {wardrobe}")
    return " | ".join(parts) if parts else appearance or wardrobe


def beat_to_sheet_line(beat: BeatSchema) -> str:
    from backend.engine.llm.storyboard_cast import strip_name_look_tags

    title = (beat.title or "").strip()
    shot = beat.shot_size.strip()
    location = beat.location.strip()
    narrative = strip_name_look_tags(beat.narrative.strip())
    if title:
        return f"{title} | {shot} | {location} | {narrative}"
    return f" | {shot} | {location} | {narrative}"


def sanitize_beat_sheet(beat_sheet: list[str]) -> list[str]:
    """Strip inline Name（…） tags from beat narratives (names-only policy)."""
    from backend.engine.llm.storyboard_cast import strip_name_look_tags

    out: list[str] = []
    for raw in beat_sheet:
        title, shot, location, narrative = _split_beat_fields_for_sanitize(raw)
        narrative = strip_name_look_tags(narrative)
        if title:
            out.append(f"{title} | {shot} | {location} | {narrative}")
        else:
            out.append(f" | {shot} | {location} | {narrative}")
    return out


def _split_beat_fields_for_sanitize(beat_raw: str) -> tuple[str, str, str, str]:
    raw = (beat_raw or "").strip()
    title, beat_body = parse_structured_beat(raw)
    shot_size = ""
    location = ""
    narrative = beat_body
    if "|" in raw:
        parts = [p.strip() for p in raw.split("|")]
        if len(parts) >= 4:
            title = parts[0] or title
            shot_size = parts[1]
            location = parts[2]
            narrative = parts[3]
        elif len(parts) == 3:
            shot_size, location, narrative = parts[0], parts[1], parts[2]
    return title, shot_size, location, narrative


def roster_from_character_rows(
    rows: list,
    *,
    style: str,
    locale: str,
    beat_sheet: list[str] | None = None,
) -> tuple[list[StoryboardCharacter], str, str]:
    from backend.engine.llm.storyboard import normalize_storyboard_locale

    loc = normalize_storyboard_locale(locale)
    default_label = _default_look_label(loc)
    characters: list[StoryboardCharacter] = []
    for row in rows:
        name = row.name.strip()
        if not name:
            raise ValueError("chapter analyze JSON has an empty character name")
        looks: list[CharacterLook] = []
        for look_i, lk in enumerate(row.looks):
            label = normalize_look_label(
                lk.label,
                locale=loc,
                name=name,
                wardrobe=lk.wardrobe,
                beat_sheet=beat_sheet,
                look_index=look_i,
            )
            if not label:
                label = default_label
            body = _format_look_body(
                role=lk.role,
                appearance=lk.appearance,
                wardrobe=lk.wardrobe,
                locale=loc,
            )
            if not body:
                raise ValueError(f"chapter analyze JSON character look '{name}/{label}' has empty body")
            looks.append(
                CharacterLook(
                    id=_stable_id("look", f"{name}|{label}"),
                    label=label,
                    body=body,
                    role=(lk.role or "").strip(),
                )
            )
        characters.append(
            StoryboardCharacter(
                id=_stable_id("char", name),
                name=name,
                looks=looks,
                default_look_id=looks[0].id,
            )
        )
    style_text = (style or "").strip()
    anchor = format_character_roster(characters, style_text, locale=loc)
    return characters, style_text, anchor


def roster_from_analyze_payload(
    payload: ChapterAnalyzeSchema,
    *,
    locale: str,
) -> tuple[list[StoryboardCharacter], str, str]:
    beat_sheet = [beat_to_sheet_line(b) for b in payload.beats]
    return roster_from_character_rows(
        payload.characters,
        style=(payload.style or "").strip(),
        locale=locale,
        beat_sheet=beat_sheet,
    )


def roster_from_roster_payload(
    payload: ChapterRosterSchema,
    *,
    locale: str,
    style_fallback: str = "",
    beat_sheet: list[str] | None = None,
) -> tuple[list[StoryboardCharacter], str, str]:
    style = (payload.style or style_fallback or "").strip()
    return roster_from_character_rows(
        payload.characters,
        style=style,
        locale=locale,
        beat_sheet=beat_sheet,
    )


def merge_plan_and_roster(
    plan: ChapterPlanSchema,
    roster: ChapterRosterSchema,
) -> ChapterAnalyzeSchema:
    style = (roster.style or plan.style or "").strip()
    return ChapterAnalyzeSchema(
        synopsis=plan.synopsis.strip(),
        mood=plan.mood.strip(),
        style=style,
        characters=roster.characters,
        beats=plan.beats,
    )


def parse_chapter_plan_json(text: str) -> ChapterPlanSchema:
    try:
        data = extract_json_object(text)
        return ChapterPlanSchema.model_validate(data)
    except ValidationError as exc:
        raise ValueError(f"chapter plan JSON schema invalid: {exc}") from exc


def parse_chapter_roster_json(text: str) -> ChapterRosterSchema:
    try:
        data = extract_json_object(text)
        return ChapterRosterSchema.model_validate(data)
    except ValidationError as exc:
        raise ValueError(f"chapter roster JSON schema invalid: {exc}") from exc


def parse_chapter_plan_response(
    text: str,
) -> tuple[str, str, list[str], str]:
    """Parse plan pass JSON into synopsis, mood, beat lines, style."""
    payload = parse_chapter_plan_json(text)
    beats = [beat_to_sheet_line(b) for b in payload.beats]
    return payload.synopsis.strip(), payload.mood.strip(), beats, (payload.style or "").strip()


def parse_chapter_analyze_json(text: str) -> ChapterAnalyzeSchema:
    try:
        data = extract_json_object(text)
        return ChapterAnalyzeSchema.model_validate(data)
    except ValidationError as exc:
        raise ValueError(f"chapter analyze JSON schema invalid: {exc}") from exc


def parse_chapter_chunk_json(text: str) -> list[str]:
    try:
        payload = ChapterChunkSchema.model_validate(extract_json_object(text))
    except ValidationError as exc:
        raise ValueError(f"chapter chunk JSON schema invalid: {exc}") from exc
    return [beat_to_sheet_line(b) for b in payload.beats]


def parse_chapter_analyze_response(
    text: str,
    *,
    locale: str = "zh",
) -> tuple[str, str, str, list[str], str]:
    """Parse full chapter analyze JSON into synopsis, mood, anchor, beat lines, style."""
    payload = parse_chapter_analyze_json(text)
    _, style, anchor = roster_from_analyze_payload(payload, locale=locale)
    if len(anchor.strip()) < 12:
        raise ValueError("chapter analyze JSON missing usable characters/style anchor")
    beats = [beat_to_sheet_line(b) for b in payload.beats]
    return payload.synopsis.strip(), payload.mood.strip(), anchor, beats, style


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


def clamp_scene_count(count: int, *, max_scenes: int = MAX_SCENES) -> int:
    n = int(count)
    if n < MIN_SCENES:
        raise ValueError(
            f"chapter analysis found {n} visual scene(s); need at least {MIN_SCENES}. "
            "Try a longer chapter or add more descriptive prose."
        )
    cap = max(MIN_SCENES, int(max_scenes))
    return min(n, cap)


def _beats_similar(a: str, b: str) -> bool:
    na = re.sub(r"\s+", "", a.lower())
    nb = re.sub(r"\s+", "", b.lower())
    if not na or not nb:
        return False
    if na == nb:
        return True
    short, long = (na, nb) if len(na) <= len(nb) else (nb, na)
    return len(short) >= 12 and short in long


def _paragraph_scene_floor(text: str) -> int:
    paragraphs = [
        p.strip()
        for p in re.split(r"\n\s*\n", (text or "").strip())
        if len(p.strip()) >= 20
    ]
    if not paragraphs:
        return MIN_SCENES
    return min(MAX_SCENES, max(MIN_SCENES, len(paragraphs)))


def _beat_count_hint(target_shot_count: int | None, narrative_budget: str) -> str:
    if target_shot_count is None:
        return (
            f"Target **beats** length: {MIN_SCENES}-{MAX_SCENES} "
            f"(about one beat per major script section).\n"
        )
    n = max(MIN_SCENES, min(MAX_SCENES, int(target_shot_count)))
    return (
        f"Suggested beat count ~{n} (soft — adjust for story needs, max {MAX_SCENES}).\n"
        f"Narrative budget: {narrative_budget}.\n"
    )


def _format_beats_for_roster_user(beat_sheet: list[str]) -> str:
    lines: list[str] = []
    for index, beat in enumerate(beat_sheet, start=1):
        line = (beat or "").strip()
        if line:
            lines.append(f"{index}. {line}")
    return "\n".join(lines)


def _roster_user_message(
    *,
    title: str,
    script: str,
    synopsis: str,
    mood: str,
    style: str,
    beat_sheet: list[str],
    locale_block: str,
) -> str:
    beat_block = _format_beats_for_roster_user(beat_sheet)
    mood_line = f"Mood: {mood.strip()}\n" if mood.strip() else ""
    style_line = f"Planned style: {style.strip()}\n" if style.strip() else ""
    return (
        f"Script title: {title or '(untitled)'}\n\n"
        f"Synopsis:\n{synopsis.strip()}\n\n"
        f"{mood_line}"
        f"{style_line}\n"
        f"Approved shot beats (fixed — build roster from these; do not rewrite beats):\n"
        f"{beat_block}\n\n"
        f"Before finishing JSON, ensure **characters** includes every full person name "
        f"that appears in the beat list above or in the source script.\n\n"
        f"Source script:\n{script.strip()}"
        f"{locale_block}"
    )


def _invoke_and_parse_plan(
    *,
    chat_fn: Callable[..., Any],
    system: str,
    user: str,
    max_tokens: int,
    think_apply: Callable[[str], str],
) -> tuple[str, str, list[str], str, int]:
    resp = invoke_text_chat(
        chat_fn,
        system=system,
        user=user,
        max_tokens=max_tokens,
        think_apply=think_apply,
    )
    synopsis, mood, beats, style = parse_chapter_plan_response(resp)
    return synopsis, mood, beats, style, 1


def _invoke_and_parse_roster(
    *,
    chat_fn: Callable[..., Any],
    user: str,
    max_tokens: int,
    think_apply: Callable[[str], str],
    locale: str,
    style_fallback: str,
    beat_sheet: list[str] | None = None,
) -> tuple[str, str, int]:
    llm_calls = 0
    correction = ""
    last_error: ValueError | None = None
    for _attempt in range(2):
        user_msg = user + correction
        resp = invoke_text_chat(
            chat_fn,
            system=CHAPTER_ROSTER_SYSTEM,
            user=user_msg,
            max_tokens=max_tokens,
            think_apply=think_apply,
        )
        llm_calls += 1
        try:
            payload = parse_chapter_roster_json(resp)
        except ValueError as exc:
            last_error = exc
            correction = (
                f"\n\nPrevious JSON failed validation: {exc}\n"
                "Fix the JSON. Each character must use a **looks** array (never a singular **look** field).\n"
            )
            continue
        _, style, anchor = roster_from_roster_payload(
            payload,
            locale=locale,
            style_fallback=style_fallback,
            beat_sheet=beat_sheet,
        )
        if len(anchor.strip()) < 12:
            raise ValueError("chapter roster JSON missing usable characters/style anchor")
        return anchor, style, llm_calls
    if last_error is not None:
        raise last_error
    raise ValueError("chapter roster JSON parse failed")


def _invoke_and_parse_analyze(
    *,
    chat_fn: Callable[..., Any],
    system: str,
    user: str,
    max_tokens: int,
    think_apply: Callable[[str], str],
    locale: str,
    script: str,
    locale_block: str,
    title: str,
    token_budget: Callable[[int], int],
) -> tuple[str, str, str, list[str], str, int]:
    synopsis, mood, beat_sheet, style_anchor, n_plan = _invoke_and_parse_plan(
        chat_fn=chat_fn,
        system=system,
        user=user,
        max_tokens=max_tokens,
        think_apply=think_apply,
    )
    roster_user = _roster_user_message(
        title=title,
        script=script,
        synopsis=synopsis,
        mood=mood,
        style=style_anchor,
        beat_sheet=beat_sheet,
        locale_block=locale_block,
    )
    character_anchor, style_resolved, n_roster = _invoke_and_parse_roster(
        chat_fn=chat_fn,
        user=roster_user,
        max_tokens=token_budget(2400),
        think_apply=think_apply,
        locale=locale,
        style_fallback=style_anchor,
        beat_sheet=beat_sheet,
    )
    return synopsis, mood, character_anchor, beat_sheet, style_resolved, n_plan + n_roster


def _invoke_and_parse_chunk(
    *,
    chat_fn: Callable[..., Any],
    system: str,
    user: str,
    max_tokens: int,
    think_apply: Callable[[str], str],
) -> tuple[list[str], int]:
    resp = invoke_text_chat(
        chat_fn,
        system=system,
        user=user,
        max_tokens=max_tokens,
        think_apply=think_apply,
    )
    return parse_chapter_chunk_json(resp), 1


def _minimum_required_beats(para_floor: int) -> int:
    """Soft lower bound — about 65% of major script sections, at least MIN_SCENES."""
    if para_floor <= MIN_SCENES:
        return MIN_SCENES
    return max(MIN_SCENES, min(para_floor, (para_floor * 13 + 9) // 20))


def run_chapter_analyze(
    *,
    chapter_text: str,
    chapter_title: str = "",
    locale: str = "zh",
    target_shot_count: int | None = None,
    narrative_budget: str = "standard",
    chat_fn: Callable[..., Any],
    think_apply: Callable[[str], str] | None = None,
    token_budget: Callable[[int], int] | None = None,
    source_mode: Literal["brief", "chapter"] | None = None,
) -> ChapterAnalyzeResult:
    del source_mode
    from backend.engine.llm.prompts.locale import chapter_json_user_locale_block
    from backend.engine.llm.storyboard import normalize_storyboard_locale

    raw = validate_script_text(chapter_text)
    loc = normalize_storyboard_locale(locale)
    locale_block = chapter_json_user_locale_block(loc)
    apply_think = think_apply or (lambda t: t)
    budget = token_budget or (lambda b: b)

    llm_calls = 0
    title = (chapter_title or "").strip()
    max_scenes = MAX_SCENES
    beat_hint = _beat_count_hint(target_shot_count, narrative_budget)

    chunks = split_chapter_chunks(raw)
    style_anchor = ""

    if len(chunks) == 1:
        user = (
            f"Script title: {title or '(untitled)'}\n\n"
            f"{beat_hint}"
            f"Text:\n{raw[:CHAPTER_CHUNK_CHARS * 2] if len(raw) <= CHAPTER_CHUNK_CHARS * 2 else raw}"
            + locale_block
        )
        synopsis, mood, character_anchor, beat_sheet, style_anchor, n_calls = _invoke_and_parse_analyze(
            chat_fn=chat_fn,
            system=CHAPTER_PLAN_SYSTEM,
            user=user,
            max_tokens=budget(4000),
            think_apply=apply_think,
            locale=loc,
            script=raw,
            locale_block=locale_block,
            title=title,
            token_budget=budget,
        )
        llm_calls += n_calls
    else:
        partial_lists: list[list[str]] = []
        for idx, chunk in enumerate(chunks):
            user = (
                f"Excerpt {idx + 1}/{len(chunks)} of «{title or 'untitled'}»:\n\n{chunk}"
                + locale_block
            )
            beats, n_calls = _invoke_and_parse_chunk(
                chat_fn=chat_fn,
                system=CHAPTER_CHUNK_SYSTEM,
                user=user,
                max_tokens=budget(1400),
                think_apply=apply_think,
            )
            llm_calls += n_calls
            partial_lists.append(beats)
        merged = merge_partial_beats(partial_lists)
        beat_block = "\n".join(f"- {b}" for b in merged[: max_scenes + 4])
        reduce_user = (
            f"Script title: {title or '(untitled)'}\n\n"
            f"{beat_hint}"
            f"Partial beats from {len(chunks)} excerpts ({len(merged)} lines):\n{beat_block}\n\n"
            f"Merge into one JSON object with synopsis, mood, style, and "
            f"{MIN_SCENES}-{max_scenes} ordered beats (no characters field)."
            + locale_block
        )
        synopsis, mood, character_anchor, beat_sheet, style_anchor, n_calls = _invoke_and_parse_analyze(
            chat_fn=chat_fn,
            system=CHAPTER_REDUCE_SYSTEM,
            user=reduce_user,
            max_tokens=budget(3600),
            think_apply=apply_think,
            locale=loc,
            script=raw,
            locale_block=locale_block,
            title=title,
            token_budget=budget,
        )
        llm_calls += n_calls

    scene_count = clamp_scene_count(len(beat_sheet), max_scenes=max_scenes)
    beat_sheet = sanitize_beat_sheet(beat_sheet[:scene_count])

    para_floor = _paragraph_scene_floor(raw)
    min_beats = _minimum_required_beats(para_floor)
    if len(beat_sheet) < min_beats:
        raise ValueError(
            f"chapter analysis returned {len(beat_sheet)} visual scene beat(s) but the script "
            f"needs at least {min_beats} (~{para_floor} major sections); re-parse or add more descriptive prose."
        )

    return ChapterAnalyzeResult(
        chapter_title=title,
        synopsis=synopsis,
        mood=mood,
        character_anchor=character_anchor,
        style_anchor=style_anchor,
        beat_sheet=beat_sheet,
        llm_calls=llm_calls,
    )
