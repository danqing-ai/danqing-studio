"""Pass 1 — script decompose (beats + cast + scenes in one artifact)."""
from __future__ import annotations

from typing import Any, Callable

from pydantic import ValidationError

from backend.engine.llm.chapter_analyze import (
    CHAPTER_CHUNK_CHARS,
    merge_partial_beats,
    split_chapter_chunks,
    validate_script_text,
)
from backend.engine.llm.json_output import extract_json_object
from backend.engine.llm.prompts.locale import chapter_json_user_locale_block
from backend.engine.llm.script_parse.prompts import SCRIPT_DECOMPOSE_SYSTEM
from backend.engine.llm.script_parse.review import invoke_pass_with_review, validate_pydantic_json
from backend.engine.llm.script_parse.rules import (
    assign_character_scene_ids,
    canonicalize_script_locations,
    location_matches,
    resolve_canonical_scene_name,
)
from backend.engine.llm.script_parse.schemas import (
    DecomposeLLMSchema,
    DecomposeResult,
    ScriptArtifact,
    ScriptBeatArtifact,
)
from backend.engine.llm.storyboard import normalize_storyboard_locale

ProgressFn = Callable[[str, str], None]


def _canonicalize_decompose_beats(payload: DecomposeLLMSchema) -> DecomposeLLMSchema:
    scene_names = [s.name.strip() for s in payload.scenes if (s.name or "").strip()]
    if not scene_names:
        return payload
    beats = []
    for beat in payload.beats:
        canon = resolve_canonical_scene_name(beat.location, scene_names)
        beats.append(beat.model_copy(update={"location": canon or beat.location.strip()}))
    return payload.model_copy(update={"beats": beats})


def _validate_decompose_payload(payload: DecomposeLLMSchema, locale: str) -> tuple[bool, str]:
    payload = _canonicalize_decompose_beats(payload)
    issues: list[str] = []
    protagonists = [c for c in payload.characters if c.role == "protagonist"]
    if len(protagonists) != 1:
        issues.append(f"expected exactly 1 protagonist, got {len(protagonists)}")
    for ch in payload.characters:
        for lk in ch.looks:
            if not (lk.body or "").strip():
                issues.append(f"character {ch.name!r} look {lk.label!r} has empty body")
    scene_names = [s.name.strip() for s in payload.scenes if (s.name or "").strip()]
    for beat in payload.beats:
        if not beat.enhancement_cues:
            issues.append(f"beat {beat.index}: enhancement_cues required")
        loc = beat.location.strip()
        if loc and scene_names and not any(location_matches(loc, sn) for sn in scene_names):
            issues.append(f"beat location {loc!r} has no matching scene entity")
    if issues:
        return False, "\n".join(issues[:12])
    return True, ""


def _llm_to_artifact(payload: DecomposeLLMSchema, *, title: str) -> ScriptArtifact:
    beats = [
        ScriptBeatArtifact(
            index=i,
            title=b.title,
            location=b.location,
            narrative=b.narrative,
            enhancement_cues=list(b.enhancement_cues),
            suggested_shot_size=b.suggested_shot_size,
            estimated_duration_sec=b.estimated_duration_sec,
        )
        for i, b in enumerate(payload.beats)
    ]
    art = ScriptArtifact(
        title=title or payload.title,
        synopsis=payload.synopsis,
        mood=payload.mood,
        style_anchor=payload.style_anchor,
        beats=beats,
        characters=payload.characters,
        scenes=payload.scenes,
    )
    return assign_character_scene_ids(canonicalize_script_locations(art))


def run_script_decompose(
    *,
    script_text: str,
    title: str = "",
    locale: str = "zh",
    chat_fn: Callable[..., Any],
    think_apply: Callable[[str], str],
    token_budget: Callable[[int], int],
    on_progress: ProgressFn | None = None,
) -> DecomposeResult:
    def progress(phase: str, msg: str = "") -> None:
        if on_progress:
            on_progress(phase, msg)

    loc = normalize_storyboard_locale(locale)
    locale_block = chapter_json_user_locale_block(loc)
    text = validate_script_text(script_text)
    llm_calls = 0
    phases = ["decompose"]

    chunks = split_chapter_chunks(text, chunk_size=CHAPTER_CHUNK_CHARS)

    if len(chunks) == 1:
        user = f"Title: {title or '(untitled)'}\n\nScript:\n{text}\n{locale_block}"

        def validate(resp: str) -> tuple[bool, str]:
            ok, fb, payload = validate_pydantic_json(resp, DecomposeLLMSchema)
            if not ok or payload is None:
                return ok, fb
            return _validate_decompose_payload(payload, loc)

        progress("decompose", "decompose")
        resp, n = invoke_pass_with_review(
            chat_fn,
            system=SCRIPT_DECOMPOSE_SYSTEM,
            user=user,
            max_tokens=token_budget(4000),
            think_apply=think_apply,
            validate=validate,
            max_attempts=2,
            pass_name="decompose",
            on_progress=on_progress,
        )
        llm_calls += n
        ok, fb, payload = validate_pydantic_json(resp, DecomposeLLMSchema)
        if not ok or payload is None:
            raise ValueError(f"decompose validation failed: {fb}")
        ok2, fb2 = _validate_decompose_payload(payload, loc)
        if not ok2:
            raise ValueError(f"decompose rule validation failed: {fb2}")
        artifact = _llm_to_artifact(payload, title=title)
        return DecomposeResult(artifact=artifact, llm_calls=llm_calls, phases=phases)

    # Map-reduce for long scripts: chunk beats only, then full decompose on merged outline
    from backend.engine.llm.prompts.system import CHAPTER_CHUNK_SYSTEM, CHAPTER_REDUCE_SYSTEM

    partial_beats: list[list[str]] = []
    for chunk in chunks:
        user = f"Excerpt:\n{chunk}\n{locale_block}"

        def validate_chunk(resp: str) -> tuple[bool, str]:
            try:
                data = extract_json_object(resp)
                beats = data.get("beats") or []
                if not beats:
                    return False, "beats array empty"
                return True, ""
            except ValueError as exc:
                return False, str(exc)

        resp, n = invoke_pass_with_review(
            chat_fn,
            system=CHAPTER_CHUNK_SYSTEM,
            user=user,
            max_tokens=token_budget(2000),
            think_apply=think_apply,
            validate=validate_chunk,
            max_attempts=2,
            pass_name="decompose_chunk",
            on_progress=on_progress,
        )
        llm_calls += n
        data = extract_json_object(resp)
        lines = []
        for b in data.get("beats") or []:
            if isinstance(b, dict):
                lines.append(
                    f"{b.get('title', '')} | {b.get('shot_size', '')} | {b.get('location', '')} | {b.get('narrative', '')}"
                )
        partial_beats.append(lines)

    merged_outline = "\n\n".join(merge_partial_beats(partial_beats))
    reduce_user = (
        f"Title: {title or '(untitled)'}\n\n"
        f"Merged beat outline from long script:\n{merged_outline}\n\n"
        f"Full script (reference):\n{text[:8000]}\n{locale_block}"
    )

    def validate_full(resp: str) -> tuple[bool, str]:
        ok, fb, payload = validate_pydantic_json(resp, DecomposeLLMSchema)
        if not ok or payload is None:
            return ok, fb
        return _validate_decompose_payload(payload, loc)

    progress("decompose", "decompose_reduce")
    resp, n = invoke_pass_with_review(
        chat_fn,
        system=SCRIPT_DECOMPOSE_SYSTEM,
        user=reduce_user,
        max_tokens=token_budget(4500),
        think_apply=think_apply,
        validate=validate_full,
        max_attempts=2,
        pass_name="decompose",
        on_progress=on_progress,
    )
    llm_calls += n
    ok, fb, payload = validate_pydantic_json(resp, DecomposeLLMSchema)
    if not ok or payload is None:
        raise ValueError(f"decompose validation failed: {fb}")
    artifact = _llm_to_artifact(payload, title=title)
    phases.append("decompose_reduce")
    return DecomposeResult(artifact=artifact, llm_calls=llm_calls, phases=phases)
