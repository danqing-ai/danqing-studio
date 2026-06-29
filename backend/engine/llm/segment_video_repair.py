"""Re-invoke segment_video for beat groups that failed motion quality checks."""
from __future__ import annotations

from typing import Any, Callable

from backend.engine.common.long_video.segment_video_quality import (
    SegmentVideoValidationResult,
    validate_segment_video_prompts,
)


def repair_segment_video_groups(
    segments: list[Any],
    video_by_index: dict[int, str],
    *,
    chat_fn: Callable[..., Any],
    think_apply: Callable[[str], str],
    max_tokens: int,
    synopsis: str,
    style_anchor: str,
    character_anchor: str,
    mood: str = "",
    locale_block: str,
    max_groups: int = 6,
) -> tuple[dict[int, str], int, SegmentVideoValidationResult]:
    """Re-generate video_prompt for beat groups with duplicate/undifferentiated motion."""
    from backend.engine.llm.chapter_segment_plan import _invoke_segment_video_batch

    validation = validate_segment_video_prompts(
        segments,
        video_by_index,
        style_anchor=style_anchor,
        character_anchor=character_anchor,
    )
    if validation.ok:
        return video_by_index, 0, validation

    group_ids: list[str] = []
    seen: set[str] = set()
    for issue in validation.issues:
        gid = issue.message.split(":", 1)[0].strip()
        if gid.startswith("beat_") and gid not in seen:
            seen.add(gid)
            group_ids.append(gid)

    out = dict(video_by_index)
    calls = 0
    for gid in group_ids[: max(1, int(max_groups))]:
        batch = [s for s in segments if str(s.segment_group_id or "") == gid]
        if not batch:
            continue
        patch, n = _invoke_segment_video_batch(
            chat_fn=chat_fn,
            think_apply=think_apply,
            max_tokens=max_tokens,
            segments=batch,
            synopsis=synopsis,
            mood=mood,
            style_anchor=style_anchor,
            character_anchor=character_anchor,
            locale_block=locale_block,
            repair_feedback=(
                f"Prior video_prompt rows for {gid} failed motion differentiation:\n"
                f"{validation.feedback}\n"
                "Regenerate ONLY the listed indices with clearly different motion per role."
            ),
        )
        out.update(patch)
        calls += n

    final = validate_segment_video_prompts(
        segments,
        out,
        style_anchor=style_anchor,
        character_anchor=character_anchor,
    )
    return out, calls, final
