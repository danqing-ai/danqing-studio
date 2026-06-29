"""Optional LLM shot repair when rule repair still fails validation."""
from __future__ import annotations

import json
from typing import Any, Callable

from pydantic import ValidationError

from backend.engine.llm.chat_invoke import invoke_text_chat
from backend.engine.llm.json_output import extract_json_object
from backend.engine.llm.llm_retry import invoke_text_chat_with_feedback
from backend.engine.llm.prompts.system import CHAPTER_SHOT_REPAIR_SYSTEM
from backend.engine.llm.schemas.long_video import ShotRepairBatchSchema


def _issues_summary(issues: list[Any], limit: int = 6) -> str:
    lines = [getattr(i, "message", str(i)) for i in issues[:limit]]
    return "\n".join(f"- {x}" for x in lines if x)


def repair_shots_with_llm(
    shots: list[dict[str, Any]],
    *,
    issues: list[Any],
    character_anchor: str,
    chat_fn: Callable[..., Any],
    think_apply: Callable[[str], str],
    max_tokens: int,
    locale_block: str = "",
) -> tuple[list[dict[str, Any]], int]:
    if not shots or not issues:
        return shots, 0
    compact = [
        {
            "order": i,
            "segment_role": s.get("segment_role"),
            "duration_sec": s.get("duration_sec"),
            "first_frame_visibility": s.get("first_frame_visibility"),
            "end_visibility": s.get("end_visibility"),
            "characters_on_screen": s.get("characters_on_screen"),
            "start_visual_prompt": (s.get("start_visual_prompt") or "")[:240],
            "video_prompt": (s.get("video_prompt") or "")[:240],
        }
        for i, s in enumerate(shots)
    ]
    user = (
        "Fix ONLY the listed validation issues. Return JSON with repairs array.\n\n"
        f"Issues:\n{_issues_summary(issues)}\n\n"
        f"Character anchor:\n{character_anchor.strip()[:1200]}\n\n"
        f"Current shots:\n{json.dumps(compact, ensure_ascii=False)}\n"
        + locale_block
    )

    def _validate(resp: str) -> tuple[bool, str]:
        try:
            payload = ShotRepairBatchSchema.model_validate(extract_json_object(resp))
        except (ValueError, ValidationError) as exc:
            return False, str(exc)
        missing = {r.order for r in payload.repairs} - set(range(len(shots)))
        if missing:
            return False, f"repairs missing order indices: {sorted(missing)}"
        return True, ""

    try:
        resp, calls = invoke_text_chat_with_feedback(
            chat_fn,
            system=CHAPTER_SHOT_REPAIR_SYSTEM,
            user=user,
            max_tokens=max_tokens,
            think_apply=think_apply,
            validate=_validate,
            max_attempts=2,
        )
        payload = ShotRepairBatchSchema.model_validate(extract_json_object(resp))
    except (ValueError, ValidationError):
        return shots, 0

    out = [dict(s) for s in shots]
    for rep in payload.repairs:
        idx = int(rep.order)
        if idx < 0 or idx >= len(out):
            continue
        row = out[idx]
        if rep.first_frame_visibility:
            row["first_frame_visibility"] = rep.first_frame_visibility
        if rep.end_visibility:
            row["end_visibility"] = rep.end_visibility
        if rep.characters_on_screen:
            row["characters_on_screen"] = list(rep.characters_on_screen)
        if rep.start_visual_prompt.strip():
            row["start_visual_prompt"] = rep.start_visual_prompt.strip()
            row["visual_prompt"] = rep.start_visual_prompt.strip()
        if rep.video_prompt.strip():
            row["video_prompt"] = rep.video_prompt.strip()
            row["motion_prompt"] = rep.video_prompt.strip()
        if rep.first_frame_requirement.strip():
            row["first_frame_requirement"] = rep.first_frame_requirement.strip()
    return out, calls
