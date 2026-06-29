"""Layer 1 — story graph: beat-level visibility timeline (LLM + rule fallback)."""
from __future__ import annotations

from typing import Any, Callable

from pydantic import ValidationError

from backend.engine.llm.chapter_segment_plan import _split_beat_fields
from backend.engine.llm.json_output import extract_json_object
from backend.engine.llm.prompts.system import CHAPTER_STORY_GRAPH_SYSTEM
from backend.engine.llm.schemas.long_video import StoryGraphBatchSchema

StoryGraphByBeat = dict[int, dict[str, Any]]


def _rule_story_graph(beat_sheet: list[str], character_anchor: str) -> StoryGraphByBeat:
    names: list[str] = []
    for line in (character_anchor or "").splitlines():
        line = line.strip()
        if line.startswith("-") or line.startswith("*"):
            head = line.lstrip("-* ").split("（")[0].split("(")[0].strip()
            if head and len(head) <= 24:
                names.append(head)
    primary = names[0] if names else ""
    out: StoryGraphByBeat = {}
    for i, beat_raw in enumerate(beat_sheet):
        _, shot_size, _loc, narrative = _split_beat_fields(beat_raw)
        on_screen = [n for n in names if n and n in narrative]
        if i == 0 and primary and primary not in on_screen:
            on_screen = [primary]
        wide = shot_size in ("远景", "全景", "大远景", "wide", "establishing")
        if not on_screen:
            start_vis, end_vis = "invisible", "invisible"
        elif wide:
            start_vis, end_vis = "silhouette", "partial"
        elif i == 0:
            start_vis, end_vis = "silhouette", "full_face"
        else:
            start_vis, end_vis = "partial", "full_face"
        out[i] = {
            "characters_on_screen": on_screen,
            "start_visibility": start_vis,
            "end_visibility": end_vis,
            "action_summary": narrative[:200],
        }
    return out


def run_story_graph_pass(
    *,
    beat_sheet: list[str],
    character_anchor: str,
    synopsis: str,
    locale_block: str,
    chat_fn: Callable[..., Any],
    think_apply: Callable[[str], str],
    max_tokens: int,
) -> tuple[StoryGraphByBeat, int]:
    rows: list[str] = []
    for i, beat_raw in enumerate(beat_sheet):
        title, shot_size, location, narrative = _split_beat_fields(beat_raw)
        rows.append(
            f"[{i}] title={title}\nshot_size={shot_size}\nlocation={location}\nnarrative={narrative}"
        )
    user = (
        f"Synopsis:\n{synopsis.strip()}\n\nBeats:\n"
        + "\n\n".join(rows)
        + f"\n\nCharacter roster:\n{character_anchor.strip()}\n"
        + locale_block
    )
    try:
        from backend.engine.llm.llm_retry import invoke_text_chat_with_feedback

        def _validate(resp: str) -> tuple[bool, str]:
            try:
                payload = StoryGraphBatchSchema.model_validate(extract_json_object(resp))
            except (ValueError, ValidationError) as exc:
                return False, str(exc)
            issues: list[str] = []
            for ev in payload.events:
                if int(ev.beat_index) == 0 and ev.characters_on_screen:
                    if ev.start_visibility == "invisible":
                        issues.append("beat_index=0 with characters must not start invisible")
            return (not issues, "\n".join(issues))

        resp, calls = invoke_text_chat_with_feedback(
            chat_fn,
            system=CHAPTER_STORY_GRAPH_SYSTEM,
            user=user,
            max_tokens=max_tokens,
            think_apply=think_apply,
            validate=_validate,
            max_attempts=2,
        )
        payload = StoryGraphBatchSchema.model_validate(extract_json_object(resp))
        out: StoryGraphByBeat = {}
        for ev in payload.events:
            out[int(ev.beat_index)] = {
                "characters_on_screen": list(ev.characters_on_screen),
                "start_visibility": ev.start_visibility,
                "end_visibility": ev.end_visibility,
                "action_summary": ev.action_summary.strip(),
            }
        if len(out) < len(beat_sheet):
            fallback = _rule_story_graph(beat_sheet, character_anchor)
            for i in range(len(beat_sheet)):
                out.setdefault(i, fallback.get(i, fallback[0]))
        return out, calls
    except (ValueError, ValidationError):
        return _rule_story_graph(beat_sheet, character_anchor), 0
