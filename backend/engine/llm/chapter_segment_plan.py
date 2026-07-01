"""Plan segment shots from narrative beats, then multi-pass LLM video + visual prompts."""
from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from pydantic import ValidationError

from backend.engine.common.long_video.constants import MAX_CLIP_SEC, MIN_CLIP_SEC
from backend.engine.common.long_video.plan import allocate_shot_durations
from backend.engine.llm.chat_invoke import invoke_text_chat
from backend.engine.llm.chapter_analyze import parse_structured_beat
from backend.engine.llm.json_output import extract_json_object
from backend.engine.llm.prompts.system import (
    CHAPTER_ANCHOR_SPLIT_SYSTEM,
    CHAPTER_ANCHOR_VISUAL_SYSTEM,
    CHAPTER_FACE_REACHABILITY_SYSTEM,
    CHAPTER_SEGMENT_VIDEO_SYSTEM,
    CHAPTER_START_VISUAL_SYSTEM,
)
from backend.engine.llm.schemas.long_video import (
    AnchorSplitBatchSchema,
    FaceReachabilityBatchSchema,
    SegmentAnchorVisualBatchSchema,
    SegmentStartVisualBatchSchema,
    SegmentVideoBatchSchema,
)
from backend.engine.llm.storyboard_cast import strip_name_look_tags

SegmentRole = Literal[
    "establishing",
    "pre_anchor",
    "face_anchor",
    "post_anchor",
    "keyframe",
    "tail_continuation",
]
FlfMode = Literal["none", "first_last", "continuation"]
StartFrameMode = Literal["keyframe", "prev_segment_tail", "anchor_link"]
FaceReachability = Literal["identity_critical", "establishing", "action_wide", "empty"]
ProgressFn = Callable[[str, str], None]

DEFAULT_MAX_CLIP_SEC = MAX_CLIP_SEC
SEGMENT_VIDEO_BATCH = 5
START_VISUAL_BATCH = 4
REACHABILITY_BATCH = 12
ANCHOR_SPLIT_BATCH = 8

# Rule-split first-frame constraints (locale-neutral; LLM passes translate via user locale block).
_FFR_OPENING_SILHOUETTE = "on-screen cast visible at least as silhouette at t=0"
_FFR_PRE_APPROACH = "approach framing toward anchor; cast visible in scene"
_FFR_FACE_ANCHOR = "readable face close-up at t=0 for identity lock"

_WIDE_SHOT_RE = re.compile(
    r"^(远景|全景|大远景|wide|ws|fs|establishing|long\s*shot|full\s*shot)$",
    re.I,
)
_CLOSE_SHOT_RE = re.compile(
    r"^(特写|近景|大特写|close|cu|mcu|close-up|closeup|medium\s*close)$",
    re.I,
)
_FAST_ACTION_RE = re.compile(r"击飞|秒杀|一闪|骤|快切|震飞|瞬间|突然|quick|flash|smash", re.I)


@dataclass(frozen=True)
class SubsegmentPlan:
    role: SegmentRole
    duration_sec: float
    shot_size: str
    flf_mode: FlfMode
    start_visibility: str = "full_face"
    end_visibility: str = "full_face"
    characters_on_screen: tuple[str, ...] = ()
    first_frame_requirement: str = ""


@dataclass(frozen=True)
class PlannedSegment:
    segment_index: int
    narrative_beat_index: int
    segment_group_id: str
    segment_group_index: int
    duration_sec: float
    segment_role: SegmentRole
    start_frame_mode: StartFrameMode
    flf_mode: FlfMode
    face_anchor_shot_id: str
    title: str
    shot_size: str
    location: str
    narrative: str
    start_visibility: str = "full_face"
    end_visibility: str = "full_face"
    characters_on_screen: tuple[str, ...] = ()
    first_frame_requirement: str = ""
    camera_zone_id: str = ""


@dataclass
class SegmentPipelineResult:
    shots: list[dict[str, Any]]
    llm_calls: int = 0
    phases: list[str] = field(default_factory=list)
    beat_budgets: dict[int, float] = field(default_factory=dict)


def _stable_shot_id(prefix: str, key: str) -> str:
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:10]
    return f"{prefix}_{digest}"


def _split_beat_fields(beat_raw: str) -> tuple[str, str, str, str]:
    """Return title, shot_size, location, narrative from pipe beat lines."""
    raw = (beat_raw or "").strip()
    title, beat_body = parse_structured_beat(raw)
    shot_size = ""
    location = ""
    narrative = beat_body
    if "|" in raw:
        parts = [p.strip() for p in raw.split("|")]
        if len(parts) >= 4:
            title = parts[0] or title
            shot_size = re.sub(r"^(?:景别|shot\s*size)[:：]\s*", "", parts[1], flags=re.I).strip()
            location = re.sub(r"^(?:地点|location|场景)[:：]\s*", "", parts[2], flags=re.I).strip()
            narrative = "|".join(parts[3:]).strip()
        elif len(parts) == 3:
            title = parts[0] or title
            shot_size = parts[1]
            narrative = parts[2]
    m = re.match(r"^【([^】]+)】(.*)$", narrative)
    if m and not shot_size:
        shot_size = m.group(1).strip()
        rest = m.group(2).strip()
        if "，" in rest:
            loc, body = rest.split("，", 1)
            if not location:
                location = loc.strip()
            narrative = body.strip() or rest
        else:
            narrative = rest
    return title, shot_size, location, strip_name_look_tags(narrative.strip() or beat_body)


def _split_duration_parts(total_sec: float, *, max_clip_sec: float) -> list[float]:
    from backend.engine.common.long_video.beat_budget import split_duration_parts

    return split_duration_parts(total_sec, max_clip_sec=max_clip_sec)


def _is_wide_shot(shot_size: str) -> bool:
    return bool(_WIDE_SHOT_RE.match((shot_size or "").strip()))


def _is_close_shot(shot_size: str) -> bool:
    return bool(_CLOSE_SHOT_RE.match((shot_size or "").strip()))


def _is_fast_action(narrative: str) -> bool:
    return bool(_FAST_ACTION_RE.search(narrative or ""))


def _rule_reachability(shot_size: str, narrative: str, *, beat_index: int = 0) -> FaceReachability:
    text = (narrative or "").strip()
    if not text or re.search(r"空镜|纯环境|无人物|establishing only|empty", text, re.I):
        if beat_index == 0 and re.search(r"[\u4e00-\u9fff]{2,}|[A-Za-z]{2,}", text):
            return "action_wide"
        return "empty"
    if _is_wide_shot(shot_size):
        if re.search(r"对白|说|喊|表情|特写|dialogue|speak|expression", text, re.I):
            return "identity_critical"
        return "action_wide"
    if re.search(r"对白|说|喊|表情|dialogue|speak|expression|反应", text, re.I):
        return "identity_critical"
    return "establishing"


def _rule_split_beat(
    *,
    beat_index: int,
    beat_dur: float,
    shot_size: str,
    reachability: FaceReachability,
    max_clip_sec: float,
    story_graph: dict[str, Any] | None = None,
    narrative: str = "",
) -> list[SubsegmentPlan]:
    """Return subsegments for one beat."""
    from backend.engine.common.long_video.shot_repair import normalize_subsegment_plans
    from backend.engine.common.long_video.visibility import clamp_vis_progression, vis_label, vis_rank

    graph = story_graph or {}
    g_start = vis_label(str(graph.get("start_visibility") or ""))
    g_end = vis_label(str(graph.get("end_visibility") or ""))
    chars = tuple(graph.get("characters_on_screen") or ())

    need_anchor = reachability in ("identity_critical", "action_wide")
    cap = max(MIN_CLIP_SEC, float(max_clip_sec))

    def _clamp_dur(d: float) -> float:
        return round(max(MIN_CLIP_SEC, min(cap, float(d))), 1)

    # Single-segment beats: close-up or very low budget
    if _is_close_shot(shot_size) or beat_dur <= 4.5:
        start_vis = g_start or ("silhouette" if beat_index == 0 else "partial")
        if beat_index == 0 and chars:
            start_vis = "silhouette" if vis_rank(start_vis) < vis_rank("silhouette") else start_vis
        return normalize_subsegment_plans(
            [
                SubsegmentPlan(
                    role="keyframe",
                    duration_sec=_clamp_dur(beat_dur),
                    shot_size=shot_size or "中景",
                    flf_mode="none",
                    start_visibility=start_vis,
                    end_visibility=g_end or "full_face",
                    characters_on_screen=chars,
                    first_frame_requirement=_FFR_OPENING_SILHOUETTE if beat_index == 0 else "",
                )
            ],
            max_clip_sec=cap,
        )

    if _is_fast_action(narrative) and beat_dur <= 6.0:
        return normalize_subsegment_plans(
            [
                SubsegmentPlan(
                    role="keyframe",
                    duration_sec=_clamp_dur(beat_dur),
                    shot_size=shot_size or "中景",
                    flf_mode="none",
                    start_visibility=g_start or "partial",
                    end_visibility=g_end or "partial",
                    characters_on_screen=chars,
                )
            ],
            max_clip_sec=cap,
        )

    if reachability == "empty":
        parts = _split_duration_parts(beat_dur, max_clip_sec=max_clip_sec)
        out: list[SubsegmentPlan] = []
        for i, p in enumerate(parts):
            role: SegmentRole = "establishing" if i == 0 else "tail_continuation"
            flf: FlfMode = "continuation" if i > 0 else "none"
            out.append(
                SubsegmentPlan(
                    role=role,
                    duration_sec=_clamp_dur(p),
                    shot_size=shot_size or "远景",
                    flf_mode=flf,
                    start_visibility="invisible",
                    end_visibility="invisible",
                )
            )
        return out

    if not need_anchor:
        parts = _split_duration_parts(beat_dur, max_clip_sec=max_clip_sec)
        out = []
        for i, p in enumerate(parts):
            role = "keyframe" if i == 0 else "tail_continuation"
            flf = "continuation" if i > 0 else "none"
            start_vis = g_start or ("silhouette" if beat_index == 0 and i == 0 else "partial")
            out.append(
                SubsegmentPlan(
                    role=role,
                    duration_sec=_clamp_dur(p),
                    shot_size=shot_size or "中景",
                    flf_mode=flf,
                    start_visibility=start_vis,
                    end_visibility=g_end or "full_face",
                    characters_on_screen=chars,
                    first_frame_requirement=_FFR_OPENING_SILHOUETTE if beat_index == 0 and i == 0 else "",
                )
            )
        return normalize_subsegment_plans(out, max_clip_sec=cap)

    anchor_dur = min(3.0, max(1.5, beat_dur * 0.25))
    remain = max(2.0, beat_dur - anchor_dur)
    use_pre = (_is_wide_shot(shot_size) or reachability == "action_wide") and beat_dur >= 6.0
    pre_dur = round(remain * 0.45, 1) if use_pre else 0.0
    post_dur = round(remain - pre_dur, 1) if beat_dur >= 7.0 else 0.0
    segs: list[SubsegmentPlan] = []
    prev_end = g_start or ("silhouette" if beat_index == 0 else "partial")
    if pre_dur >= 2.0:
        end_pre = "partial"
        segs.append(
            SubsegmentPlan(
                role="pre_anchor",
                duration_sec=_clamp_dur(pre_dur),
                shot_size=shot_size or "远景",
                flf_mode="none",
                start_visibility=prev_end,
                end_visibility=end_pre,
                characters_on_screen=chars,
                first_frame_requirement=_FFR_PRE_APPROACH,
            )
        )
        prev_end = end_pre
    face_start = clamp_vis_progression(prev_end, "partial")
    segs.append(
        SubsegmentPlan(
            role="face_anchor",
            duration_sec=_clamp_dur(anchor_dur),
            shot_size="特写",
            flf_mode="none",
            start_visibility=face_start,
            end_visibility=g_end or "full_face",
            characters_on_screen=chars,
            first_frame_requirement=_FFR_FACE_ANCHOR,
        )
    )
    if post_dur >= 2.0:
        segs.append(
            SubsegmentPlan(
                role="post_anchor",
                duration_sec=_clamp_dur(post_dur),
                shot_size="中景",
                flf_mode="none",
                start_visibility="full_face",
                end_visibility="partial",
                characters_on_screen=chars,
            )
        )
    if not segs:
        segs.append(
            SubsegmentPlan(
                role="face_anchor",
                duration_sec=_clamp_dur(beat_dur),
                shot_size="特写",
                flf_mode="none",
                start_visibility=clamp_vis_progression(prev_end, "partial"),
                end_visibility=g_end or "full_face",
                characters_on_screen=chars,
            )
        )
    return normalize_subsegment_plans(segs, max_clip_sec=cap)


def plan_segments_from_beats(
    beat_sheet: list[str],
    *,
    target_duration_sec: float,
    default_segment_sec: float,
    max_clip_sec: float = DEFAULT_MAX_CLIP_SEC,
) -> list[PlannedSegment]:
    """A0 — coarse duration split (one row per duration part, role=keyframe placeholder)."""
    if not beat_sheet:
        return []
    durations = allocate_shot_durations(
        scene_count=len(beat_sheet),
        target_duration_sec=max(float(default_segment_sec), float(target_duration_sec)),
        default_segment_sec=float(default_segment_sec),
        beat_texts=beat_sheet,
        max_sec=max_clip_sec,
    )
    planned: list[PlannedSegment] = []
    global_idx = 0
    for beat_i, beat_raw in enumerate(beat_sheet):
        title, shot_size, location, narrative = _split_beat_fields(beat_raw)
        beat_dur = float(durations[beat_i]) if beat_i < len(durations) else float(default_segment_sec)
        group_id = f"beat_{beat_i}"
        for part_j, part_dur in enumerate(_split_duration_parts(beat_dur, max_clip_sec=max_clip_sec)):
            mode: StartFrameMode = "keyframe" if part_j == 0 else "prev_segment_tail"
            role: SegmentRole = "tail_continuation" if part_j > 0 else "keyframe"
            planned.append(
                PlannedSegment(
                    segment_index=global_idx,
                    narrative_beat_index=beat_i,
                    segment_group_id=group_id,
                    segment_group_index=part_j,
                    duration_sec=part_dur,
                    segment_role=role,
                    start_frame_mode=mode,
                    flf_mode="continuation" if part_j > 0 else "none",
                    face_anchor_shot_id="",
                    title=title,
                    shot_size=shot_size,
                    location=location,
                    narrative=narrative,
                )
            )
            global_idx += 1
    return planned


def expand_segments_with_anchor_plan(
    beat_sheet: list[str],
    beat_durations: dict[int, float],
    reachability_by_beat: dict[int, FaceReachability],
    *,
    max_clip_sec: float = DEFAULT_MAX_CLIP_SEC,
    anchor_split_by_beat: dict[int, list[SubsegmentPlan]] | None = None,
    story_graph: dict[int, dict[str, Any]] | None = None,
    reachability_chars_by_beat: dict[int, tuple[str, ...]] | None = None,
) -> list[PlannedSegment]:
    """A2 — replace coarse segments with anchor-aware subsegments per beat."""
    planned: list[PlannedSegment] = []
    global_idx = 0
    for beat_i, beat_raw in enumerate(beat_sheet):
        title, shot_size, location, narrative = _split_beat_fields(beat_raw)
        beat_dur = float(beat_durations.get(beat_i, 5.0))
        group_id = f"beat_{beat_i}"
        reach = reachability_by_beat.get(beat_i, _rule_reachability(shot_size, narrative, beat_index=beat_i))
        if anchor_split_by_beat and beat_i in anchor_split_by_beat:
            subsegs = anchor_split_by_beat[beat_i]
        else:
            subsegs = _rule_split_beat(
                beat_index=beat_i,
                beat_dur=beat_dur,
                shot_size=shot_size,
                reachability=reach,
                max_clip_sec=max_clip_sec,
                story_graph=(story_graph or {}).get(beat_i),
                narrative=narrative,
            )
        anchor_id = _stable_shot_id("anchor", f"{group_id}:face")
        for part_j, sub in enumerate(subsegs):
            role = sub.role
            dur = sub.duration_sec
            seg_shot = sub.shot_size
            flf = sub.flf_mode
            if role == "tail_continuation" and part_j == 0:
                role = "keyframe"
                flf = "none"
            start_mode: StartFrameMode
            if role == "tail_continuation":
                start_mode = "prev_segment_tail"
            elif role == "post_anchor":
                start_mode = "anchor_link"
            else:
                start_mode = "keyframe"
            link_anchor = anchor_id if role in ("pre_anchor", "post_anchor") else ""
            if role == "face_anchor":
                link_anchor = anchor_id
            beat_chars = tuple(sub.characters_on_screen)
            if not beat_chars and reachability_chars_by_beat:
                beat_chars = reachability_chars_by_beat.get(beat_i, ())
            if not beat_chars and story_graph:
                beat_chars = tuple(story_graph.get(beat_i, {}).get("characters_on_screen") or ())
            planned.append(
                PlannedSegment(
                    segment_index=global_idx,
                    narrative_beat_index=beat_i,
                    segment_group_id=group_id,
                    segment_group_index=part_j,
                    duration_sec=round(float(dur), 1),
                    segment_role=role,
                    start_frame_mode=start_mode,
                    flf_mode=flf,
                    face_anchor_shot_id=link_anchor if role != "face_anchor" else anchor_id,
                    title=title,
                    shot_size=seg_shot or shot_size,
                    location=location,
                    narrative=narrative,
                    start_visibility=sub.start_visibility,
                    end_visibility=sub.end_visibility,
                    characters_on_screen=beat_chars or sub.characters_on_screen,
                    first_frame_requirement=sub.first_frame_requirement,
                )
            )
            global_idx += 1
    return planned


def _batch_output_token_budget(base: int, batch_len: int) -> int:
    """Scale completion budget with batch size to avoid truncated JSON arrays."""
    n = max(1, int(batch_len))
    return min(8192, max(base, base + 280 * (n - 1)))


def _required_indices_clause(segments: list[PlannedSegment]) -> str:
    indices = sorted(seg.segment_index for seg in segments)
    return (
        f"Return exactly {len(indices)} JSON row(s) with index values {indices}. "
        f"Every listed index is required; do not omit or renumber.\n"
    )


def _role_motion_hint(role: str) -> str:
    hints = {
        "pre_anchor": "motion_hint=approach/wide-to-anchor; camera dolly-in; no face dialogue",
        "face_anchor": "motion_hint=MCU hold; breath/blink only; minimal camera movement",
        "post_anchor": "motion_hint=continue from anchor; distinct action from pre/face",
        "tail_continuation": "motion_hint=extend prior clip tail",
        "establishing": "motion_hint=slow establishing drift",
        "keyframe": "motion_hint=single-beat action for full clip",
    }
    return hints.get(role, "motion_hint=role-appropriate pacing")


def _characters_on_screen_clause(seg: PlannedSegment) -> str:
    names = [n.strip() for n in seg.characters_on_screen if str(n).strip()]
    if not names:
        return "characters_on_screen=(none listed — use names from video_prompt only, no wardrobe tags)"
    return f"characters_on_screen={', '.join(names)}"


def _format_segment_plan_block(segments: list[PlannedSegment]) -> str:
    lines: list[str] = []
    for seg in segments:
        lines.append(
            f"[{seg.segment_index}] group={seg.segment_group_id} part={seg.segment_group_index} "
            f"role={seg.segment_role} flf={seg.flf_mode} duration_sec={seg.duration_sec}\n"
            f"{_role_motion_hint(seg.segment_role)}\n"
            f"{_characters_on_screen_clause(seg)}\n"
            f"title={seg.title or '(untitled)'}\n"
            f"shot_size={seg.shot_size or 'medium'}\n"
            f"location={seg.location or 'unspecified'}\n"
            f"narrative={seg.narrative}"
        )
    return "\n\n".join(lines)


def _face_reachability_user_message(
    *,
    beat_sheet: list[str],
    character_anchor: str,
    locale_block: str,
) -> str:
    rows: list[str] = []
    for i, beat_raw in enumerate(beat_sheet):
        title, shot_size, location, narrative = _split_beat_fields(beat_raw)
        rows.append(
            f"[{i}] title={title}\nshot_size={shot_size}\nlocation={location}\nnarrative={narrative}"
        )
    return (
        "Classify face reachability for each beat:\n\n"
        + "\n\n".join(rows)
        + f"\n\nCharacter roster:\n{character_anchor.strip()}\n"
        + locale_block
    )


def _anchor_split_user_message(
    *,
    beat_sheet: list[str],
    beat_durations: dict[int, float],
    reachability: dict[int, FaceReachability],
    locale_block: str,
) -> str:
    rows: list[str] = []
    for i, beat_raw in enumerate(beat_sheet):
        title, shot_size, location, narrative = _split_beat_fields(beat_raw)
        dur = beat_durations.get(i, 5.0)
        reach = reachability.get(i, "identity_critical")
        rows.append(
            f"[{i}] guide_sec={dur} (soft — prioritize smooth motion + story completeness) reachability={reach}\n"
            f"title={title}\nshot_size={shot_size}\nlocation={location}\nnarrative={narrative}"
        )
    return (
        "Split each beat into subsegments (pre_anchor / face_anchor / post_anchor as needed). "
        "Total duration per beat may exceed guide_sec when pre→anchor→post needs more time:\n\n"
        + "\n\n".join(rows)
        + locale_block
    )


def _parse_anchor_split(payload: AnchorSplitBatchSchema) -> dict[int, list[SubsegmentPlan]]:
    out: dict[int, list[SubsegmentPlan]] = {}
    cap = DEFAULT_MAX_CLIP_SEC
    for beat in payload.beats:
        rows: list[SubsegmentPlan] = []
        for sub in beat.subsegments:
            flf = sub.flf_mode
            if flf == "first_last":
                flf = "none"
            dur = max(MIN_CLIP_SEC, min(cap, float(sub.duration_sec)))
            rows.append(
                SubsegmentPlan(
                    role=sub.role,
                    duration_sec=dur,
                    shot_size=sub.shot_size,
                    flf_mode=flf,
                    start_visibility=sub.start_visibility,
                    end_visibility=sub.end_visibility,
                    characters_on_screen=tuple(sub.characters_on_screen),
                    first_frame_requirement=sub.first_frame_requirement.strip(),
                )
            )
        out[int(beat.beat_index)] = rows
    return out


def _invoke_face_reachability_batch(
    *,
    chat_fn: Callable[..., Any],
    think_apply: Callable[[str], str],
    max_tokens: int,
    beat_sheet: list[str],
    character_anchor: str,
    locale_block: str,
) -> tuple[dict[int, FaceReachability], dict[int, tuple[str, ...]], int]:
    user = _face_reachability_user_message(
        beat_sheet=beat_sheet,
        character_anchor=character_anchor,
        locale_block=locale_block,
    )
    resp = invoke_text_chat(
        chat_fn,
        system=CHAPTER_FACE_REACHABILITY_SYSTEM,
        user=user,
        max_tokens=max_tokens,
        think_apply=think_apply,
    )
    try:
        payload = FaceReachabilityBatchSchema.model_validate(extract_json_object(resp))
    except ValidationError as exc:
        raise ValueError(f"face reachability JSON invalid: {exc}") from exc
    out: dict[int, FaceReachability] = {}
    chars_out: dict[int, tuple[str, ...]] = {}
    for row in payload.beats:
        bi = int(row.beat_index)
        out[bi] = row.reachability
        names = tuple(n.strip() for n in row.characters_on_screen if str(n).strip())
        if names:
            chars_out[bi] = names
    return out, chars_out, 1


def _validate_anchor_split_payload(
    payload: AnchorSplitBatchSchema,
    beat_durations: dict[int, float],
) -> tuple[bool, str]:
    del beat_durations  # soft guide only — do not fail validation on total drift
    issues: list[str] = []
    cap = DEFAULT_MAX_CLIP_SEC
    for beat in payload.beats:
        for sub in beat.subsegments:
            if sub.flf_mode == "first_last":
                issues.append(f"beat {beat.beat_index}: flf_mode=first_last is forbidden")
            d = float(sub.duration_sec)
            if d > cap + 0.05:
                issues.append(
                    f"beat {beat.beat_index}: duration_sec {d:.1f} exceeds clip cap {cap:.1f}s "
                    f"(split into tail_continuation parts)"
                )
    return (not issues, "\n".join(issues))


def _invoke_anchor_split_batch(
    *,
    chat_fn: Callable[..., Any],
    think_apply: Callable[[str], str],
    max_tokens: int,
    beat_sheet: list[str],
    beat_durations: dict[int, float],
    reachability: dict[int, FaceReachability],
    locale_block: str,
) -> tuple[dict[int, list[SubsegmentPlan]], int]:
    from backend.engine.llm.llm_retry import invoke_text_chat_with_feedback

    user = _anchor_split_user_message(
        beat_sheet=beat_sheet,
        beat_durations=beat_durations,
        reachability=reachability,
        locale_block=locale_block,
    )

    def _validate(resp: str) -> tuple[bool, str]:
        try:
            payload = AnchorSplitBatchSchema.model_validate(extract_json_object(resp))
        except (ValueError, ValidationError) as exc:
            return False, str(exc)
        return _validate_anchor_split_payload(payload, beat_durations)

    resp, calls = invoke_text_chat_with_feedback(
        chat_fn,
        system=CHAPTER_ANCHOR_SPLIT_SYSTEM,
        user=user,
        max_tokens=max_tokens,
        think_apply=think_apply,
        validate=_validate,
        max_attempts=2,
    )
    payload = AnchorSplitBatchSchema.model_validate(extract_json_object(resp))
    parsed = _parse_anchor_split(payload)
    from backend.engine.common.long_video.shot_repair import normalize_subsegment_plans

    out: dict[int, list[SubsegmentPlan]] = {}
    for beat_i, rows in parsed.items():
        out[beat_i] = normalize_subsegment_plans(rows, max_clip_sec=DEFAULT_MAX_CLIP_SEC)
    return out, calls


def _segment_video_user_message(
    *,
    segments: list[PlannedSegment],
    synopsis: str,
    mood: str,
    style_anchor: str,
    character_anchor: str,
    locale_block: str,
) -> str:
    style_line = f"Style: {style_anchor.strip()}\n" if style_anchor.strip() else ""
    mood_line = f"Mood: {mood.strip()}\n" if mood.strip() else ""
    return (
        f"Synopsis:\n{synopsis.strip()}\n\n"
        f"{mood_line}"
        f"{style_line}"
        f"Character reference (do not paste verbatim into every segment):\n{character_anchor.strip()}\n\n"
        f"Segment plan ({len(segments)} clips):\n\n"
        f"{_format_segment_plan_block(segments)}\n\n"
        f"{_required_indices_clause(segments)}"
        f"Return JSON with one **segments[]** row per index above.\n"
        f"Each index in the same group=beat_N must have a **unique** video_prompt (different camera + action).\n"
        f"Use full character names only — no Name（…） wardrobe tags in video_prompt.\n"
        f"Do **not** copy the Style line above into video_prompt — motion and camera only.\n"
        f"For role=pre_anchor: approach motion toward anchor; role=post_anchor: continue from anchor; "
        f"role=face_anchor: minimal hold/breath; role=tail_continuation: continue prior clip.\n"
        f"{locale_block}"
    )


def _story_context_header(*, synopsis: str = "", mood: str = "") -> str:
    parts: list[str] = []
    if (synopsis or "").strip():
        parts.append(f"Synopsis:\n{synopsis.strip()}")
    if (mood or "").strip():
        parts.append(f"Mood: {mood.strip()}")
    return "\n\n".join(parts) + ("\n\n" if parts else "")


def _start_visual_user_message(
    *,
    segments: list[PlannedSegment],
    video_by_index: dict[int, str],
    locale_block: str,
    synopsis: str = "",
    mood: str = "",
) -> str:
    rows: list[str] = []
    for seg in segments:
        if seg.start_frame_mode not in ("keyframe",):
            continue
        if seg.segment_role in ("face_anchor",):
            continue
        vp = video_by_index.get(seg.segment_index, "").strip()
        req = (seg.first_frame_requirement or "").strip()
        vis = (seg.start_visibility or "").strip()
        narrative = (seg.narrative or "").strip()
        rows.append(
            f"[{seg.segment_index}] role={seg.segment_role} duration_sec={seg.duration_sec}\n"
            f"{_characters_on_screen_clause(seg)}\n"
            f"video_prompt={vp}\n"
            f"location={seg.location}\n"
            f"shot_size={seg.shot_size or 'medium'}\n"
            f"start_visibility={vis or 'partial'}\n"
            f"first_frame_requirement={req or '(none)'}\n"
            f"narrative={narrative}"
        )
    return (
        _story_context_header(synopsis=synopsis, mood=mood)
        + "Derive **start_visual** (t=0 still) for each segment below.\n"
        "Use **full character names only** — no Name（…） tags or clothing in start_visual.\n\n"
        + "\n\n".join(rows)
        + "\n\n"
        + _required_indices_clause(segments)
        + locale_block
    )


def _anchor_visual_user_message(
    *,
    segments: list[PlannedSegment],
    video_by_index: dict[int, str],
    locale_block: str,
    synopsis: str = "",
    mood: str = "",
) -> str:
    rows: list[str] = []
    for seg in segments:
        if seg.segment_role != "face_anchor":
            continue
        vp = video_by_index.get(seg.segment_index, "").strip()
        req = (seg.first_frame_requirement or "").strip()
        narrative = (seg.narrative or "").strip()
        rows.append(
            f"[{seg.segment_index}] duration_sec={seg.duration_sec}\n"
            f"{_characters_on_screen_clause(seg)}\n"
            f"video_prompt={vp}\n"
            f"location={seg.location}\n"
            f"shot_size={seg.shot_size or '特写'}\n"
            f"first_frame_requirement={req or '(none)'}\n"
            f"narrative={narrative}"
        )
    return (
        _story_context_header(synopsis=synopsis, mood=mood)
        + "Derive **anchor_visual** (MCU/CU face lock still) for each face_anchor segment.\n"
        "Use **full character names only** — no Name（…） tags or clothing in anchor_visual.\n"
        "anchor_visual = frozen t=0 still (expression/gaze only); defer button presses, walking, and camera moves to video_prompt.\n\n"
        + "\n\n".join(rows)
        + "\n\n"
        + _required_indices_clause(segments)
        + locale_block
    )


def _invoke_segment_video_batch(
    *,
    chat_fn: Callable[..., Any],
    think_apply: Callable[[str], str],
    max_tokens: int,
    segments: list[PlannedSegment],
    synopsis: str,
    mood: str,
    style_anchor: str,
    character_anchor: str,
    locale_block: str,
    repair_feedback: str = "",
) -> tuple[dict[int, str], int]:
    from backend.engine.common.long_video.segment_video_quality import (
        validate_segment_video_batch_payload,
    )
    from backend.engine.llm.llm_retry import invoke_text_chat_with_feedback

    user = _segment_video_user_message(
        segments=segments,
        synopsis=synopsis,
        mood=mood,
        style_anchor=style_anchor,
        character_anchor=character_anchor,
        locale_block=locale_block,
    )
    if repair_feedback.strip():
        user = f"{user.strip()}\n\n---\nRepair pass:\n{repair_feedback.strip()}"

    def _validate(resp: str) -> tuple[bool, str]:
        try:
            payload = SegmentVideoBatchSchema.model_validate(extract_json_object(resp))
        except (ValueError, ValidationError) as exc:
            return False, str(exc)
        return validate_segment_video_batch_payload(
            segments,
            payload.segments,
            style_anchor=style_anchor,
            character_anchor=character_anchor,
        )

    resp, calls = invoke_text_chat_with_feedback(
        chat_fn,
        system=CHAPTER_SEGMENT_VIDEO_SYSTEM,
        user=user,
        max_tokens=max_tokens,
        think_apply=think_apply,
        validate=_validate,
        max_attempts=2,
    )
    payload = SegmentVideoBatchSchema.model_validate(extract_json_object(resp))
    from backend.engine.common.long_video.parse_quality import strip_style_from_motion_prompt

    out: dict[int, str] = {}
    for row in payload.segments:
        text = strip_name_look_tags(row.video_prompt.strip())
        text = strip_style_from_motion_prompt(
            text,
            style_anchor=style_anchor,
            character_anchor=character_anchor,
        )
        if not text:
            raise ValueError(f"segment video_prompt empty for index {row.index}")
        out[int(row.index)] = text
    expected = {s.segment_index for s in segments}
    missing = expected - set(out.keys())
    if missing:
        raise ValueError(f"segment video JSON missing indices: {sorted(missing)}")
    return out, calls


def _invoke_start_visual_batch(
    *,
    chat_fn: Callable[..., Any],
    think_apply: Callable[[str], str],
    max_tokens: int,
    keyframe_segments: list[PlannedSegment],
    video_by_index: dict[int, str],
    locale_block: str,
    synopsis: str = "",
    mood: str = "",
) -> tuple[dict[int, str], int]:
    if not keyframe_segments:
        return {}, 0
    user = _start_visual_user_message(
        segments=keyframe_segments,
        video_by_index=video_by_index,
        locale_block=locale_block,
        synopsis=synopsis,
        mood=mood,
    )
    body = user.split("Derive **start_visual**")[1] if "Derive **start_visual**" in user else ""
    if not body.strip().replace(locale_block.strip(), "").strip():
        return {}, 0
    resp = invoke_text_chat(
        chat_fn,
        system=CHAPTER_START_VISUAL_SYSTEM,
        user=user,
        max_tokens=max_tokens,
        think_apply=think_apply,
    )
    try:
        payload = SegmentStartVisualBatchSchema.model_validate(extract_json_object(resp))
    except ValidationError as exc:
        raise ValueError(f"start visual JSON schema invalid: {exc}") from exc
    out: dict[int, str] = {}
    for row in payload.starts:
        text = strip_name_look_tags(row.start_visual.strip())
        if not text:
            raise ValueError(f"start_visual empty for index {row.index}")
        out[int(row.index)] = text
    expected = {s.segment_index for s in keyframe_segments if s.segment_role not in ("face_anchor",)}
    missing = expected - set(out.keys())
    if missing:
        raise ValueError(f"start visual JSON missing indices: {sorted(missing)}")
    return out, 1


def _invoke_anchor_visual_batch(
    *,
    chat_fn: Callable[..., Any],
    think_apply: Callable[[str], str],
    max_tokens: int,
    anchor_segments: list[PlannedSegment],
    video_by_index: dict[int, str],
    locale_block: str,
    synopsis: str = "",
    mood: str = "",
) -> tuple[dict[int, str], int]:
    if not anchor_segments:
        return {}, 0
    user = _anchor_visual_user_message(
        segments=anchor_segments,
        video_by_index=video_by_index,
        locale_block=locale_block,
        synopsis=synopsis,
        mood=mood,
    )
    resp = invoke_text_chat(
        chat_fn,
        system=CHAPTER_ANCHOR_VISUAL_SYSTEM,
        user=user,
        max_tokens=max_tokens,
        think_apply=think_apply,
    )
    try:
        payload = SegmentAnchorVisualBatchSchema.model_validate(extract_json_object(resp))
    except ValidationError as exc:
        raise ValueError(f"anchor visual JSON schema invalid: {exc}") from exc
    out: dict[int, str] = {}
    for row in payload.anchors:
        text = strip_name_look_tags(row.anchor_visual.strip())
        if not text:
            raise ValueError(f"anchor_visual empty for index {row.index}")
        out[int(row.index)] = text
    expected = {s.segment_index for s in anchor_segments}
    missing = expected - set(out.keys())
    if missing:
        raise ValueError(f"anchor visual JSON missing indices: {sorted(missing)}")
    return out, 1


def _batched(items: list[PlannedSegment], size: int) -> list[list[PlannedSegment]]:
    if size <= 0:
        return [items]
    return [items[i : i + size] for i in range(0, len(items), size)]


def _batch_map_complete(
    items: list[PlannedSegment],
    *,
    batch_size: int,
    invoke: Callable[[list[PlannedSegment]], tuple[dict[int, str], int]],
    error_label: str,
) -> tuple[dict[int, str], int]:
    """Invoke batched LLM maps; bisect batch on missing indices until complete or fail loud."""

    def _collect(batch: list[PlannedSegment]) -> tuple[dict[int, str], int]:
        if not batch:
            return {}, 0
        try:
            batch_map, calls = invoke(batch)
        except ValueError:
            if len(batch) == 1:
                raise
            mid = max(1, len(batch) // 2)
            left_map, left_calls = _collect(batch[:mid])
            right_map, right_calls = _collect(batch[mid:])
            merged = {**left_map, **right_map}
            expected = {seg.segment_index for seg in batch}
            still_missing = expected - set(merged.keys())
            if still_missing:
                raise ValueError(f"{error_label} JSON missing indices: {sorted(still_missing)}")
            return merged, left_calls + right_calls
        expected = {seg.segment_index for seg in batch}
        missing = expected - set(batch_map.keys())
        if not missing:
            return batch_map, calls
        if len(batch) == 1:
            raise ValueError(f"{error_label} JSON missing indices: {sorted(missing)}")
        mid = max(1, len(batch) // 2)
        left_map, left_calls = _collect(batch[:mid])
        right_map, right_calls = _collect(batch[mid:])
        merged = {**left_map, **right_map}
        still_missing = expected - set(merged.keys())
        if still_missing:
            raise ValueError(f"{error_label} JSON missing indices: {sorted(still_missing)}")
        return merged, calls + left_calls + right_calls

    out: dict[int, str] = {}
    llm_calls = 0
    for batch in _batched(items, batch_size):
        batch_map, n = _collect(batch)
        out.update(batch_map)
        llm_calls += n
    return out, llm_calls


def _chain_mode_for_segment(seg: PlannedSegment) -> str:
    if seg.start_frame_mode == "prev_segment_tail" or seg.flf_mode == "continuation":
        return "last_frame"
    return "keyframe_only"


def _first_frame_strategy_for_segment(seg: PlannedSegment) -> str:
    if seg.start_frame_mode == "prev_segment_tail":
        return "reuse_prev_tail"
    if seg.segment_role == "face_anchor":
        return "direct_reuse_portrait"
    if seg.start_frame_mode == "anchor_link":
        return "img2img_light"
    return "t2i_from_grounding"


def _shot_location(seg: PlannedSegment) -> str:
    from backend.engine.llm.storyboard_scenes import parse_scene_beat_location

    loc = (seg.location or "").strip()
    if loc:
        return loc
    if seg.shot_size:
        parsed = parse_scene_beat_location(f"【{seg.shot_size}】{seg.narrative}")
        if parsed:
            return parsed
    return parse_scene_beat_location(seg.narrative)


def _shot_scene_prompt(seg: PlannedSegment) -> str:
    loc = _shot_location(seg)
    narrative = strip_name_look_tags((seg.narrative or "").strip())
    if loc and narrative:
        if loc in narrative or narrative.startswith(loc):
            return narrative
        return f"{loc}，{narrative}"
    return narrative or loc


def _shots_from_segments(
    segments: list[PlannedSegment],
    *,
    video_by_index: dict[int, str],
    start_by_index: dict[int, str],
    anchor_by_index: dict[int, str],
    story_graph: dict[int, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    anchor_keyframe_ids: dict[str, str] = {}
    group_anchor_visual: dict[str, str] = {}
    for seg in segments:
        if seg.segment_role == "face_anchor":
            anchor_keyframe_ids[seg.face_anchor_shot_id] = _stable_shot_id(
                "shot", f"{seg.segment_group_id}:anchor"
            )
            av = anchor_by_index.get(seg.segment_index, "").strip()
            if av:
                group_anchor_visual[seg.segment_group_id] = av

    shots: list[dict[str, Any]] = []
    for seg in segments:
        video_prompt = video_by_index[seg.segment_index]
        if seg.segment_role == "face_anchor":
            start_visual = anchor_by_index.get(seg.segment_index, "")
        else:
            start_visual = start_by_index.get(seg.segment_index, "")
        if not start_visual and seg.segment_role == "post_anchor":
            start_visual = group_anchor_visual.get(seg.segment_group_id, "")
        shot_id = _stable_shot_id("shot", f"{seg.segment_group_id}:{seg.segment_group_index}:{seg.segment_role}")
        if seg.segment_role == "face_anchor":
            shot_id = anchor_keyframe_ids.get(seg.face_anchor_shot_id, shot_id)
        graph = (story_graph or {}).get(seg.narrative_beat_index, {})
        chars = list(seg.characters_on_screen) or list(graph.get("characters_on_screen") or [])
        start_vis = seg.start_visibility or graph.get("start_visibility") or "partial"
        end_vis = seg.end_visibility or graph.get("end_visibility") or start_vis
        if seg.segment_index == 0 and chars and _vis_rank(start_vis) < _vis_rank("silhouette"):
            start_vis = "silhouette"
        shots.append(
            {
                "id": shot_id,
                "order": seg.segment_index,
                "visual_prompt": start_visual,
                "motion_prompt": video_prompt,
                "video_prompt": video_prompt,
                "start_visual_prompt": start_visual,
                "end_visual_prompt": "",
                "anchor_visual_prompt": anchor_by_index.get(seg.segment_index, "")
                if seg.segment_role == "face_anchor"
                else "",
                "start_frame_mode": seg.start_frame_mode,
                "segment_role": seg.segment_role,
                "segment_group_id": seg.segment_group_id,
                "segment_group_index": seg.segment_group_index,
                "face_anchor_shot_id": seg.face_anchor_shot_id if seg.segment_role != "face_anchor" else shot_id,
                "flf_mode": "none" if seg.flf_mode == "first_last" else seg.flf_mode,
                "duration_sec": seg.duration_sec,
                "chain_mode": _chain_mode_for_segment(seg),
                "scene_prompt": _shot_scene_prompt(seg),
                "location": _shot_location(seg),
                "shot_size": seg.shot_size,
                "narrative_beat_index": seg.narrative_beat_index,
                "end_frame_sync_anchor": False,
                "first_frame_visibility": start_vis,
                "end_visibility": end_vis,
                "characters_on_screen": chars,
                "clip_start_state": start_visual or seg.first_frame_requirement,
                "clip_end_state": graph.get("action_summary", "")[:120],
                "first_frame_requirement": (seg.first_frame_requirement or "").strip(),
                "camera_zone_id": seg.camera_zone_id,
                "first_frame_strategy": _first_frame_strategy_for_segment(seg),
            }
        )
    return shots


def _vis_rank(value: str) -> int:
    order = {"invisible": 0, "silhouette": 1, "partial": 2, "full_face": 3}
    return order.get(str(value or "invisible"), 0)


def run_segment_shot_pipeline(
    *,
    beat_sheet: list[str],
    synopsis: str,
    character_anchor: str,
    style_anchor: str,
    mood: str = "",
    locale: str,
    target_duration_sec: float,
    segment_duration_sec: float,
    max_clip_sec: float = DEFAULT_MAX_CLIP_SEC,
    chat_fn: Callable[..., Any],
    think_apply: Callable[[str], str],
    token_budget: Callable[[int], int],
    on_progress: ProgressFn | None = None,
    story_graph: dict[int, dict[str, Any]] | None = None,
) -> SegmentPipelineResult:
    from backend.engine.llm.prompts.locale import chapter_json_user_locale_block
    from backend.engine.llm.storyboard import normalize_storyboard_locale

    def progress(phase: str, message: str) -> None:
        if on_progress:
            on_progress(phase, message)

    loc = normalize_storyboard_locale(locale)
    locale_block = chapter_json_user_locale_block(loc)
    apply_think = think_apply or (lambda t: t)
    budget = token_budget or (lambda b: b)
    phases: list[str] = []
    llm_calls = 0

    progress("segment_plan", "segment_plan")
    phases.append("segment_plan")
    coarse = plan_segments_from_beats(
        beat_sheet,
        target_duration_sec=target_duration_sec,
        default_segment_sec=segment_duration_sec,
        max_clip_sec=max_clip_sec,
    )
    if not coarse:
        raise ValueError("segment plan produced no shots")

    beat_durations: dict[int, float] = {}
    for seg in coarse:
        beat_durations[seg.narrative_beat_index] = beat_durations.get(seg.narrative_beat_index, 0.0) + seg.duration_sec

    beat_budgets = dict(beat_durations)  # soft per-beat duration guides (warnings only)

    reachability: dict[int, FaceReachability] = {}
    reachability_chars: dict[int, tuple[str, ...]] = {}
    progress("face_reachability", "face_reachability")
    phases.append("face_reachability")
    try:
        reach_map, reach_chars, n = _invoke_face_reachability_batch(
            chat_fn=chat_fn,
            think_apply=apply_think,
            max_tokens=budget(2000),
            beat_sheet=beat_sheet,
            character_anchor=character_anchor,
            locale_block=locale_block,
        )
        reachability.update(reach_map)
        reachability_chars.update(reach_chars)
        llm_calls += n
    except (ValueError, ValidationError):
        for beat_i, beat_raw in enumerate(beat_sheet):
            _, shot_size, _, narrative = _split_beat_fields(beat_raw)
            r = _rule_reachability(shot_size, narrative, beat_index=beat_i)
            reachability[beat_i] = r

    anchor_split: dict[int, list[tuple[SegmentRole, float, str, FlfMode]]] | None = None
    progress("anchor_split_plan", "anchor_split_plan")
    phases.append("anchor_split_plan")
    try:
        anchor_split, n = _invoke_anchor_split_batch(
            chat_fn=chat_fn,
            think_apply=apply_think,
            max_tokens=budget(2400),
            beat_sheet=beat_sheet,
            beat_durations=beat_durations,
            reachability=reachability,
            locale_block=locale_block,
        )
        llm_calls += n
    except (ValueError, ValidationError):
        anchor_split = None

    segments = expand_segments_with_anchor_plan(
        beat_sheet,
        beat_durations,
        reachability,
        max_clip_sec=max_clip_sec,
        anchor_split_by_beat=anchor_split,
        story_graph=story_graph,
        reachability_chars_by_beat=reachability_chars,
    )
    if not segments:
        raise ValueError("anchor split produced no segments")

    video_by_index: dict[int, str] = {}
    progress("segment_video", "segment_video")
    phases.append("segment_video")

    def _segment_video_invoke(batch: list[PlannedSegment]) -> tuple[dict[int, str], int]:
        return _invoke_segment_video_batch(
            chat_fn=chat_fn,
            think_apply=apply_think,
            max_tokens=budget(_batch_output_token_budget(2400, len(batch))),
            segments=batch,
            synopsis=synopsis,
            mood=mood,
            style_anchor=style_anchor,
            character_anchor=character_anchor,
            locale_block=locale_block,
        )

    video_by_index, n = _batch_map_complete(
        segments,
        batch_size=SEGMENT_VIDEO_BATCH,
        invoke=_segment_video_invoke,
        error_label="segment video",
    )
    llm_calls += n

    progress("segment_video_repair", "segment_video_repair")
    phases.append("segment_video_repair")
    from backend.engine.llm.segment_video_repair import repair_segment_video_groups

    video_by_index, n_repair, _video_quality = repair_segment_video_groups(
        segments,
        video_by_index,
        chat_fn=chat_fn,
        think_apply=apply_think,
        max_tokens=budget(_batch_output_token_budget(2400, SEGMENT_VIDEO_BATCH)),
        synopsis=synopsis,
        mood=mood,
        style_anchor=style_anchor,
        character_anchor=character_anchor,
        locale_block=locale_block,
    )
    llm_calls += n_repair

    start_segments = [s for s in segments if s.start_frame_mode == "keyframe" and s.segment_role != "face_anchor"]
    anchor_segments = [s for s in segments if s.segment_role == "face_anchor"]

    start_by_index: dict[int, str] = {}
    if start_segments:
        progress("start_visual", "start_visual")
        phases.append("start_visual")

        def _start_visual_invoke(batch: list[PlannedSegment]) -> tuple[dict[int, str], int]:
            return _invoke_start_visual_batch(
                chat_fn=chat_fn,
                think_apply=apply_think,
                max_tokens=budget(_batch_output_token_budget(2000, len(batch))),
                keyframe_segments=batch,
                video_by_index=video_by_index,
                locale_block=locale_block,
                synopsis=synopsis,
                mood=mood,
            )

        start_by_index, n = _batch_map_complete(
            start_segments,
            batch_size=START_VISUAL_BATCH,
            invoke=_start_visual_invoke,
            error_label="start visual",
        )
        llm_calls += n

    anchor_by_index: dict[int, str] = {}
    if anchor_segments:
        progress("anchor_visual", "anchor_visual")
        phases.append("anchor_visual")

        def _anchor_visual_invoke(batch: list[PlannedSegment]) -> tuple[dict[int, str], int]:
            return _invoke_anchor_visual_batch(
                chat_fn=chat_fn,
                think_apply=apply_think,
                max_tokens=budget(_batch_output_token_budget(2000, len(batch))),
                anchor_segments=batch,
                video_by_index=video_by_index,
                locale_block=locale_block,
                synopsis=synopsis,
                mood=mood,
            )

        anchor_by_index, n = _batch_map_complete(
            anchor_segments,
            batch_size=START_VISUAL_BATCH,
            invoke=_anchor_visual_invoke,
            error_label="anchor visual",
        )
        llm_calls += n

    shots = _shots_from_segments(
        segments,
        video_by_index=video_by_index,
        start_by_index=start_by_index,
        anchor_by_index=anchor_by_index,
        story_graph=story_graph,
    )

    progress("done", "done")
    phases.append("done")
    return SegmentPipelineResult(
        shots=shots,
        llm_calls=llm_calls,
        phases=phases,
        beat_budgets=beat_budgets,
    )
