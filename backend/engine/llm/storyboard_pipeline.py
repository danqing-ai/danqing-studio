"""Orchestrator for chapter analyze storyboard pipeline (L1–L5)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from backend.engine.common.long_video.constants import MAX_CLIP_SEC
from backend.engine.common.long_video.scene_grounding import apply_grounding_to_scene_dtos
from backend.engine.common.long_video.shot_contract_validator import (
    clamp_shot_durations,
    validate_shot_contracts,
)
from backend.engine.common.long_video.shot_repair import repair_shot_contracts
from backend.engine.llm.chapter_segment_plan import SegmentPipelineResult, run_segment_shot_pipeline
from backend.engine.llm.spatial_layout import (
    attach_spatial_layout_to_scenes,
    ensure_scenes_from_beats,
    run_spatial_layout_pass,
)
from backend.engine.llm.story_graph import run_story_graph_pass
from backend.engine.llm.storyboard import normalize_character_anchor
from backend.engine.llm.storyboard_cast import (
    cast_looks_to_dtos,
    dtos_to_roster,
    infer_shot_cast_looks,
    strip_name_look_tags,
    supplement_roster_from_shots,
)
from backend.engine.llm.storyboard_scenes import (
    dtos_to_roster as dtos_to_scene_roster,
    infer_shot_scene_look,
    scene_look_to_dtos,
)
from backend.engine.llm.spatial_layout import SpatialLayoutByKey, _location_keys

ProgressFn = Callable[[str, str], None]

_SHOT_PROMPT_TEXT_FIELDS = (
    "scene_prompt",
    "start_visual_prompt",
    "visual_prompt",
    "anchor_visual_prompt",
    "video_prompt",
    "motion_prompt",
    "first_frame_requirement",
    "clip_start_state",
    "clip_end_state",
)


def _sanitize_shot_prompt_fields(shots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for shot in shots:
        row = dict(shot)
        for fld in _SHOT_PROMPT_TEXT_FIELDS:
            val = row.get(fld)
            if val:
                row[fld] = strip_name_look_tags(str(val))
        out.append(row)
    return out


@dataclass
class StoryboardPipelineResult:
    shots: list[dict[str, Any]]
    scene_dtos: list[dict[str, Any]]
    character_dtos: list[dict[str, Any]] = field(default_factory=list)
    llm_calls: int = 0
    phases: list[str] = field(default_factory=list)
    validation_warnings: list[str] = field(default_factory=list)
    quality_issues: list[dict[str, Any]] = field(default_factory=list)


def _apply_cast_lock(
    shots: list[dict[str, Any]],
    *,
    character_dtos: list[dict[str, Any]],
    scene_dtos: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    roster = dtos_to_roster(character_dtos)
    scenes = dtos_to_scene_roster(scene_dtos) if scene_dtos else []
    out: list[dict[str, Any]] = []
    prev_cast = None
    prev_scene = None
    for shot in shots:
        row = dict(shot)
        visual = str(
            row.get("start_visual_prompt") or row.get("visual_prompt") or ""
        ).strip()
        location = str(row.get("location") or "").strip()
        scene_text = str(row.get("scene_prompt") or visual)
        beat = str(row.get("video_prompt") or row.get("motion_prompt") or visual)
        on_screen = [
            str(n).strip()
            for n in (row.get("characters_on_screen") or [])
            if str(n).strip()
        ]
        roster_for_shot = (
            [ch for ch in roster if ch.name in on_screen] if on_screen else roster
        )
        binding = infer_shot_scene_look(beat=scene_text, scenes=scenes, prev=prev_scene) if scenes else None
        scene_changed = bool(
            prev_scene
            and binding
            and binding.scene_id != prev_scene.scene_id
        )
        scene_hints: list[str] = []
        if binding:
            sc = next((s for s in scenes if s.id == binding.scene_id), None)
            if sc:
                scene_hints.append(sc.name)
                lk = next((l for l in sc.looks if l.id == binding.look_id), None)
                if lk and lk.label:
                    scene_hints.append(lk.label)
        cast = infer_shot_cast_looks(
            scene="\n".join(p for p in (location, visual) if p),
            beat="\n".join(p for p in (scene_text, beat) if p),
            characters=roster_for_shot,
            prev=None if scene_changed else prev_cast,
            on_screen_names=on_screen or None,
            scene_hints=scene_hints or None,
        )
        prev_cast = cast
        row["cast_looks"] = cast_looks_to_dtos(cast)
        if binding:
            dto = scene_look_to_dtos(binding)
            if dto:
                row["scene_look"] = dto
                prev_scene = binding
                out.append(row)
                continue
        for sc in scenes:
            if sc.name and sc.name in scene_text:
                look_id = sc.default_look_id or (sc.looks[0].id if sc.looks else "")
                if look_id:
                    row["scene_look"] = {"scene_id": sc.id, "look_id": look_id}
                    prev_scene = infer_shot_scene_look(
                        beat=scene_text,
                        scenes=scenes,
                        prev=prev_scene,
                    )
                break
        out.append(row)
    return out


def _assign_shot_camera_zones(
    shots: list[dict[str, Any]],
    layouts: SpatialLayoutByKey,
    beat_sheet: list[str],
) -> None:
    if not layouts:
        return
    loc_to_key = {loc: key for key, loc in _location_keys(beat_sheet)}
    for shot in shots:
        if str(shot.get("camera_zone_id") or "").strip():
            continue
        loc = str(shot.get("location") or shot.get("scene_prompt") or "").strip()
        key = loc_to_key.get(loc)
        if not key and loc:
            for lk, k in loc_to_key.items():
                if lk in loc or loc in lk:
                    key = k
                    break
        if not key or key not in layouts:
            continue
        zones = layouts[key].get("camera_zones") or []
        zone_id = _pick_camera_zone_id(
            zones,
            segment_role=str(shot.get("segment_role") or ""),
            first_frame_visibility=str(shot.get("first_frame_visibility") or ""),
        )
        if zone_id:
            shot["camera_zone_id"] = zone_id


def _pick_camera_zone_id(
    zones: list[dict[str, Any]],
    *,
    segment_role: str,
    first_frame_visibility: str,
) -> str:
    if not zones:
        return ""
    if len(zones) == 1:
        return str(zones[0].get("id") or "")

    close_hints = ("close", "cu", "mcu", "portrait", "face", "entry", "近", "特写")
    wide_hints = ("wide", "establish", "full", "room", "远", "全景", "广角")

    def _score(zone: dict[str, Any]) -> int:
        blob = f"{zone.get('id', '')} {zone.get('description', '')} {zone.get('visible_area', '')}".lower()
        score = 0
        if segment_role == "face_anchor" or first_frame_visibility == "full_face":
            score += sum(2 for h in close_hints if h in blob)
        if segment_role in ("pre_anchor", "establishing", "keyframe"):
            score += sum(1 for h in wide_hints if h in blob)
        if segment_role == "post_anchor":
            score += sum(1 for h in close_hints if h in blob)
        return score

    best = max(zones, key=_score)
    return str(best.get("id") or zones[0].get("id") or "")


def _validate_and_repair_shots(
    shots: list[dict[str, Any]],
    *,
    character_anchor: str,
    max_clip_sec: float,
    target_duration_sec: float,
    beat_budgets: dict[int, float],
    chat_fn: Callable[..., Any] | None,
    think_apply: Callable[[str], str],
    token_budget: Callable[[int], int],
    locale_block: str,
    on_progress: ProgressFn | None,
) -> tuple[list[dict[str, Any]], list[str], int]:
    llm_calls = 0
    warnings: list[str] = []
    cap = max_clip_sec

    shots = clamp_shot_durations(shots, max_clip_sec=cap)
    validation = validate_shot_contracts(
        shots,
        character_anchor=character_anchor,
        max_clip_sec=cap,
        target_duration_sec=target_duration_sec,
        beat_budgets=beat_budgets,
    )

    if not validation.ok:
        if on_progress:
            on_progress("shot_repair", "shot_repair")
        shots = repair_shot_contracts(
            shots,
            character_anchor=character_anchor,
            max_clip_sec=cap,
            beat_budgets=beat_budgets,
        )
        validation = validate_shot_contracts(
            shots,
            character_anchor=character_anchor,
            max_clip_sec=cap,
            target_duration_sec=target_duration_sec,
            beat_budgets=beat_budgets,
        )

    if not validation.ok and chat_fn is not None:
        from backend.engine.llm.shot_repair_llm import repair_shots_with_llm

        if on_progress:
            on_progress("shot_repair", "shot_repair_llm")
        shots, n = repair_shots_with_llm(
            shots,
            issues=validation.issues,
            character_anchor=character_anchor,
            chat_fn=chat_fn,
            think_apply=think_apply,
            max_tokens=token_budget(2400),
            locale_block=locale_block,
        )
        llm_calls += n
        shots = repair_shot_contracts(
            shots,
            character_anchor=character_anchor,
            max_clip_sec=cap,
            beat_budgets=beat_budgets,
        )
        validation = validate_shot_contracts(
            shots,
            character_anchor=character_anchor,
            max_clip_sec=cap,
            target_duration_sec=target_duration_sec,
            beat_budgets=beat_budgets,
        )

    warnings = [w.message for w in validation.warnings]
    if not validation.ok:
        codes = "; ".join(i.message for i in validation.issues[:3])
        raise RuntimeError(f"shot contract validation failed: {codes}")
    return shots, warnings, llm_calls


def run_storyboard_pipeline(
    *,
    beat_sheet: list[str],
    synopsis: str,
    character_anchor: str,
    style_anchor: str,
    mood: str = "",
    locale: str,
    target_duration_sec: float,
    segment_duration_sec: float,
    max_clip_sec: float = MAX_CLIP_SEC,
    character_dtos: list[dict[str, Any]],
    scene_dtos: list[dict[str, Any]],
    chat_fn: Callable[..., Any],
    think_apply: Callable[[str], str],
    token_budget: Callable[[int], int],
    on_progress: ProgressFn | None = None,
) -> StoryboardPipelineResult:
    from backend.engine.llm.prompts.locale import chapter_json_user_locale_block
    from backend.engine.llm.storyboard import normalize_storyboard_locale

    def progress(phase: str, message: str = "") -> None:
        if on_progress:
            on_progress(phase, message)

    loc = normalize_storyboard_locale(locale)
    locale_block = chapter_json_user_locale_block(loc)
    character_anchor = normalize_character_anchor(character_anchor, locale=loc)
    phases: list[str] = []
    llm_calls = 0
    cap = min(float(max_clip_sec), MAX_CLIP_SEC)

    progress("story_graph", "story_graph")
    phases.append("story_graph")
    story_graph, n = run_story_graph_pass(
        beat_sheet=beat_sheet,
        character_anchor=character_anchor,
        synopsis=synopsis,
        locale_block=locale_block,
        chat_fn=chat_fn,
        think_apply=think_apply,
        max_tokens=token_budget(2000),
    )
    llm_calls += n

    progress("spatial_layout", "spatial_layout")
    phases.append("spatial_layout")
    layouts, n = run_spatial_layout_pass(
        beat_sheet=beat_sheet,
        synopsis=synopsis,
        locale_block=locale_block,
        chat_fn=chat_fn,
        think_apply=think_apply,
        max_tokens=token_budget(2000),
    )
    llm_calls += n
    scene_rows = ensure_scenes_from_beats(scene_dtos, beat_sheet, layouts)
    scene_rows = attach_spatial_layout_to_scenes(scene_rows, beat_sheet, layouts)
    scene_rows = apply_grounding_to_scene_dtos(scene_rows, layouts)

    progress("scene_grounding", "scene_grounding")
    phases.append("scene_grounding")

    segment_result: SegmentPipelineResult = run_segment_shot_pipeline(
        beat_sheet=beat_sheet,
        synopsis=synopsis,
        character_anchor=character_anchor,
        style_anchor=style_anchor,
        mood=mood,
        locale=locale,
        target_duration_sec=target_duration_sec,
        segment_duration_sec=segment_duration_sec,
        max_clip_sec=cap,
        chat_fn=chat_fn,
        think_apply=think_apply,
        token_budget=token_budget,
        on_progress=on_progress,
        story_graph=story_graph,
    )
    llm_calls += segment_result.llm_calls
    phases.extend(p for p in segment_result.phases if p not in phases)

    beat_budgets = segment_result.beat_budgets

    shots = segment_result.shots
    shots = _sanitize_shot_prompt_fields(shots)
    _assign_shot_camera_zones(shots, layouts, beat_sheet)
    for i, shot in enumerate(shots):
        beat_i = shot.get("narrative_beat_index")
        if beat_i is None:
            gid = str(shot.get("segment_group_id") or "")
            if gid.startswith("beat_"):
                try:
                    beat_i = int(gid.split("_", 1)[1])
                except ValueError:
                    beat_i = None
        if beat_i is not None and beat_i in story_graph:
            g = story_graph[beat_i]
            if not shot.get("characters_on_screen"):
                shot["characters_on_screen"] = list(g.get("characters_on_screen") or [])

    character_dtos = supplement_roster_from_shots(
        list(character_dtos),
        shots,
        locale=loc,
    )

    progress("cast_lock", "cast_lock")
    phases.append("cast_lock")
    shots = _apply_cast_lock(shots, character_dtos=character_dtos, scene_dtos=scene_rows)

    progress("shot_validate", "shot_validate")
    phases.append("shot_validate")
    shots, warnings, n_repair = _validate_and_repair_shots(
        shots,
        character_anchor=character_anchor,
        max_clip_sec=cap,
        target_duration_sec=target_duration_sec,
        beat_budgets=beat_budgets,
        chat_fn=chat_fn,
        think_apply=think_apply,
        token_budget=token_budget,
        locale_block=locale_block,
        on_progress=progress,
    )
    llm_calls += n_repair
    if "shot_repair" not in phases:
        phases.append("shot_repair")

    from backend.engine.common.long_video.parse_quality import validate_parse_quality

    progress("parse_quality", "parse_quality")
    phases.append("parse_quality")
    quality = validate_parse_quality(
        shots,
        beat_sheet=beat_sheet,
        character_anchor=character_anchor,
        character_dtos=character_dtos,
        style_anchor=style_anchor,
    )
    warnings.extend(quality.warning_messages)

    progress("done", "done")
    phases.append("done")
    return StoryboardPipelineResult(
        shots=shots,
        scene_dtos=scene_rows,
        character_dtos=character_dtos,
        llm_calls=llm_calls,
        phases=phases,
        validation_warnings=warnings,
        quality_issues=quality.issue_dicts(),
    )
