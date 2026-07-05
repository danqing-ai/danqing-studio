"""Map script_parse artifacts to API contract DTOs."""
from __future__ import annotations

from typing import Any

from backend.core.contracts import (
    LongVideoChapterParsePhaseDTO,
    LongVideoChapterSceneDTO,
    LongVideoCharacterDTO,
    LongVideoCharacterLookDTO,
    LongVideoParseQualityIssueDTO,
    LongVideoSceneDTO,
    LongVideoSceneLookDTO,
    LongVideoStoryboardShotDTO,
    ScriptParseDecomposeResponse,
    ScriptParseExpandResponse,
)
from backend.engine.llm.script_parse.schemas import DecomposeResult, ExpandResult, ScriptArtifact
from backend.engine.llm.storyboard_cast import dtos_to_roster, format_character_roster


def script_dict_to_artifact(data: dict[str, Any]) -> ScriptArtifact:
    return ScriptArtifact.model_validate(data)


def artifact_to_scene_beats(artifact: ScriptArtifact) -> list[LongVideoChapterSceneDTO]:
    rows: list[LongVideoChapterSceneDTO] = []
    for beat in artifact.beats:
        beat_text = beat.narrative
        if beat.location:
            beat_text = f"{beat.location} — {beat_text}"
        rows.append(
            LongVideoChapterSceneDTO(
                order=beat.index + 1,
                title=beat.title,
                beat=beat_text,
            )
        )
    return rows


def _character_dtos(raw: list[dict[str, Any]]) -> list[LongVideoCharacterDTO]:
    out: list[LongVideoCharacterDTO] = []
    for row in raw:
        looks = [
            LongVideoCharacterLookDTO(**lk)
            for lk in (row.get("looks") or [])
            if isinstance(lk, dict)
        ]
        out.append(
            LongVideoCharacterDTO(
                id=str(row.get("id", "")),
                name=str(row.get("name", "")),
                default_look_id=str(row.get("default_look_id", "")),
                looks=looks,
            )
        )
    return out


def _scene_dtos(raw: list[dict[str, Any]]) -> list[LongVideoSceneDTO]:
    out: list[LongVideoSceneDTO] = []
    for row in raw:
        looks = [
            LongVideoSceneLookDTO(**lk)
            for lk in (row.get("looks") or [])
            if isinstance(lk, dict)
        ]
        out.append(
            LongVideoSceneDTO(
                id=str(row.get("id", "")),
                name=str(row.get("name", "")),
                default_look_id=str(row.get("default_look_id", "")),
                looks=looks,
                spatial_layout_json=row.get("spatial_layout_json") or {},
                grounding_panorama_asset_id=str(row.get("grounding_panorama_asset_id", "")),
                grounding_depth_asset_id=str(row.get("grounding_depth_asset_id", "")),
            )
        )
    return out


def decompose_to_response(
    result: DecomposeResult,
    *,
    parse_phases: list[LongVideoChapterParsePhaseDTO],
    parse_run_id: str = "",
    project_id: str = "",
    locale: str = "zh",
) -> ScriptParseDecomposeResponse:
    artifact = result.artifact
    roster = dtos_to_roster([c.model_dump() for c in artifact.characters])
    character_anchor = (
        format_character_roster(roster, artifact.style_anchor, locale=locale) if roster else ""
    )
    return ScriptParseDecomposeResponse(
        chapter_title=artifact.title,
        synopsis=artifact.synopsis,
        mood=artifact.mood,
        style_anchor=artifact.style_anchor,
        character_anchor=character_anchor,
        characters=_character_dtos([c.model_dump() for c in artifact.characters]),
        scenes=_scene_dtos([s.model_dump() for s in artifact.scenes]),
        scene_beats=artifact_to_scene_beats(artifact),
        scene_count=len(artifact.beats),
        script_artifact=artifact.model_dump(mode="json"),
        llm_calls=result.llm_calls,
        parse_phases=parse_phases,
        parse_run_id=parse_run_id,
        long_video_project_id=project_id,
    )


def expand_to_response(
    result: ExpandResult,
    *,
    artifact: ScriptArtifact,
    parse_phases: list[LongVideoChapterParsePhaseDTO],
    parse_run_id: str = "",
    project_id: str = "",
) -> ScriptParseExpandResponse:
    shot_dtos = [
        LongVideoStoryboardShotDTO(**row)
        for row in result.shots
        if isinstance(row, dict)
    ]
    return ScriptParseExpandResponse(
        chapter_title=artifact.title,
        synopsis=artifact.synopsis,
        mood=artifact.mood,
        character_anchor=result.character_anchor,
        style_anchor=result.style_anchor,
        characters=_character_dtos(result.characters),
        scenes=_scene_dtos(result.scenes),
        scene_beats=artifact_to_scene_beats(artifact),
        scene_count=len(artifact.beats),
        shots=shot_dtos,
        script_artifact=artifact.model_dump(mode="json"),
        llm_calls=result.llm_calls,
        parse_phases=parse_phases,
        quality_warnings=list(result.validation_warnings),
        quality_issues=[
            LongVideoParseQualityIssueDTO(**row)
            for row in result.quality_issues
            if isinstance(row, dict)
        ],
        parse_run_id=parse_run_id,
        long_video_project_id=project_id,
    )
