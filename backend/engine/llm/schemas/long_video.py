"""JSON schemas for long-video chapter analyze and scene entity extraction."""
from __future__ import annotations

import re

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

_PLACEHOLDER_RE = re.compile(r"^<[^>]+>$")


def _reject_placeholder(value: str, field_name: str) -> str:
    text = (value or "").strip()
    if not text:
        raise ValueError(f"{field_name} is empty")
    if _PLACEHOLDER_RE.match(text):
        raise ValueError(f"{field_name} still contains a placeholder: {text}")
    return text


class CharacterLookSchema(BaseModel):
    label: str
    role: str = ""
    appearance: str = ""
    wardrobe: str = ""

    @field_validator("label", "role", "appearance", "wardrobe")
    @classmethod
    def _strip(cls, value: str) -> str:
        return (value or "").strip()


class CharacterSchema(BaseModel):
    name: str
    looks: list[CharacterLookSchema] = Field(min_length=1)

    @field_validator("name")
    @classmethod
    def _name(cls, value: str) -> str:
        return _reject_placeholder(value, "characters[].name")


class BeatSchema(BaseModel):
    title: str = ""
    shot_size: str
    location: str
    narrative: str = ""

    @model_validator(mode="before")
    @classmethod
    def _legacy_visual_field(cls, data: object) -> object:
        if isinstance(data, dict):
            if not str(data.get("narrative") or "").strip():
                visual = str(data.get("visual") or "").strip()
                if visual:
                    data = {**data, "narrative": visual}
        return data

    @field_validator("shot_size", "location")
    @classmethod
    def _required_meta(cls, value: str, info) -> str:
        return _reject_placeholder(value, f"beats[].{info.field_name}")

    @field_validator("narrative")
    @classmethod
    def _required_narrative(cls, value: str) -> str:
        return _reject_placeholder(value, "beats[].narrative")


class ChapterAnalyzeSchema(BaseModel):
    synopsis: str
    mood: str = ""
    style: str = ""
    characters: list[CharacterSchema] = Field(min_length=1)
    beats: list[BeatSchema] = Field(min_length=2)

    @field_validator("synopsis")
    @classmethod
    def _synopsis(cls, value: str) -> str:
        text = _reject_placeholder(value, "synopsis")
        if len(text) < 8:
            raise ValueError("synopsis is too short")
        return text

    @field_validator("mood", "style")
    @classmethod
    def _optional(cls, value: str) -> str:
        text = (value or "").strip()
        if text and _PLACEHOLDER_RE.match(text):
            raise ValueError(f"field still contains a placeholder: {text}")
        return text


class ChapterPlanSchema(BaseModel):
    """Round 1 — synopsis, mood, style, beats (no cast roster)."""

    synopsis: str
    mood: str = ""
    style: str = ""
    beats: list[BeatSchema] = Field(min_length=2)

    @field_validator("synopsis")
    @classmethod
    def _synopsis(cls, value: str) -> str:
        text = _reject_placeholder(value, "synopsis")
        if len(text) < 8:
            raise ValueError("synopsis is too short")
        return text

    @field_validator("mood", "style")
    @classmethod
    def _optional(cls, value: str) -> str:
        text = (value or "").strip()
        if text and _PLACEHOLDER_RE.match(text):
            raise ValueError(f"field still contains a placeholder: {text}")
        return text


class ChapterRosterSchema(BaseModel):
    """Round 2 — cast roster aligned to approved beats."""

    characters: list[CharacterSchema] = Field(min_length=1)
    style: str = ""

    @field_validator("style")
    @classmethod
    def _optional(cls, value: str) -> str:
        text = (value or "").strip()
        if text and _PLACEHOLDER_RE.match(text):
            raise ValueError(f"field still contains a placeholder: {text}")
        return text


class ChapterChunkSchema(BaseModel):
    beats: list[BeatSchema] = Field(min_length=1)


class SceneLookSchema(BaseModel):
    label: str = ""
    environment: str
    set_dressing: str = ""

    @field_validator("environment")
    @classmethod
    def _environment(cls, value: str) -> str:
        return _reject_placeholder(value, "scenes[].looks[].environment")

    @field_validator("label", "set_dressing")
    @classmethod
    def _strip(cls, value: str) -> str:
        return (value or "").strip()


class SceneEntitySchema(BaseModel):
    name: str
    looks: list[SceneLookSchema] = Field(min_length=1)

    @field_validator("name")
    @classmethod
    def _name(cls, value: str) -> str:
        return _reject_placeholder(value, "scenes[].name")


class SceneEntityPayloadSchema(BaseModel):
    scenes: list[SceneEntitySchema] = Field(min_length=1)


class SegmentVideoRowSchema(BaseModel):
    index: int = Field(ge=0)
    video_prompt: str

    @field_validator("video_prompt")
    @classmethod
    def _video(cls, value: str) -> str:
        return _reject_placeholder(value, "segments[].video_prompt")


class SegmentVideoBatchSchema(BaseModel):
    segments: list[SegmentVideoRowSchema] = Field(min_length=1)


class SegmentStartVisualRowSchema(BaseModel):
    index: int = Field(ge=0)
    start_visual: str

    @field_validator("start_visual")
    @classmethod
    def _start(cls, value: str) -> str:
        return _reject_placeholder(value, "starts[].start_visual")


class SegmentStartVisualBatchSchema(BaseModel):
    starts: list[SegmentStartVisualRowSchema] = Field(min_length=1)


class FaceReachabilityRowSchema(BaseModel):
    beat_index: int = Field(ge=0)
    reachability: Literal["identity_critical", "establishing", "action_wide", "empty"]
    characters_on_screen: list[str] = Field(default_factory=list)


class FaceReachabilityBatchSchema(BaseModel):
    beats: list[FaceReachabilityRowSchema] = Field(min_length=1)


class AnchorSubsegmentSchema(BaseModel):
    role: Literal[
        "establishing",
        "pre_anchor",
        "face_anchor",
        "post_anchor",
        "keyframe",
        "tail_continuation",
    ]
    duration_sec: float = Field(ge=1.0, le=10.0)
    shot_size: str = ""
    flf_mode: Literal["none", "first_last", "continuation"] = "none"
    start_visibility: Literal["invisible", "silhouette", "partial", "full_face"] = "full_face"
    end_visibility: Literal["invisible", "silhouette", "partial", "full_face"] = "full_face"
    characters_on_screen: list[str] = Field(default_factory=list)
    first_frame_requirement: str = ""

    @field_validator("shot_size")
    @classmethod
    def _strip_shot(cls, value: str) -> str:
        return (value or "").strip()


class AnchorSplitBeatSchema(BaseModel):
    beat_index: int = Field(ge=0)
    subsegments: list[AnchorSubsegmentSchema] = Field(min_length=1)


class AnchorSplitBatchSchema(BaseModel):
    beats: list[AnchorSplitBeatSchema] = Field(min_length=1)


class SegmentEndVisualRowSchema(BaseModel):
    index: int = Field(ge=0)
    end_visual: str

    @field_validator("end_visual")
    @classmethod
    def _end(cls, value: str) -> str:
        return _reject_placeholder(value, "ends[].end_visual")


class SegmentEndVisualBatchSchema(BaseModel):
    ends: list[SegmentEndVisualRowSchema] = Field(min_length=1)


class SegmentAnchorVisualRowSchema(BaseModel):
    index: int = Field(ge=0)
    anchor_visual: str

    @field_validator("anchor_visual")
    @classmethod
    def _anchor(cls, value: str) -> str:
        return _reject_placeholder(value, "anchors[].anchor_visual")


class SegmentAnchorVisualBatchSchema(BaseModel):
    anchors: list[SegmentAnchorVisualRowSchema] = Field(min_length=1)


class StoryGraphEventSchema(BaseModel):
    beat_index: int = Field(ge=0)
    characters_on_screen: list[str] = Field(default_factory=list)
    start_visibility: Literal["invisible", "silhouette", "partial", "full_face"] = "silhouette"
    end_visibility: Literal["invisible", "silhouette", "partial", "full_face"] = "full_face"
    action_summary: str = ""


class StoryGraphBatchSchema(BaseModel):
    events: list[StoryGraphEventSchema] = Field(min_length=1)


class CameraZoneSchema(BaseModel):
    id: str
    description: str = ""
    visible_area: str = ""


class SpatialLayoutSceneSchema(BaseModel):
    scene_key: str
    location: str = ""
    dimensions: str = ""
    objects: list[str] = Field(default_factory=list)
    camera_zones: list[CameraZoneSchema] = Field(default_factory=list)


class SpatialLayoutBatchSchema(BaseModel):
    scenes: list[SpatialLayoutSceneSchema] = Field(min_length=1)


class ShotRepairRowSchema(BaseModel):
    order: int = Field(ge=0)
    first_frame_visibility: Literal["invisible", "silhouette", "partial", "full_face"] | None = None
    end_visibility: Literal["invisible", "silhouette", "partial", "full_face"] | None = None
    characters_on_screen: list[str] | None = None
    start_visual_prompt: str = ""
    video_prompt: str = ""
    first_frame_requirement: str = ""


class ShotRepairBatchSchema(BaseModel):
    repairs: list[ShotRepairRowSchema] = Field(min_length=1)
