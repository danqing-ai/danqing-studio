"""Unit tests for script_parse artifact schemas."""
from __future__ import annotations

import unittest

from pydantic import ValidationError

from backend.engine.llm.script_parse.rules import (
    canonicalize_script_locations,
    expand_beat_plan_rows,
    location_matches,
)
from backend.engine.llm.script_parse.schemas import (
    BeatPlanLLMSchema,
    BeatPlanRowLLM,
    BeatPlanSegmentLLM,
    FiveAspectArtifact,
    ScriptArtifact,
    ScriptBeatArtifact,
    ScriptCharacterArtifact,
    ScriptSceneArtifact,
    ShotLanguageArtifact,
    ShotSpecArtifact,
    sanitize_narrative_role,
    sanitize_reachability,
    sanitize_segment_role,
    sanitize_visibility,
)


class ScriptParseSchemaTests(unittest.TestCase):
    def test_sanitize_narrative_role_reachability_alias(self) -> None:
        self.assertEqual(sanitize_narrative_role("action_wide"), "build_tension")
        self.assertEqual(sanitize_narrative_role("identity_critical"), "introduce_subject")
        self.assertEqual(sanitize_narrative_role("establish_context"), "establish_context")
        self.assertEqual(sanitize_narrative_role("not_a_role"), "")

    def test_sanitize_segment_enums(self) -> None:
        self.assertEqual(sanitize_segment_role("action_wide"), "keyframe")
        self.assertEqual(sanitize_segment_role("anchor"), "face_anchor")
        self.assertEqual(sanitize_visibility("visible"), "full_face")
        self.assertEqual(sanitize_reachability("wide"), "action_wide")

    def test_location_matches_suffix_and_interior(self) -> None:
        scene = "废弃港区老仓库"
        self.assertTrue(location_matches("老仓库内部 · 深夜", scene))
        self.assertTrue(location_matches("城郊旧公寓", "林晓的旧公寓"))
        canon = canonicalize_script_locations(
            ScriptArtifact(
                title="t",
                synopsis="synopsis long enough for schema",
                mood="m",
                style_anchor="",
                beats=[
                    ScriptBeatArtifact(
                        index=0,
                        title="b",
                        location="老仓库内部 · 深夜",
                        narrative="n",
                        enhancement_cues=["cue"],
                        suggested_shot_size="中景",
                        estimated_duration_sec=5.0,
                    ),
                    ScriptBeatArtifact(
                        index=1,
                        title="b2",
                        location=scene,
                        narrative="n2",
                        enhancement_cues=["cue"],
                        suggested_shot_size="中景",
                        estimated_duration_sec=5.0,
                    ),
                ],
                characters=[
                    ScriptCharacterArtifact(
                        name="林晓",
                        role="protagonist",
                        looks=[{"label": "常服", "body": "短发"}],
                    ),
                ],
                scenes=[ScriptSceneArtifact(name=scene, looks=[{"label": "夜", "body": "室内"}])],
            )
        )
        self.assertEqual(canon.beats[0].location, scene)

    def test_expand_beat_plan_sanitizes_segment_enums(self) -> None:
        script = ScriptArtifact(
            title="t",
            synopsis="林晓雨夜赴约老仓库，与老周联手夺证并在拂晓逃离。",
            mood="m",
            style_anchor="",
                beats=[
                    ScriptBeatArtifact(
                        index=0,
                        title="b0",
                        location="loc",
                        narrative="n0",
                        enhancement_cues=["cue"],
                        suggested_shot_size="中景",
                        estimated_duration_sec=5.0,
                    ),
                    ScriptBeatArtifact(
                        index=1,
                        title="b1",
                        location="loc",
                        narrative="n1",
                        enhancement_cues=["cue"],
                        suggested_shot_size="中景",
                        estimated_duration_sec=5.0,
                    ),
                ],
            characters=[
                ScriptCharacterArtifact(
                    name="林晓",
                    role="protagonist",
                    looks=[{"label": "常服", "body": "短发"}],
                ),
            ],
            scenes=[ScriptSceneArtifact(name="loc", looks=[{"label": "夜", "body": "室内"}])],
        )
        payload = BeatPlanLLMSchema(
            beats=[
                BeatPlanRowLLM(
                    beat_index=0,
                    shot_intent="test",
                    narrative_role="establish_context",
                    segments=[
                        BeatPlanSegmentLLM(
                            role="action_wide",
                            duration_sec=5.0,
                            characters_on_screen=["林晓"],
                            start_visibility="visible",
                            end_visibility="face",
                            reachability="wide",
                        ),
                    ],
                ),
            ],
        )
        plan = expand_beat_plan_rows(script, payload.beats)
        seg = plan.beats[0].segments[0]
        self.assertEqual(seg.role, "keyframe")
        self.assertEqual(seg.start_visibility, "full_face")
        self.assertEqual(seg.end_visibility, "full_face")
        self.assertEqual(seg.reachability, "action_wide")

    def test_expand_beat_plan_sanitizes_narrative_role(self) -> None:
        script = ScriptArtifact(
            title="t",
            synopsis="林晓雨夜赴约老仓库，与老周联手夺证并在拂晓逃离。",
            mood="m",
            style_anchor="",
            beats=[
                ScriptBeatArtifact(
                    index=0,
                    title="b0",
                    location="loc",
                    narrative="n0",
                    enhancement_cues=["cue"],
                    suggested_shot_size="中景",
                    estimated_duration_sec=5.0,
                ),
                ScriptBeatArtifact(
                    index=1,
                    title="b1",
                    location="loc",
                    narrative="n1",
                    enhancement_cues=["cue"],
                    suggested_shot_size="中景",
                    estimated_duration_sec=5.0,
                ),
            ],
            characters=[
                ScriptCharacterArtifact(
                    name="林晓",
                    role="protagonist",
                    looks=[{"label": "常服", "body": "短发"}],
                ),
            ],
            scenes=[ScriptSceneArtifact(name="loc", looks=[{"label": "夜", "body": "室内"}])],
        )
        payload = BeatPlanLLMSchema(
            beats=[
                BeatPlanRowLLM(
                    beat_index=0,
                    shot_intent="test",
                    narrative_role="action_wide",
                    segments=[
                        BeatPlanSegmentLLM(
                            role="face_anchor",
                            duration_sec=5.0,
                            characters_on_screen=["林晓"],
                            reachability="identity_critical",
                        ),
                    ],
                ),
            ],
        )
        plan = expand_beat_plan_rows(script, payload.beats)
        self.assertEqual(plan.beats[0].narrative_role, "build_tension")

    def test_shot_spec_subject_closure(self) -> None:
        spec = ShotSpecArtifact(
            segment_index=0,
            beat_index=0,
            five_aspect=FiveAspectArtifact(
                subject="林晓站在门口",
                subject_motion="林晓推门进入",
                scene="旧仓库",
                spatial_framing="中景",
                camera="static",
            ),
            shot_language=ShotLanguageArtifact(camera_movement="static"),
            video_prompt="林晓缓慢推门，镜头固定",
            start_visual="林晓站在锈迹铁门前",
            characters_on_screen=["林晓"],
        )
        self.assertIn("林晓", spec.start_visual)

    def test_shot_spec_rejects_missing_subject(self) -> None:
        with self.assertRaises(ValidationError):
            ShotSpecArtifact(
                segment_index=1,
                beat_index=0,
                five_aspect=FiveAspectArtifact(
                    subject="空走廊",
                    subject_motion="镜头推进",
                    scene="仓库",
                    spatial_framing="远景",
                    camera="dolly_in",
                ),
                shot_language=ShotLanguageArtifact(camera_movement="dolly_in"),
                video_prompt="镜头缓慢推进",
                start_visual="空走廊",
                characters_on_screen=["林晓"],
            )

    def test_static_camera_motion_conflict(self) -> None:
        with self.assertRaises(ValidationError):
            ShotSpecArtifact(
                segment_index=2,
                beat_index=0,
                five_aspect=FiveAspectArtifact(
                    subject="林晓",
                    subject_motion="林晓转身",
                    scene="仓库",
                    spatial_framing="近景",
                    camera="static",
                ),
                shot_language=ShotLanguageArtifact(camera_movement="static"),
                video_prompt="镜头 pan left 跟随林晓",
                start_visual="林晓侧脸",
                characters_on_screen=["林晓"],
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
