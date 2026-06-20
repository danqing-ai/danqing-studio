"""Unit tests for long-video storyboard planning and parsing."""
from __future__ import annotations

import unittest

from backend.engine.families.ltx.long_video_plan import build_long_video_plan, compute_extend_pass_count
from backend.engine.llm.storyboard import (
    expand_batches_for_plan,
    parse_expand_script,
    parse_plan_script,
    storyboard_quality_ok,
)


class LongVideoPlanTests(unittest.TestCase):
    def test_extend_pass_count_scales_with_target(self) -> None:
        self.assertEqual(compute_extend_pass_count(30, 8, 8), 3)
        self.assertEqual(compute_extend_pass_count(60, 8, 8), 7)
        self.assertEqual(compute_extend_pass_count(90, 8, 8), 11)

    def test_build_plan_segments(self) -> None:
        plan = build_long_video_plan(target_duration_sec=60, initial_duration_sec=8, segment_extend_sec=8)
        self.assertEqual(plan.extend_pass_count, 7)
        self.assertEqual(plan.total_segments, 8)
        self.assertEqual(plan.narrative_budget, "standard")

    def test_expand_batches_when_many_passes(self) -> None:
        plan = build_long_video_plan(target_duration_sec=90, initial_duration_sec=8, segment_extend_sec=8)
        batches = expand_batches_for_plan(plan)
        self.assertGreater(len(batches), 1)
        self.assertEqual(sum(c for _, c in batches), plan.extend_pass_count)


class StoryboardParserTests(unittest.TestCase):
    def test_parse_plan_script(self) -> None:
        text = (
            "[Anchor] Red coat detective, neon alley, cool blue rim light.\n"
            "[Beat 1] Opens on rain and footsteps.\n"
            "[Beat 2] She turns toward camera.\n"
        )
        anchor, beats = parse_plan_script(text, expected_beats=2)
        self.assertIn("Red coat", anchor)
        self.assertEqual(len(beats), 2)

    def test_parse_expand_script(self) -> None:
        text = (
            "[Opening] Anchor scene with slow dolly in, rain ambience.\n"
            "[Segment 1] She walks deeper into the alley, same red coat.\n"
            "[Segment 2] Close-up reaction, thunder rumble.\n"
        )
        opening, segs = parse_expand_script(text, expected_segments=2)
        self.assertTrue(opening)
        self.assertEqual(len(segs), 2)

    def test_storyboard_quality_ok(self) -> None:
        plan = build_long_video_plan(target_duration_sec=30, initial_duration_sec=8, segment_extend_sec=8)
        segs = [
            "She continues down the alley, same red coat, side tracking shot.",
            "Close on face under neon, rain rhythm continues.",
            "She reaches the door, hand on handle, ambient thunder.",
        ]
        beats = ["open rain", "alley walk", "door approach", "enter tension"]
        ok = storyboard_quality_ok(
            character_anchor="A woman in a red coat under neon signs in a wet alley.",
            opening_prompt="Red-coated detective enters rainy neon alley, slow dolly, ambient rain.",
            segment_prompts=segs,
            beat_sheet=beats,
            plan=plan,
        )
        self.assertTrue(ok)


if __name__ == "__main__":
    unittest.main()
