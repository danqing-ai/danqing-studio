"""Unit tests for segment shot planning (no LLM)."""
from __future__ import annotations

import unittest

from backend.engine.llm.chapter_segment_plan import (
    _batch_map_complete,
    _rule_reachability,
    _rule_split_beat,
    _split_beat_fields,
    _split_duration_parts,
    expand_segments_with_anchor_plan,
    plan_segments_from_beats,
)


class SegmentPlanTests(unittest.TestCase):
    def test_split_duration_over_cap(self):
        parts = _split_duration_parts(18.0, max_clip_sec=10.0)
        self.assertEqual(len(parts), 2)
        self.assertAlmostEqual(sum(parts), 18.0, places=1)

    def test_split_duration_under_cap(self):
        parts = _split_duration_parts(8.0, max_clip_sec=10.0)
        self.assertEqual(parts, [8.0])

    def test_plan_marks_continuation_tail_mode(self):
        beats = [
            "对决 | 全景 | 山巅·晨 | 赵今麦与孙悟空长时间交锋，云雾翻涌，多次跃起对撞",
        ]
        segs = expand_segments_with_anchor_plan(
            beats,
            {0: 18.0},
            {0: "establishing"},
            max_clip_sec=10.0,
        )
        cont = [s for s in segs if s.segment_role == "tail_continuation"]
        self.assertGreaterEqual(len(cont), 1)
        self.assertEqual(cont[0].start_frame_mode, "prev_segment_tail")

    def test_split_beat_fields_pipe_format(self):
        title, shot, loc, narrative = _split_beat_fields(
            "卧室 | 中景 | 卧室·夜 | 赵今麦刷手机收到挑战"
        )
        self.assertEqual(title, "卧室")
        self.assertEqual(shot, "中景")
        self.assertEqual(loc, "卧室·夜")
        self.assertIn("赵今麦", narrative)

    def test_rule_reachability_wide(self):
        self.assertEqual(_rule_reachability("远景", "空镜，城市夜景", beat_index=1), "empty")
        self.assertEqual(_rule_reachability("远景", "赵今麦走进房间"), "action_wide")

    def test_expand_inserts_face_anchor(self):
        beats = ["入场 | 远景 | 大厅 | 赵今麦缓步走入大厅，环顾四周"]
        reach = {0: "action_wide"}
        durations = {0: 10.0}
        segs = expand_segments_with_anchor_plan(
            beats,
            durations,
            reach,
            max_clip_sec=10.0,
        )
        roles = [s.segment_role for s in segs]
        self.assertIn("face_anchor", roles)
        for seg in segs:
            self.assertLessEqual(seg.duration_sec, 10.0)
        anchor = next(s for s in segs if s.segment_role == "face_anchor")
        self.assertEqual(anchor.shot_size, "特写")
        post = [s for s in segs if s.segment_role == "post_anchor"]
        if post:
            self.assertEqual(post[0].start_frame_mode, "anchor_link")
            self.assertEqual(post[0].face_anchor_shot_id, anchor.face_anchor_shot_id)

    def test_rule_split_beat_identity(self):
        rows = _rule_split_beat(
            beat_index=0,
            beat_dur=10.0,
            shot_size="中景",
            reachability="identity_critical",
            max_clip_sec=10.0,
        )
        roles = [r.role for r in rows]
        self.assertIn("face_anchor", roles)

    def test_batch_map_complete_bisects_on_missing(self):
        from backend.engine.llm.chapter_segment_plan import PlannedSegment

        segments = [
            PlannedSegment(
                segment_index=i,
                narrative_beat_index=0,
                segment_group_id="beat_0",
                segment_group_index=i,
                duration_sec=5.0,
                segment_role="keyframe",
                start_frame_mode="keyframe",
                flf_mode="none",
                face_anchor_shot_id="",
                title="t",
                shot_size="中景",
                location="loc",
                narrative="n",
            )
            for i in range(3)
        ]
        calls: list[int] = []

        def invoke(batch: list[PlannedSegment]) -> tuple[dict[int, str], int]:
            calls.append(len(batch))
            if len(batch) > 1:
                return {batch[0].segment_index: "a"}, 1
            return {batch[0].segment_index: "b"}, 1

        out, llm_calls = _batch_map_complete(
            segments,
            batch_size=3,
            invoke=invoke,
            error_label="test",
        )
        self.assertEqual(out, {0: "b", 1: "b", 2: "b"})
        self.assertGreaterEqual(llm_calls, 3)
        self.assertIn(3, calls)
        self.assertIn(1, calls)


if __name__ == "__main__":
    unittest.main()
