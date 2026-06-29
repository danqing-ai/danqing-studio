"""Unit tests for deterministic I2V motion prompt derivation."""
from __future__ import annotations

import unittest

from backend.engine.llm.motion_prompt import motion_prompt_from_beat
from backend.engine.llm.storyboard import dual_pairs_from_beats


class MotionPromptTests(unittest.TestCase):
    def test_motion_from_structured_beat(self) -> None:
        visual = "【中景】木屋内，赵今麦低头看手机，暖黄台灯光从左侧照脸"
        motion = motion_prompt_from_beat(visual, locale="zh")
        self.assertIn("赵今麦", motion)
        self.assertIn("镜头", motion)
        self.assertNotEqual(motion.strip(), visual.strip())

    def test_wide_shot_camera(self) -> None:
        visual = "【远景】晨曦下，赵今麦站在木屋前"
        motion = motion_prompt_from_beat(visual, locale="zh")
        self.assertIn("推近", motion)

    def test_dual_pairs_motion_differs_from_visual(self) -> None:
        beats = [
            "【远景】晨曦下，赵今麦站在木屋前",
            "【中景】木屋内，赵今麦坐在桌前看手机",
            "【特写】赵今麦凝视手机屏幕",
        ]
        pairs = dual_pairs_from_beats(beats, 3, locale="zh")
        for visual, motion in pairs:
            self.assertNotEqual(visual, motion)
            self.assertIn("镜头", motion)


if __name__ == "__main__":
    unittest.main()
