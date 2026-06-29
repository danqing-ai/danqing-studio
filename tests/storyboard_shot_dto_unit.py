"""Shot DTO preserves structured parse fields."""
from __future__ import annotations

import unittest

from backend.core.contracts import LongVideoStoryboardShotDTO


class StoryboardShotDtoTests(unittest.TestCase):
    def test_location_and_beat_index_roundtrip(self) -> None:
        row = {
            "id": "shot_1",
            "order": 0,
            "visual_prompt": "visual",
            "motion_prompt": "motion",
            "video_prompt": "motion",
            "location": "卧室 · 深夜",
            "shot_size": "特写",
            "narrative_beat_index": 2,
        }
        dto = LongVideoStoryboardShotDTO(**row)
        self.assertEqual(dto.location, "卧室 · 深夜")
        self.assertEqual(dto.shot_size, "特写")
        self.assertEqual(dto.narrative_beat_index, 2)


if __name__ == "__main__":
    unittest.main()
