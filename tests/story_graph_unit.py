"""Unit tests for story graph rule fallback."""
from __future__ import annotations

import unittest

from backend.engine.llm.story_graph import _rule_story_graph


class StoryGraphTests(unittest.TestCase):
    def test_beat_zero_has_protagonist_visibility(self):
        beats = ["门铃 | 远景 | 客厅·夜 | 李明听到门铃，走向门口"]
        graph = _rule_story_graph(beats, "- 李明\n  30岁男性")
        self.assertIn(0, graph)
        self.assertIn("李明", graph[0]["characters_on_screen"])
        self.assertNotEqual(graph[0]["start_visibility"], "invisible")


if __name__ == "__main__":
    unittest.main()
