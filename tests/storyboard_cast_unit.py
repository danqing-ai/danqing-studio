"""Unit tests for multi-look character cast in long-video storyboard."""
from __future__ import annotations

import unittest

from backend.engine.llm.storyboard import KEYFRAME_REF_DIVIDER, is_structured_keyframe_visual
from backend.engine.llm.storyboard_cast import (
    compose_keyframe_with_cast,
    infer_shot_cast_looks,
    parse_character_roster,
    format_character_roster,
)


class StoryboardCastTests(unittest.TestCase):
    def test_parse_multi_look_roster(self) -> None:
        anchor = (
            "【角色·赵今麦·日常】现代白T恤，黑色短发\n"
            "---\n"
            "【角色·赵今麦·地府】素色襦裙，青竹簪\n"
            "---\n"
            "【角色·孙悟空·默认】金色锁子甲\n"
            "---\n"
            "【画风】写实电影感"
        )
        roster, style = parse_character_roster(anchor, locale="zh")
        self.assertEqual(style, "写实电影感")
        names = {c.name for c in roster}
        self.assertIn("赵今麦", names)
        zjm = next(c for c in roster if c.name == "赵今麦")
        self.assertEqual(len(zjm.looks), 2)
        labels = {lk.label for lk in zjm.looks}
        self.assertIn("日常", labels)
        self.assertIn("地府", labels)

    def test_infer_look_from_beat_tag(self) -> None:
        anchor = (
            "【角色·赵今麦·日常】白T恤\n"
            "---\n"
            "【角色·赵今麦·地府】素色襦裙\n"
            "---\n"
            "【画风】冷色"
        )
        roster, _style = parse_character_roster(anchor, locale="zh")
        cast = infer_shot_cast_looks(
            scene="低角度，赵今麦踉跄落地",
            beat="赵今麦（地府）坠入幽冥地府",
            characters=roster,
        )
        zjm = next(c for c in roster if c.name == "赵今麦")
        look_id = next(c.look_id for c in cast if c.character_id == zjm.id)
        look = next(lk for lk in zjm.looks if lk.id == look_id)
        self.assertEqual(look.label, "地府")

    def test_compose_keyframe_with_selected_look(self) -> None:
        anchor = (
            "【角色·赵今麦·日常】白T恤\n"
            "---\n"
            "【角色·赵今麦·地府】素色襦裙\n"
            "---\n"
            "【画风】冷色"
        )
        roster, style = parse_character_roster(anchor, locale="zh")
        cast = infer_shot_cast_looks(
            scene="赵今麦在卧室",
            beat="赵今麦在卧室",
            characters=roster,
        )
        composed = compose_keyframe_with_cast(
            "近景，赵今麦在卧室暖光下",
            characters=roster,
            cast=cast,
            style_anchor=style,
            locale="zh",
        )
        self.assertIn(KEYFRAME_REF_DIVIDER, composed)
        self.assertIn("白T恤", composed)
        self.assertNotIn("素色襦裙", composed)
        self.assertTrue(is_structured_keyframe_visual(composed))

    def test_format_roundtrip(self) -> None:
        anchor = "【角色·小明·校服】蓝白校服\n---\n【角色·小明·晚礼服】黑色礼服\n---\n【画风】暖色"
        roster, style = parse_character_roster(anchor, locale="zh")
        formatted = format_character_roster(roster, style, locale="zh")
        roster2, style2 = parse_character_roster(formatted, locale="zh")
        self.assertEqual(style2, style)
        self.assertEqual(len(roster2), 1)
        self.assertEqual(len(roster2[0].looks), 2)


if __name__ == "__main__":
    unittest.main()
