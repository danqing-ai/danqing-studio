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

    def test_infer_cast_uses_context_label_overlap(self) -> None:
        """Names-only beats; look binding uses narrative/location token overlap on look labels."""
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
            beat="赵今麦坠入幽冥地府",
            characters=roster,
        )
        zjm = next(c for c in roster if c.name == "赵今麦")
        look_id = next(c.look_id for c in cast if c.character_id == zjm.id)
        look = next(lk for lk in zjm.looks if lk.id == look_id)
        self.assertEqual(look.label, "地府")

    def test_supplement_roster_from_shots_adds_extras(self) -> None:
        from backend.engine.llm.storyboard_cast import supplement_roster_from_shots

        roster = [
            {
                "id": "char_a",
                "name": "林晓",
                "default_look_id": "look_a",
                "looks": [{"id": "look_a", "label": "默认", "body": "定位：lead | 外貌：年轻女性"}],
            }
        ]
        shots = [{"characters_on_screen": ["林晓", "护卫"]}]
        out = supplement_roster_from_shots(roster, shots, locale="zh")
        names = {row["name"] for row in out}
        self.assertIn("护卫", names)

    def test_infer_cast_uses_scene_hints_without_prev(self) -> None:
        anchor = (
            "【角色·林晓·公寓】居家便服\n"
            "---\n"
            "【角色·林晓·仓库现场】工装外套\n"
            "---\n"
            "【画风】冷色"
        )
        roster, _ = parse_character_roster(anchor, locale="zh")
        zjm = next(c for c in roster if c.name == "林晓")
        cast = infer_shot_cast_looks(
            scene="港区废弃仓库外，雨水打湿地面",
            beat="林晓推开生锈铁门",
            characters=[zjm],
            prev=None,
            scene_hints=["港区废弃仓库"],
        )
        look_id = cast[0].look_id
        look = next(lk for lk in zjm.looks if lk.id == look_id)
        self.assertEqual(look.label, "仓库现场")

    def test_strip_name_look_tags(self) -> None:
        from backend.engine.llm.storyboard_cast import strip_name_look_tags

        raw = "【特写】赵今麦（白 T 恤 黑短裤）坐在床边刷手机"
        self.assertEqual(strip_name_look_tags(raw), "【特写】赵今麦坐在床边刷手机")

    def test_sanitize_beat_sheet(self) -> None:
        from backend.engine.llm.chapter_analyze import sanitize_beat_sheet

        rows = sanitize_beat_sheet(
            ["Opening | close-up | apartment night | Alex（pajamas）scrolls phone and taps confirm"]
        )
        self.assertIn("Alex", rows[0])
        self.assertNotIn("pajamas", rows[0])

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

    def test_rejects_prose_as_character(self) -> None:
        prose = (
            "赵今麦深夜收到挑战孙悟空的短信后，穿红薄外套攀登山顶。"
            "在云雾之巅遭遇孙悟空分身，坠落时穿越黑暗坠入阎罗王大殿。"
        )
        roster, _style = parse_character_roster(prose, locale="zh")
        self.assertEqual(roster, [])

    def test_cast_lock_ignores_beat_scene_prompt_for_names(self) -> None:
        from backend.engine.llm.storyboard_pipeline import _apply_cast_lock

        character_dtos = [
            {
                "id": "char_zjm",
                "name": "赵今麦",
                "looks": [{"id": "look_zjm", "label": "日常", "body": "白T恤"}],
                "default_look_id": "look_zjm",
            },
            {
                "id": "char_wk",
                "name": "孙悟空",
                "looks": [{"id": "look_wk", "label": "默认", "body": "金甲"}],
                "default_look_id": "look_wk",
            },
        ]
        shots = [
            {
                "visual_prompt": "【特写】赵今麦刷手机",
                "scene_prompt": "赵今麦挑战孙悟空，按下确认键",
                "video_prompt": "赵今麦犹豫后按下按钮",
                "characters_on_screen": ["赵今麦"],
            }
        ]
        locked = _apply_cast_lock(shots, character_dtos=character_dtos, scene_dtos=[])
        cast_ids = {row["character_id"] for row in locked[0]["cast_looks"]}
        self.assertEqual(cast_ids, {"char_zjm"})

    def test_normalize_placeholder_look_label(self) -> None:
        from backend.engine.llm.storyboard_cast import normalize_look_label

        self.assertEqual(
            normalize_look_label("（无标签）", locale="zh", name="赵今麦", wardrobe="白 T 恤 黑短裤"),
            "白T恤黑短裤",
        )
        self.assertEqual(
            normalize_look_label("卧室便装", locale="zh", name="赵今麦"),
            "卧室便装",
        )


if __name__ == "__main__":
    unittest.main()
