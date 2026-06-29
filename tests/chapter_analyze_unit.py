"""Unit tests for novel chapter / brief analysis JSON parsing (no LLM)."""
from __future__ import annotations

import json
import unittest

from backend.engine.llm.chapter_analyze import (
    MAX_SCENES,
    MIN_SCENES,
    beat_to_sheet_line,
    clamp_scene_count,
    merge_partial_beats,
    merge_plan_and_roster,
    parse_chapter_analyze_response,
    parse_chapter_chunk_json,
    parse_chapter_plan_response,
    parse_structured_beat,
    roster_from_analyze_payload,
    split_chapter_chunks,
    validate_chapter_text,
)
from backend.engine.llm.schemas.long_video import (
    BeatSchema,
    ChapterAnalyzeSchema,
    ChapterPlanSchema,
    ChapterRosterSchema,
)


SAMPLE_ANALYZE_JSON = {
    "synopsis": "赵今麦挑战孙悟空失败，坠入地府被嘲笑。",
    "mood": "热血逆袭，由挫败转向觉醒。",
    "style": "写实电影感，35mm 浅景深",
    "characters": [
        {
            "name": "赵今麦",
            "looks": [
                {
                    "label": "日常",
                    "role": "主角",
                    "appearance": "黑色短发，瘦削体型",
                    "wardrobe": "白T恤",
                }
            ],
        },
        {
            "name": "孙悟空",
            "looks": [
                {
                    "label": "云端",
                    "role": "对手",
                    "appearance": "金色毛发",
                    "wardrobe": "红色战袍",
                }
            ],
        },
        {
            "name": "阎罗王",
            "looks": [
                {
                    "label": "大殿",
                    "role": "配角",
                    "appearance": "威严长须",
                    "wardrobe": "冥府王袍",
                }
            ],
        },
    ],
    "beats": [
        {
            "title": "卧室刷手机",
            "shot_size": "中景",
            "location": "卧室/夜",
            "narrative": "赵今麦在卧室刷手机，豆包对话框弹出挑战建议。",
        },
        {
            "title": "登山",
            "shot_size": "远景",
            "location": "云雾山巅/晨",
            "narrative": "赵今麦独自走上云雾山巅，远处孙悟空剪影。",
        },
        {
            "title": "云端对峙",
            "shot_size": "全景",
            "location": "云端",
            "narrative": "孙悟空云端盘坐，指尖拈着一根毫毛。",
        },
        {
            "title": "被击飞",
            "shot_size": "中景",
            "location": "山巅",
            "narrative": "毫毛化作分身，赵今麦被击飞。",
        },
        {
            "title": "坠入地府",
            "shot_size": "远景",
            "location": "幽冥地府",
            "narrative": "赵今麦坠入幽冥地府，鬼火环绕。",
        },
        {
            "title": "阎罗嘲笑",
            "shot_size": "近景",
            "location": "地府大殿",
            "narrative": "阎罗王端坐案后，掩口嘲笑。",
        },
    ],
}


class ChapterAnalyzeUnitTests(unittest.TestCase):
    def test_validate_chapter_text_empty(self) -> None:
        with self.assertRaises(ValueError):
            validate_chapter_text("   ")

    def test_validate_script_too_short(self) -> None:
        from backend.engine.llm.chapter_analyze import validate_script_text

        with self.assertRaises(ValueError):
            validate_script_text("短")

    def test_needs_script_expand(self) -> None:
        from backend.engine.llm.chapter_analyze import needs_script_expand, SCRIPT_EXPAND_CHAR_THRESHOLD

        self.assertTrue(needs_script_expand("a" * (SCRIPT_EXPAND_CHAR_THRESHOLD - 1)))
        self.assertFalse(needs_script_expand("a" * SCRIPT_EXPAND_CHAR_THRESHOLD))

    def test_parse_chapter_analyze_json(self) -> None:
        raw = json.dumps(SAMPLE_ANALYZE_JSON, ensure_ascii=False)
        synopsis, mood, anchor, beats, style = parse_chapter_analyze_response(raw, locale="zh")
        self.assertIn("赵今麦", synopsis)
        self.assertIn("逆袭", mood)
        self.assertIn("赵今麦", anchor)
        self.assertIn("日常", anchor)
        self.assertIn("写实", style)
        self.assertEqual(len(beats), 6)
        self.assertIn("豆包", beats[0])

    def test_parse_chapter_analyze_json_respects_llm_character_roster(self) -> None:
        payload = dict(SAMPLE_ANALYZE_JSON)
        payload["characters"] = [payload["characters"][0]]
        synopsis, _, anchor, beats, _ = parse_chapter_analyze_response(
            json.dumps(payload, ensure_ascii=False),
            locale="zh",
        )
        self.assertIn("赵今麦", synopsis)
        self.assertIn("赵今麦", anchor)
        self.assertNotIn("孙悟空", anchor)
        self.assertEqual(len(beats), 6)

    def test_parse_chapter_plan_json(self) -> None:
        plan = {
            "synopsis": "赵今麦挑战孙悟空失败，坠入地府。",
            "mood": "热血逆袭",
            "style": "写实电影感",
            "beats": SAMPLE_ANALYZE_JSON["beats"],
        }
        synopsis, mood, beats, style = parse_chapter_plan_response(
            json.dumps(plan, ensure_ascii=False),
        )
        self.assertIn("赵今麦", synopsis)
        self.assertEqual(len(beats), 6)
        self.assertEqual(style, "写实电影感")

    def test_merge_plan_and_roster(self) -> None:
        plan = ChapterPlanSchema.model_validate(
            {
                "synopsis": "赵今麦挑战孙悟空。",
                "mood": "逆袭",
                "style": "写实",
                "beats": SAMPLE_ANALYZE_JSON["beats"],
            }
        )
        roster = ChapterRosterSchema.model_validate({"characters": SAMPLE_ANALYZE_JSON["characters"]})
        merged = merge_plan_and_roster(plan, roster)
        names = {c.name for c in merged.characters}
        self.assertEqual(names, {"赵今麦", "孙悟空", "阎罗王"})
        _, _, anchor = roster_from_analyze_payload(merged, locale="zh")
        self.assertIn("孙悟空", anchor)

    def test_format_beats_for_roster_user(self) -> None:
        from backend.engine.llm.chapter_analyze import _format_beats_for_roster_user

        block = _format_beats_for_roster_user(["卧室 | 中景 | 卧室/夜 | 赵今麦刷手机"])
        self.assertIn("1.", block)
        self.assertIn("赵今麦", block)

    def test_parse_chapter_chunk_json(self) -> None:
        chunk = {
            "beats": [
                {
                    "title": "卧室",
                    "shot_size": "近景",
                    "location": "卧室/夜",
                    "narrative": "赵今麦刷手机",
                }
            ]
        }
        lines = parse_chapter_chunk_json(json.dumps(chunk, ensure_ascii=False))
        self.assertEqual(len(lines), 1)
        self.assertIn("卧室", lines[0])

    def test_roster_from_payload(self) -> None:
        payload = ChapterAnalyzeSchema.model_validate(SAMPLE_ANALYZE_JSON)
        roster, style, anchor = roster_from_analyze_payload(payload, locale="zh")
        self.assertEqual(style, "写实电影感，35mm 浅景深")
        self.assertEqual({c.name for c in roster}, {"赵今麦", "孙悟空", "阎罗王"})
        self.assertIn("【角色·赵今麦·日常】", anchor)

    def test_beat_to_sheet_line(self) -> None:
        line = beat_to_sheet_line(
            BeatSchema(
                title="卧室",
                shot_size="中景",
                location="卧室/夜",
                visual="赵今麦刷手机",
            )
        )
        self.assertEqual(line, "卧室 | 中景 | 卧室/夜 | 赵今麦刷手机")

    def test_parse_structured_beat(self) -> None:
        title, beat = parse_structured_beat(
            "卧室刷手机 | 中景 | 卧室/夜 | 赵今麦在卧室刷手机"
        )
        self.assertEqual(title, "卧室刷手机")
        self.assertIn("中景", beat)
        self.assertIn("卧室", beat)
        self.assertIn("赵今麦", beat)

    def test_split_chapter_chunks_single(self) -> None:
        text = "短章节。" * 10
        chunks = split_chapter_chunks(text, chunk_size=3500)
        self.assertEqual(len(chunks), 1)

    def test_split_chapter_chunks_multi(self) -> None:
        para = "这是一段较长的叙事文字，包含可拍摄的场景与人物动作。" * 20
        text = "\n\n".join([para] * 8)
        chunks = split_chapter_chunks(text, chunk_size=3500)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 3600)

    def test_merge_partial_beats_dedupes(self) -> None:
        merged = merge_partial_beats([
            ["赵今麦在卧室刷手机", "她独自走上山巅"],
            ["她独自走上山巅", "孙悟空云端盘坐"],
        ])
        self.assertEqual(merged, ["赵今麦在卧室刷手机", "她独自走上山巅", "孙悟空云端盘坐"])

    def test_clamp_scene_count_bounds(self) -> None:
        self.assertEqual(clamp_scene_count(MIN_SCENES), MIN_SCENES)
        self.assertEqual(clamp_scene_count(MAX_SCENES), MAX_SCENES)
        self.assertEqual(clamp_scene_count(8, max_scenes=12), 8)
        with self.assertRaises(ValueError):
            clamp_scene_count(1)
        self.assertEqual(clamp_scene_count(MAX_SCENES + 1), MAX_SCENES)
        self.assertEqual(clamp_scene_count(10, max_scenes=9), 9)

    def test_rejects_placeholder_values(self) -> None:
        bad = dict(SAMPLE_ANALYZE_JSON)
        bad["synopsis"] = "<2-3 sentences: plot summary>"
        with self.assertRaises(ValueError):
            parse_chapter_analyze_response(json.dumps(bad, ensure_ascii=False))

    def test_rejects_plain_beat_line(self) -> None:
        with self.assertRaises(ValueError):
            parse_structured_beat("赵今麦在卧室刷手机")

    def test_paragraph_scene_floor(self) -> None:
        from backend.engine.llm.chapter_analyze import _paragraph_scene_floor

        text = "\n\n".join(["第一段足够长的叙事场景描述。" * 3] * 4)
        self.assertEqual(_paragraph_scene_floor(text), 4)

    def test_roster_rejects_singular_look_field(self) -> None:
        payload = {
            "characters": [
                {
                    "name": "判官",
                    "look": {
                        "label": "大殿",
                        "role": "配角",
                        "appearance": "威严",
                        "wardrobe": "判官服饰",
                    },
                }
            ]
        }
        with self.assertRaises(ValueError):
            ChapterRosterSchema.model_validate(payload)


if __name__ == "__main__":
    unittest.main()
