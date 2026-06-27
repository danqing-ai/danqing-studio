"""Unit tests for novel chapter analysis (no LLM)."""
from __future__ import annotations

import unittest

from backend.engine.llm.chapter_analyze import (
    MAX_SCENES,
    MIN_SCENES,
    clamp_scene_count,
    merge_partial_beats,
    parse_chapter_analyze_script,
    split_chapter_chunks,
    validate_chapter_text,
)


SAMPLE_ANALYZE_SCRIPT = """
[Synopsis] 赵今麦挑战孙悟空失败，坠入地府被嘲笑。
[Anchor]
【角色·赵今麦·日常】黑色短发，白T恤，神情认真
---
【角色·孙悟空·云端】金箍，红色战袍
---
【画风】写实电影感，35mm 浅景深
[Beat 1] 赵今麦在卧室刷手机，豆包对话框弹出挑战建议。
[Beat 2] 她独自走上云雾山巅，远处孙悟空剪影。
[Beat 3] 孙悟空云端盘坐，指尖拈着一根毫毛。
[Beat 4] 毫毛化作分身，赵今麦被击飞。
[Beat 5] 赵今麦坠入幽冥地府，鬼火环绕。
[Beat 6] 阎罗王端坐案后，掩口嘲笑。
"""


class ChapterAnalyzeUnitTests(unittest.TestCase):
    def test_validate_chapter_text_empty(self) -> None:
        with self.assertRaises(ValueError):
            validate_chapter_text("   ")

    def test_parse_chapter_analyze_script(self) -> None:
        synopsis, anchor, beats = parse_chapter_analyze_script(SAMPLE_ANALYZE_SCRIPT)
        self.assertIn("赵今麦", synopsis)
        self.assertIn("赵今麦", anchor)
        self.assertEqual(len(beats), 6)
        self.assertIn("豆包", beats[0])

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
        with self.assertRaises(ValueError):
            clamp_scene_count(1)
        with self.assertRaises(ValueError):
            clamp_scene_count(MAX_SCENES + 1)


if __name__ == "__main__":
    unittest.main()
