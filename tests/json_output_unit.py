"""Unit tests for LLM JSON extraction."""
from __future__ import annotations

import json
import unittest

from backend.engine.llm.json_output import extract_json_object


class JsonOutputTests(unittest.TestCase):
    def test_parse_plain_object(self) -> None:
        data = extract_json_object('{"synopsis": "hello", "beats": []}')
        self.assertEqual(data["synopsis"], "hello")

    def test_parse_markdown_fence(self) -> None:
        raw = '```json\n{"mood": "tense"}\n```'
        self.assertEqual(extract_json_object(raw)["mood"], "tense")

    def test_parse_with_prefix_suffix(self) -> None:
        raw = 'Here is the result:\n{"style": "film"}\nDone.'
        self.assertEqual(extract_json_object(raw)["style"], "film")

    def test_parse_with_english_preamble(self) -> None:
        raw = 'Okay, planning first.\n{"synopsis": "测试", "mood": "紧张", "style": "胶片", "characters": [], "beats": [{"title": "a", "shot_size": "中景", "location": "卧室", "visual": "林晓在卧室"}, {"title": "b", "shot_size": "远景", "location": "山", "visual": "林晓登山"}]}'
        data = extract_json_object(raw)
        self.assertEqual(data["synopsis"], "测试")
        self.assertEqual(len(data["beats"]), 2)

    def test_rejects_invalid(self) -> None:
        with self.assertRaises(ValueError):
            extract_json_object("not json at all")

    def test_roundtrip_sample(self) -> None:
        payload = {"synopsis": "测试", "beats": [{"visual": "画面"}]}
        raw = json.dumps(payload, ensure_ascii=False)
        self.assertEqual(extract_json_object(raw), payload)


if __name__ == "__main__":
    unittest.main()
