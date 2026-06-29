"""Unit tests for scene entity JSON extraction."""
from __future__ import annotations

import json
import unittest

from backend.engine.llm.scene_entity_extract import scenes_from_entity_json


class SceneEntityJsonTests(unittest.TestCase):
    def test_scenes_from_entity_json(self) -> None:
        payload = {
            "scenes": [
                {
                    "name": "卧室",
                    "looks": [
                        {
                            "label": "深夜",
                            "environment": "狭小卧室，冷蓝手机光",
                            "set_dressing": "床、书桌、手机",
                        }
                    ],
                },
                {
                    "name": "云雾山",
                    "looks": [
                        {
                            "label": "夜",
                            "environment": "陡峭山径，湿冷山风",
                            "set_dressing": "岩壁、云雾",
                        }
                    ],
                },
            ]
        }
        scenes = scenes_from_entity_json(json.dumps(payload, ensure_ascii=False), locale="zh")
        self.assertEqual(len(scenes), 2)
        names = {sc.name for sc in scenes}
        self.assertIn("卧室", names)
        self.assertIn("云雾山", names)
        bedroom = next(sc for sc in scenes if sc.name == "卧室")
        self.assertEqual(bedroom.looks[0].label, "深夜")
        self.assertIn("环境：", bedroom.looks[0].body)

    def test_disambiguates_duplicate_look_labels(self) -> None:
        payload = {
            "scenes": [
                {
                    "name": "卧室",
                    "looks": [
                        {
                            "label": "深夜",
                            "environment": "昏暗室内，手机弹窗",
                            "set_dressing": "床、手机",
                        },
                        {
                            "label": "深夜",
                            "environment": "手指悬于确认键",
                            "set_dressing": "确认键",
                        },
                    ],
                }
            ]
        }
        scenes = scenes_from_entity_json(json.dumps(payload, ensure_ascii=False), locale="zh")
        bedroom = scenes[0]
        labels = [lk.label for lk in bedroom.looks]
        self.assertEqual(len(labels), len(set(labels)))
        self.assertEqual(len({lk.id for lk in bedroom.looks}), 2)

    def test_merges_similar_scene_names(self) -> None:
        payload = {
            "scenes": [
                {
                    "name": "地府深渊",
                    "looks": [
                        {
                            "label": "深夜",
                            "environment": "冰冷石壁",
                            "set_dressing": "石壁",
                        }
                    ],
                },
                {
                    "name": "地下深渊",
                    "looks": [
                        {
                            "label": "深夜",
                            "environment": "漆黑深渊",
                            "set_dressing": "阴风",
                        }
                    ],
                },
            ]
        }
        scenes = scenes_from_entity_json(json.dumps(payload, ensure_ascii=False), locale="zh")
        self.assertEqual(len(scenes), 1)
        self.assertEqual(len(scenes[0].looks), 2)
        self.assertEqual(len({lk.label for lk in scenes[0].looks}), 2)


if __name__ == "__main__":
    unittest.main()
