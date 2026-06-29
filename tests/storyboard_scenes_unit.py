"""Unit tests for storyboard scene roster + shot binding."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.engine.llm.scene_entity_extract import scenes_from_entity_json
from backend.engine.llm.storyboard_scenes import (
    infer_shot_scene_look,
    parse_scene_beat_location,
    parse_scene_roster,
    scenes_from_beat_locations,
)

BEATS = [
    "【远景】阴森古宅外·夜，枯树摇曳，薄雾弥漫，赵今麦立于石阶",
    "【中景】阴森古宅外·夜，赵今麦推门进入",
    "【全景】古宅大厅·夜，烛火摇曳，空荡长廊",
]

assert parse_scene_beat_location(BEATS[0]) == "阴森古宅外·夜"
assert parse_scene_beat_location(BEATS[2]).startswith("古宅大厅")

fallback = scenes_from_beat_locations(BEATS, locale="zh")
assert len(fallback) >= 2, "dedupe locations into scene entities"

roster = scenes_from_entity_json(
    """{
  "scenes": [
    {
      "name": "阴森古宅",
      "looks": [
        {
          "label": "外·夜",
          "environment": "枯树、石阶、薄雾",
          "set_dressing": "哥特式老宅轮廓，冷青月光"
        }
      ]
    },
    {
      "name": "古宅大厅",
      "looks": [
        {
          "label": "内·夜",
          "environment": "空荡长廊、烛火",
          "set_dressing": "木质楼梯与旧家具"
        }
      ]
    }
  ]
}""",
    locale="zh",
)
assert len(roster) == 2
assert roster[0].looks[0].body.startswith("环境")

binding = infer_shot_scene_look(beat=BEATS[0], scenes=roster)
assert binding is not None
assert binding.scene_id == roster[0].id

binding2 = infer_shot_scene_look(beat=BEATS[2], scenes=roster, prev=binding)
assert binding2 is not None
assert binding2.scene_id != binding.scene_id

print("storyboard_scenes_unit: OK")
