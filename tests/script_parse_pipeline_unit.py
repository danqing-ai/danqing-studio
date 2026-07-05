"""Unit tests for script_parse pipeline helpers."""
from __future__ import annotations

from backend.engine.llm.script_parse.pipeline import merge_beat_shots


def test_merge_beat_shots_replaces_one_beat_and_reorders() -> None:
    existing = [
        {"id": "a", "order": 0, "narrative_beat_index": 0},
        {"id": "b", "order": 1, "narrative_beat_index": 1},
        {"id": "c", "order": 2, "narrative_beat_index": 2},
    ]
    new = [
        {"id": "b2", "order": 99, "narrative_beat_index": 1},
        {"id": "b3", "order": 100, "narrative_beat_index": 1},
    ]
    merged = merge_beat_shots(existing, new, beat_index=1)
    assert [s["id"] for s in merged] == ["a", "b2", "b3", "c"]
    assert [s["order"] for s in merged] == [0, 1, 2, 3]
