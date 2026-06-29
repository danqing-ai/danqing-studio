"""Fixed chapter-parse benchmark cases (speed + quality regression)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures" / "chapter_scripts"


@dataclass(frozen=True)
class ChapterParseBenchmarkCase:
    case_id: str
    title: str
    script_relpath: str
    target_duration_sec: float = 60.0
    segment_duration_sec: float = 5.0
    min_beats: int = 4
    min_shots: int = 8
    locale: str = "zh"

    @property
    def script_path(self) -> Path:
        return FIXTURES_DIR / self.script_relpath

    def load_script(self) -> str:
        text = self.script_path.read_text(encoding="utf-8").strip()
        if not text:
            raise ValueError(f"empty script fixture: {self.script_path}")
        return text


CHAPTER_PARSE_BENCHMARK_CASES: tuple[ChapterParseBenchmarkCase, ...] = (
    ChapterParseBenchmarkCase(
        case_id="wukong",
        title="大战悟空",
        script_relpath="wukong_battle.txt",
        target_duration_sec=60.0,
        segment_duration_sec=5.0,
        min_beats=4,
        min_shots=10,
    ),
    ChapterParseBenchmarkCase(
        case_id="rainy_night",
        title="雨夜访客",
        script_relpath="rainy_night_visitor.txt",
        target_duration_sec=60.0,
        segment_duration_sec=5.0,
        min_beats=4,
        min_shots=10,
    ),
)

CASE_BY_ID = {c.case_id: c for c in CHAPTER_PARSE_BENCHMARK_CASES}


def resolve_cases(case: str) -> list[ChapterParseBenchmarkCase]:
    if case == "all":
        return list(CHAPTER_PARSE_BENCHMARK_CASES)
    if case not in CASE_BY_ID:
        raise ValueError(f"unknown case {case!r}; use wukong|rainy_night|all")
    return [CASE_BY_ID[case]]
