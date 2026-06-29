"""Unittest wrapper for fixed chapter-parse benchmark cases (requires local LLM)."""
from __future__ import annotations

import unittest

from tests.chapter_parse_benchmark import run_benchmark
from tests.chapter_parse_benchmark_cases import CHAPTER_PARSE_BENCHMARK_CASES


class ChapterParseBenchmarkTests(unittest.TestCase):
    def test_fixed_cases_benchmark(self) -> None:
        from tests.long_video_chapter_analyze_integration import _load_llm_service

        svc = _load_llm_service()
        if not svc.is_available():
            self.skipTest("no local LLM installed in workspace")

        _results, summaries, all_ok = run_benchmark(
            list(CHAPTER_PARSE_BENCHMARK_CASES),
            runs=1,
            verbose=False,
        )
        for summary in summaries:
            self.assertGreaterEqual(
                summary.passed,
                1,
                f"{summary.case_id}: {summary.failed} failed; "
                f"elapsed_avg={summary.elapsed_sec_avg}s",
            )
        self.assertTrue(all_ok, "chapter parse benchmark gates failed")


if __name__ == "__main__":
    unittest.main()
