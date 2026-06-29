"""Fixed chapter-parse benchmark runner (local LLM, timing + quality gates)."""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.chapter_parse_benchmark_cases import (
    CHAPTER_PARSE_BENCHMARK_CASES,
    ChapterParseBenchmarkCase,
    resolve_cases,
)

FORBIDDEN_QUALITY_CODES = frozenset(
    {
        # Pipeline regressions only — LLM roster drift is reported, not gated.
        "motion_duplicate_in_group",
        "motion_role_undifferentiated",
        "beat_no_shots",
        "instruction_leak",
    }
)


@dataclass
class CaseRunResult:
    case_id: str
    title: str
    run_index: int
    ok: bool
    elapsed_sec: float
    beats: int = 0
    shots: int = 0
    total_duration_sec: float = 0.0
    llm_calls: int = 0
    quality_issue_count: int = 0
    quality_codes: list[str] = field(default_factory=list)
    error: str = ""


@dataclass
class CaseSummary:
    case_id: str
    title: str
    runs: int
    passed: int
    failed: int
    elapsed_sec_min: float
    elapsed_sec_max: float
    elapsed_sec_avg: float
    shots_min: int
    shots_max: int


def _load_llm_service():
    from tests.long_video_chapter_analyze_integration import _load_llm_service

    return _load_llm_service()


def run_single(
    svc,
    case: ChapterParseBenchmarkCase,
    *,
    run_index: int,
) -> CaseRunResult:
    from backend.core.contracts import LongVideoChapterAnalyzeRequest

    t0 = time.perf_counter()
    try:
        resp = svc.analyze_long_video_chapter(
            LongVideoChapterAnalyzeRequest(
                chapter_text=case.load_script(),
                chapter_title=case.title,
                locale=case.locale,
                target_duration_sec=case.target_duration_sec,
                segment_duration_sec=case.segment_duration_sec,
            )
        )
    except Exception as exc:
        return CaseRunResult(
            case_id=case.case_id,
            title=case.title,
            run_index=run_index,
            ok=False,
            elapsed_sec=time.perf_counter() - t0,
            error=str(exc),
        )

    elapsed = time.perf_counter() - t0
    beats = len(resp.scene_beats)
    shots = len(resp.shots)
    total_dur = sum(float(getattr(s, "duration_sec", 0) or 0) for s in resp.shots)
    codes = [i.code for i in resp.quality_issues]
    forbidden = [c for c in codes if c in FORBIDDEN_QUALITY_CODES]

    ok = True
    reasons: list[str] = []
    if beats < case.min_beats:
        ok = False
        reasons.append(f"beats {beats} < min {case.min_beats}")
    if shots < case.min_shots:
        ok = False
        reasons.append(f"shots {shots} < min {case.min_shots}")
    if forbidden:
        ok = False
        reasons.append(f"forbidden quality: {forbidden}")

    return CaseRunResult(
        case_id=case.case_id,
        title=case.title,
        run_index=run_index,
        ok=ok,
        elapsed_sec=round(elapsed, 2),
        beats=beats,
        shots=shots,
        total_duration_sec=round(total_dur, 1),
        llm_calls=resp.llm_calls,
        quality_issue_count=len(resp.quality_issues),
        quality_codes=codes,
        error="; ".join(reasons),
    )


def summarize_case(results: list[CaseRunResult]) -> CaseSummary:
    elapsed = [r.elapsed_sec for r in results]
    shots = [r.shots for r in results if r.shots]
    return CaseSummary(
        case_id=results[0].case_id,
        title=results[0].title,
        runs=len(results),
        passed=sum(1 for r in results if r.ok),
        failed=sum(1 for r in results if not r.ok),
        elapsed_sec_min=min(elapsed) if elapsed else 0.0,
        elapsed_sec_max=max(elapsed) if elapsed else 0.0,
        elapsed_sec_avg=round(sum(elapsed) / len(elapsed), 2) if elapsed else 0.0,
        shots_min=min(shots) if shots else 0,
        shots_max=max(shots) if shots else 0,
    )


def run_benchmark(
    cases: list[ChapterParseBenchmarkCase],
    *,
    runs: int,
    verbose: bool = True,
) -> tuple[list[CaseRunResult], list[CaseSummary], bool]:
    svc = _load_llm_service()
    if not svc.is_available():
        raise RuntimeError("local LLM not available (install qwen3.5-4b in models/ + registry)")

    model_id = svc.get_model_info().get("model_id", "")
    if verbose:
        print(f"chapter_parse_benchmark model={model_id} cases={[c.case_id for c in cases]} runs={runs}")

    all_results: list[CaseRunResult] = []
    summaries: list[CaseSummary] = []

    for case in cases:
        if verbose:
            print(f"\n{'=' * 60}")
            print(
                f"CASE {case.case_id} title={case.title!r} "
                f"script_len={len(case.load_script())} "
                f"target={case.target_duration_sec}s"
            )
        case_results: list[CaseRunResult] = []
        for run_i in range(1, runs + 1):
            if verbose:
                print(f"--- run {run_i}/{runs} ---")
            row = run_single(svc, case, run_index=run_i)
            case_results.append(row)
            all_results.append(row)
            if verbose:
                status = "PASS" if row.ok else "FAIL"
                print(
                    f"{status} elapsed={row.elapsed_sec}s beats={row.beats} shots={row.shots} "
                    f"total_dur={row.total_duration_sec}s quality={row.quality_issue_count} "
                    f"llm_calls={row.llm_calls}"
                )
                if row.error:
                    print(f"  gate: {row.error}")
                if row.quality_codes and run_i == 1:
                    for code in sorted(set(row.quality_codes)):
                        print(f"  quality.{code}")

        summaries.append(summarize_case(case_results))
        if verbose:
            s = summaries[-1]
            print(
                f"summary: pass={s.passed}/{s.runs} "
                f"elapsed_avg={s.elapsed_sec_avg}s "
                f"elapsed=[{s.elapsed_sec_min},{s.elapsed_sec_max}] "
                f"shots=[{s.shots_min},{s.shots_max}]"
            )

    all_ok = all(r.ok for r in all_results)
    return all_results, summaries, all_ok


def write_report(
    path: Path,
    *,
    model_id: str,
    results: list[CaseRunResult],
    summaries: list[CaseSummary],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_id": model_id,
        "cases": [c.case_id for c in CHAPTER_PARSE_BENCHMARK_CASES],
        "results": [asdict(r) for r in results],
        "summaries": [asdict(s) for s in summaries],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fixed chapter-parse speed/quality benchmark.")
    parser.add_argument("--case", choices=("wukong", "rainy_night", "all"), default="all")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument(
        "--out",
        default=str(ROOT / "tests" / "benchmark" / "outputs" / "chapter_parse_bench.json"),
        help="JSON report path",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    cases = resolve_cases(args.case)
    runs = max(1, int(args.runs))

    try:
        results, summaries, all_ok = run_benchmark(cases, runs=runs, verbose=not args.quiet)
    except RuntimeError as exc:
        print(f"SKIP: {exc}")
        return 2

    if not args.quiet:
        print(f"\n{'=' * 60}")
        print(f"OVERALL: {'PASS' if all_ok else 'FAIL'}")

    try:
        svc = _load_llm_service()
        model_id = str(svc.get_model_info().get("model_id", ""))
    except Exception:
        model_id = ""
    write_report(Path(args.out), model_id=model_id, results=results, summaries=summaries)
    if not args.quiet:
        print(f"report: {args.out}")

    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
