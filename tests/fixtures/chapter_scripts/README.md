# Chapter parse benchmark fixtures

Fixed scripts for **long-video chapter analyze** speed + quality regression (`make chapter-parse-bench`).

| Case id | File | Title | ~chars | min beats | min shots |
|---------|------|-------|--------|-----------|-----------|
| `wukong` | `wukong_battle.txt` | 大战悟空 | 1066 | 4 | 10 |
| `rainy_night` | `rainy_night_visitor.txt` | 雨夜访客 | 344 | 4 | 10 |

## Run

```bash
# both cases, 1 run each, JSON report
make chapter-parse-bench

# single case or multiple runs
make chapter-parse-bench CASE=rainy_night RUNS=3

# unittest (skips when LLM unavailable)
PYTHONPATH=. .venv/bin/python tests/chapter_parse_benchmark_test.py
```

Report: `tests/benchmark/outputs/chapter_parse_bench.json`

## Gates (per run)

- pipeline completes without exception
- `beats` / `shots` ≥ case minimums
- no **forbidden** `quality_issues` codes (pipeline regressions):
  `motion_duplicate_in_group`, `motion_role_undifferentiated`, `beat_no_shots`, `instruction_leak`
- other quality codes (e.g. `roster_shot_unknown_character`, `beat_narrative_undercovered`) are **reported only**
