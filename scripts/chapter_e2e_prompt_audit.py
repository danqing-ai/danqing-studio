#!/usr/bin/env python3
"""Run chapter parse on benchmark fixtures and audit T2I/I2V prompts."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.core.contracts import ScriptParseDecomposeRequest, ScriptParseExpandRequest
from tests.chapter_parse_benchmark_cases import CASE_BY_ID, CHAPTER_PARSE_BENCHMARK_CASES
from tests.script_parse_integration import _load_llm_service

OUT_DIR = ROOT / "tests" / "benchmark" / "outputs" / "chapter_e2e"


def _response_to_dict(resp) -> dict:
    if hasattr(resp, "model_dump"):
        return resp.model_dump()
    return dict(resp)


def _check_shot_fields(shot: dict, index: int) -> list[str]:
    """Generic policy checks (no story-specific strings)."""
    issues: list[str] = []
    from backend.engine.llm.storyboard_cast import find_name_look_tags

    for fld in ("scene_prompt", "start_visual_prompt", "visual_prompt", "anchor_visual_prompt", "video_prompt"):
        text = str(shot.get(fld) or "")
        if find_name_look_tags(text):
            issues.append(f"shot {index}: {fld} has inline Name（…） look tag")
    role = shot.get("segment_role")
    if role == "face_anchor":
        av = str(shot.get("anchor_visual_prompt") or shot.get("start_visual_prompt") or "").strip()
        if not av:
            issues.append(f"shot {index}: face_anchor missing anchor/start visual")
        loc = str(shot.get("location") or "").strip()
        if loc and av and loc not in av and loc not in str(shot.get("scene_prompt") or ""):
            issues.append(f"shot {index}: face_anchor visual may omit location {loc!r}")
    vp = str(shot.get("video_prompt") or shot.get("motion_prompt") or "").strip()
    if not vp:
        issues.append(f"shot {index}: empty video_prompt")
    return issues


def run_case(case_id: str, *, skip_tsx: bool = False) -> int:
    case = CASE_BY_ID[case_id]
    svc = _load_llm_service()
    if not svc.is_available():
        print("SKIP: local LLM not available", file=sys.stderr)
        return 2

    print(f"\n{'=' * 72}\nCASE {case_id} — {case.title}\n{'=' * 72}")
    t0 = time.perf_counter()
    decomposed = svc.script_parse_decompose(
        ScriptParseDecomposeRequest(
            script_text=case.load_script(),
            title=case.title,
            locale=case.locale,
        )
    )
    resp = svc.script_parse_expand(
        ScriptParseExpandRequest(
            script_artifact=decomposed.script_artifact,
            locale=case.locale,
            target_duration_sec=case.target_duration_sec,
            segment_duration_sec=case.segment_duration_sec,
        )
    )
    elapsed = time.perf_counter() - t0
    payload = _response_to_dict(resp)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_json = OUT_DIR / f"{case_id}_analyze.json"
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    shots = payload.get("shots") or []
    beats = payload.get("scene_beats") or []
    quality = payload.get("quality_issues") or []
    print(f"elapsed={elapsed:.1f}s beats={len(beats)} shots={len(shots)} llm_calls={payload.get('llm_calls')}")
    print(f"saved: {out_json.relative_to(ROOT)}")

    field_issues: list[str] = []
    for i, shot in enumerate(shots):
        field_issues.extend(_check_shot_fields(shot, i))
    if field_issues:
        print("\nFIELD CHECK WARNINGS:")
        for msg in field_issues[:20]:
            print(f"  - {msg}")
        if len(field_issues) > 20:
            print(f"  ... and {len(field_issues) - 20} more")
    else:
        print("\nFIELD CHECK: ok (no inline look tags, video_prompt present)")

    if quality:
        codes = sorted({q.get("code") for q in quality if q.get("code")})
        print(f"quality_issues ({len(quality)}): {', '.join(codes)}")

    if skip_tsx:
        return 1 if field_issues else 0

    print("\n--- Final T2I / I2V prompts (frontend helpers) ---\n")
    proc = subprocess.run(
        ["npx", "--yes", "tsx", str(ROOT / "scripts" / "chapter_e2e_prompt_audit.ts"), str(out_json)],
        cwd=ROOT / "frontend",
        capture_output=True,
        text=True,
    )
    if proc.stdout:
        print(proc.stdout)
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        return proc.returncode
    return 1 if field_issues else 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=("wukong", "rainy_night", "all"), default="all")
    parser.add_argument("--skip-tsx", action="store_true")
    args = parser.parse_args()
    case_ids = [c.case_id for c in CHAPTER_PARSE_BENCHMARK_CASES] if args.case == "all" else [args.case]
    rc = 0
    for cid in case_ids:
        code = run_case(cid, skip_tsx=args.skip_tsx)
        rc = max(rc, code)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
