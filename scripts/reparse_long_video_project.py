#!/usr/bin/env python3
"""Re-parse a long-video project and dump shot prompt fields."""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.core.contracts import ScriptParseDecomposeRequest, ScriptParseExpandRequest
from tests.script_parse_integration import _load_llm_service

SHOT_KEYS = (
    "id",
    "order",
    "segment_role",
    "segment_group_id",
    "location",
    "narrative_beat_index",
    "scene_prompt",
    "first_frame_requirement",
    "start_visual_prompt",
    "anchor_visual_prompt",
    "visual_prompt",
    "video_prompt",
    "motion_prompt",
    "first_frame_visibility",
    "end_visibility",
    "camera_zone_id",
    "characters_on_screen",
    "cast_looks",
    "scene_look",
    "first_frame_strategy",
)


def _shot_to_dict(shot: Any) -> dict[str, Any]:
    if hasattr(shot, "model_dump"):
        return shot.model_dump()
    return dict(shot)


def _merge_characters_preserve_assets(existing: list[dict], incoming: list[dict]) -> list[dict]:
    by_name = {str(c.get("name") or ""): c for c in existing}
    merged: list[dict] = []
    for inc in incoming:
        ex = by_name.get(str(inc.get("name") or ""))
        if not ex:
            merged.append(inc)
            continue
        ex_looks = {str(l.get("id") or ""): l for l in ex.get("looks") or []}
        looks = []
        for lk in inc.get("looks") or []:
            old = ex_looks.get(str(lk.get("id") or ""))
            row = dict(lk)
            if old and old.get("reference_asset_id"):
                row["reference_asset_id"] = old["reference_asset_id"]
            looks.append(row)
        merged.append(
            {
                **inc,
                "looks": looks or inc.get("looks") or [],
                "default_look_id": ex.get("default_look_id") or inc.get("default_look_id") or "",
            }
        )
    return merged or incoming


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--shot-index", type=int, default=1, help="0-based shot index (#2 => 1)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    row = conn.execute(
        "SELECT title, state_json FROM long_video_projects WHERE id = ?",
        (args.project_id,),
    ).fetchone()
    if not row:
        print(f"project not found: {args.project_id}", file=sys.stderr)
        return 1

    title, raw = row
    state: dict[str, Any] = json.loads(raw)
    script = (state.get("script_text") or state.get("chapter_text") or "").strip()
    if not script:
        print("project has no script_text", file=sys.stderr)
        return 1

    svc = _load_llm_service()
    if not svc.is_available():
        print("local LLM not available", file=sys.stderr)
        return 1

    print(f"Parsing project {args.project_id} ({title}) …", flush=True)
    decomposed = svc.script_parse_decompose(
        ScriptParseDecomposeRequest(
            script_text=script,
            title=(state.get("chapter_title") or title or "").strip(),
            locale="zh",
        )
    )
    resp = svc.script_parse_expand(
        ScriptParseExpandRequest(
            script_artifact=decomposed.script_artifact,
            locale="zh",
            target_duration_sec=float(state.get("target_duration_sec") or 60),
            segment_duration_sec=float(state.get("segment_duration_sec") or 5),
            max_clip_sec=10.0,
        )
    )
    print(
        f"Parse done: beats={len(resp.scene_beats)} shots={len(resp.shots)} "
        f"llm_calls={resp.llm_calls} quality_warnings={len(resp.quality_warnings)}",
        flush=True,
    )

    if args.dry_run:
        shots = [_shot_to_dict(s) for s in resp.shots]
    else:
        # Preserve user assets: merge rosters while keeping portrait reference_asset_id.
        existing_chars = state.get("characters") or []
        incoming_chars = [c.model_dump() for c in resp.characters]
        merged_chars = _merge_characters_preserve_assets(existing_chars, incoming_chars)
        prev_shots = state.get("shots") or []

        scene_beats = [
            {"order": s.order, "title": s.title or "", "beat": s.beat or ""}
            for s in resp.scene_beats
        ]
        api_shots = [_shot_to_dict(s) for s in resp.shots]
        state["chapter_analysis"] = {
            "synopsis": resp.synopsis or "",
            "mood": resp.mood or "",
            "scene_beats": scene_beats,
            "character_anchor": resp.character_anchor or state.get("character_anchor", ""),
            "style_anchor": resp.style_anchor or state.get("style_anchor", ""),
            "characters": merged_chars,
            "scenes": [s.model_dump() for s in resp.scenes],
            "quality_warnings": resp.quality_warnings or [],
            "quality_issues": [q.model_dump() for q in (resp.quality_issues or [])],
        }
        state["character_anchor"] = state["chapter_analysis"]["character_anchor"]
        state["style_anchor"] = state["chapter_analysis"]["style_anchor"]
        state["characters"] = merged_chars
        state["scenes"] = state["chapter_analysis"]["scenes"]
        state["chapter_title"] = resp.chapter_title or state.get("chapter_title") or title

        # Map API shots → frontend shot state (mirror storyboardShotsFromResponse).
        fallback_durations = [float(s.get("duration_sec") or 5) for s in api_shots]
        shots = []
        for i, s in enumerate(api_shots):
            video_prompt = (s.get("video_prompt") or s.get("motion_prompt") or "").strip()
            start_visual = (s.get("start_visual_prompt") or s.get("visual_prompt") or "").strip()
            shots.append(
                {
                    "id": s.get("id") or f"shot_{i:02d}",
                    "order": i,
                    "visual_prompt": start_visual,
                    "motion_prompt": video_prompt,
                    "video_prompt": video_prompt,
                    "start_visual_prompt": start_visual or None,
                    "end_visual_prompt": (s.get("end_visual_prompt") or "").strip() or None,
                    "anchor_visual_prompt": (s.get("anchor_visual_prompt") or "").strip() or None,
                    "segment_role": s.get("segment_role") or "keyframe",
                    "start_frame_mode": s.get("start_frame_mode") or "keyframe",
                    "segment_group_id": s.get("segment_group_id"),
                    "segment_group_index": s.get("segment_group_index"),
                    "face_anchor_shot_id": s.get("face_anchor_shot_id"),
                    "flf_mode": s.get("flf_mode") or "none",
                    "chain_mode": s.get("chain_mode") or "keyframe_only",
                    "scene_prompt": s.get("scene_prompt") or "",
                    "cast_looks": s.get("cast_looks") or [],
                    "scene_look": s.get("scene_look"),
                    "first_frame_visibility": s.get("first_frame_visibility"),
                    "end_visibility": s.get("end_visibility"),
                    "characters_on_screen": s.get("characters_on_screen") or [],
                    "clip_start_state": s.get("clip_start_state"),
                    "clip_end_state": s.get("clip_end_state"),
                    "first_frame_requirement": s.get("first_frame_requirement"),
                    "camera_zone_id": s.get("camera_zone_id"),
                    "first_frame_strategy": s.get("first_frame_strategy"),
                    "duration_sec": s.get("duration_sec") or fallback_durations[i],
                    "status": "draft",
                }
            )
        prev_by_id = {str(s.get("id") or ""): s for s in prev_shots if s.get("id")}
        for row in shots:
            prev = prev_by_id.get(str(row.get("id") or ""))
            if not prev:
                continue
            if prev.get("scene_look"):
                row["scene_look"] = prev["scene_look"]
            if prev.get("keyframe_asset_id"):
                row["keyframe_asset_id"] = prev["keyframe_asset_id"]
                row["status"] = prev.get("status") or row["status"]
            if prev.get("segment_asset_id"):
                row["segment_asset_id"] = prev["segment_asset_id"]
        state["shots"] = shots
        from datetime import datetime

        now = datetime.now().isoformat()
        conn.execute(
            "UPDATE long_video_projects SET state_json = ?, updated_at = ? WHERE id = ?",
            (json.dumps(state, ensure_ascii=False), now, args.project_id),
        )
        conn.commit()
        print(f"Project updated at {now}", flush=True)

    idx = args.shot_index
    if idx < 0 or idx >= len(shots):
        print(f"shot index {idx} out of range (count={len(shots)})", file=sys.stderr)
        return 1

    shot = shots[idx]
    print(f"\n=== shot #{idx + 1} (index {idx}) ===")
    for k in SHOT_KEYS:
        if k in shot or k == "location":
            print(f"{k}: {json.dumps(shot.get(k), ensure_ascii=False)}")

    out_path = ROOT / "tests" / "benchmark" / "outputs" / f"reparse_{args.project_id}_shot{idx + 1}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(shot, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
