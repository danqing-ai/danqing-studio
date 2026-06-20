"""Long-video storyboard planning, parsing, and quality checks."""
from __future__ import annotations

import re
from dataclasses import asdict

from backend.engine.families.ltx.long_video_plan import LongVideoPlan, build_long_video_plan

EXPAND_BATCH_SIZE = 4

_ANCHOR_RE = re.compile(r"\[Anchor\]\s*(.+?)(?=\[Beat\]|\Z)", re.S | re.I)
_BEAT_RE = re.compile(r"\[Beat\s*(\d+)?\]\s*(.+?)(?=\[Beat|\[Segment|\Z)", re.S | re.I)
_OPENING_RE = re.compile(r"\[Opening\]\s*(.+?)(?=\[Segment|\Z)", re.S | re.I)
_SEGMENT_RE = re.compile(r"\[Segment\s*(\d+)?\]\s*(.+?)(?=\[Segment|\[Opening|\Z)", re.S | re.I)


def plan_to_dto(plan: LongVideoPlan) -> dict:
    d = asdict(plan)
    d["segment_durations_sec"] = list(plan.segment_durations_sec)
    return d


def parse_plan_script(text: str, *, expected_beats: int) -> tuple[str, list[str]]:
    raw = (text or "").strip()
    anchor_m = _ANCHOR_RE.search(raw)
    character_anchor = anchor_m.group(1).strip() if anchor_m else ""
    beats: list[str] = []
    for m in _BEAT_RE.finditer(raw):
        line = m.group(2).strip()
        if line:
            beats.append(line)
    if not beats:
        for line in raw.splitlines():
            s = line.strip()
            if s and not s.lower().startswith("[anchor"):
                beats.append(s.lstrip("-•* ").strip())
    if expected_beats > 0 and len(beats) < expected_beats:
        raise ValueError(f"plan parse: expected {expected_beats} beats, got {len(beats)}")
    return character_anchor, beats[:expected_beats] if expected_beats else beats


def parse_expand_script(text: str, *, expected_segments: int) -> tuple[str, list[str]]:
    raw = (text or "").strip()
    opening_m = _OPENING_RE.search(raw)
    opening = opening_m.group(1).strip() if opening_m else ""
    segments: list[str] = []
    for m in _SEGMENT_RE.finditer(raw):
        seg = m.group(2).strip()
        if seg:
            segments.append(seg)
    if not segments:
        blocks = [b.strip() for b in re.split(r"\n\s*\n", raw) if b.strip()]
        if opening and blocks and blocks[0].startswith("[Opening]"):
            blocks = blocks[1:]
        segments = blocks[1:] if opening and len(blocks) > 1 else blocks
    if expected_segments and len(segments) < expected_segments:
        raise ValueError(
            f"expand parse: expected {expected_segments} segment prompts, got {len(segments)}"
        )
    return opening, segments[:expected_segments] if expected_segments else segments


def merge_expand_batches(
    opening_parts: list[str],
    segment_batches: list[list[str]],
) -> tuple[str, list[str]]:
    opening = next((o.strip() for o in opening_parts if o and o.strip()), "")
    merged: list[str] = []
    for batch in segment_batches:
        merged.extend(batch)
    return opening, merged


def storyboard_quality_ok(
    *,
    character_anchor: str,
    opening_prompt: str,
    segment_prompts: list[str],
    beat_sheet: list[str],
    plan: LongVideoPlan,
) -> bool:
    if len(character_anchor.strip()) < 12:
        return False
    if len(opening_prompt.strip()) < 20:
        return False
    if len(segment_prompts) != plan.extend_pass_count:
        return False
    if len(beat_sheet) != plan.total_segments:
        return False
    seen = set()
    for p in segment_prompts:
        key = p.strip()[:40]
        if key in seen:
            return False
        seen.add(key)
        if len(p.strip()) < 16:
            return False
    return True


def expand_batches_for_plan(plan: LongVideoPlan) -> list[tuple[int, int]]:
    """Return (start_idx, count) batches for extend segments (exclude pass0)."""
    n = plan.extend_pass_count
    if n <= 0:
        return []
    batches: list[tuple[int, int]] = []
    i = 0
    while i < n:
        take = min(EXPAND_BATCH_SIZE, n - i)
        batches.append((i, take))
        i += take
    return batches
