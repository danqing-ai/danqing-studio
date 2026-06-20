"""Long-video timing plan — shared by extend engine and LLM storyboard."""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class LongVideoPlan:
    target_duration_sec: float
    initial_duration_sec: float
    segment_extend_sec: float
    reference_duration_sec: float
    extend_pass_count: int
    total_segments: int
    segment_durations_sec: tuple[float, ...]
    narrative_budget: str


def narrative_budget_for_target(target_duration_sec: float) -> str:
    t = float(target_duration_sec)
    if t <= 35.0:
        return "compact"
    if t <= 75.0:
        return "standard"
    return "epic"


def compute_extend_pass_count(
    target_duration_sec: float,
    initial_duration_sec: float,
    segment_extend_sec: float,
) -> int:
    remaining = max(0.0, float(target_duration_sec) - float(initial_duration_sec))
    seg = max(0.001, float(segment_extend_sec))
    if remaining <= 0.0:
        return 0
    return int(math.ceil(remaining / seg))


def build_long_video_plan(
    *,
    target_duration_sec: float = 60.0,
    initial_duration_sec: float = 8.0,
    segment_extend_sec: float = 8.0,
    reference_duration_sec: float = 3.0,
) -> LongVideoPlan:
    target = max(0.0, float(target_duration_sec))
    initial = max(0.1, float(initial_duration_sec))
    extend = max(0.1, float(segment_extend_sec))
    reference = max(0.0, float(reference_duration_sec))
    extend_pass_count = compute_extend_pass_count(target, initial, extend)
    durations: list[float] = [initial]
    durations.extend([extend] * extend_pass_count)
    return LongVideoPlan(
        target_duration_sec=target,
        initial_duration_sec=initial,
        segment_extend_sec=extend,
        reference_duration_sec=reference,
        extend_pass_count=extend_pass_count,
        total_segments=1 + extend_pass_count,
        segment_durations_sec=tuple(durations),
        narrative_budget=narrative_budget_for_target(target),
    )


def duration_sec_from_num_frames(num_frames: int, fps: float) -> float:
    rate = max(1.0, float(fps))
    nf = max(1, int(num_frames))
    return max(0.0, (nf - 1) / rate)


def num_frames_for_duration_sec(duration_sec: float, fps: float) -> int:
    rate = max(1.0, float(fps))
    sec = max(0.0, float(duration_sec))
    return max(1, int(round(sec * rate)) + 1)
