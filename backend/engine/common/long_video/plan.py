"""Long-video timing plans — segmented shot plan + LTX extend plan."""
from __future__ import annotations

import math
from dataclasses import dataclass

# LTX extend plan lives in family module; re-export for orchestrator + storyboard.
from backend.engine.families.ltx.long_video_plan import (  # noqa: F401
    LongVideoPlan,
    build_long_video_plan,
    compute_extend_pass_count,
    narrative_budget_for_target,
)


def duration_sec_from_num_frames(num_frames: int, fps: float) -> float:
    rate = max(1.0, float(fps))
    nf = max(1, int(num_frames))
    return max(0.0, (nf - 1) / rate)


def num_frames_for_duration_sec(duration_sec: float, fps: float) -> int:
    rate = max(1.0, float(fps))
    sec = max(0.0, float(duration_sec))
    return max(1, int(round(sec * rate)) + 1)


@dataclass(frozen=True)
class ShotPlan:
    target_duration_sec: float
    segment_duration_sec: float
    shot_count: int
    segment_durations_sec: tuple[float, ...]
    narrative_budget: str


def build_shot_plan(
    *,
    target_duration_sec: float = 60.0,
    segment_duration_sec: float = 5.0,
) -> ShotPlan:
    target = max(0.0, float(target_duration_sec))
    seg = max(0.5, float(segment_duration_sec))
    count = max(1, int(math.ceil(target / seg)))
    durations = tuple([seg] * count)
    return ShotPlan(
        target_duration_sec=target,
        segment_duration_sec=seg,
        shot_count=count,
        segment_durations_sec=durations,
        narrative_budget=narrative_budget_for_target(target),
    )


def build_shot_plan_for_scenes(
    *,
    scene_count: int,
    segment_duration_sec: float = 5.0,
) -> ShotPlan:
    """Shot count from chapter scene analysis; duration = scenes × segment length."""
    seg = max(0.5, float(segment_duration_sec))
    count = max(2, min(24, int(scene_count)))
    target = count * seg
    durations = tuple([seg] * count)
    return ShotPlan(
        target_duration_sec=target,
        segment_duration_sec=seg,
        shot_count=count,
        segment_durations_sec=durations,
        narrative_budget=narrative_budget_for_target(target),
    )
