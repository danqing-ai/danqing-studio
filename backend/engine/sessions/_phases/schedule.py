"""Phase: scheduler + timesteps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from backend.engine.pipelines.image_create_phases import ImageCreateRunContext
from backend.engine.sessions._context import ResolvedRun


@dataclass
class ScheduleState:
    scheduler: Any
    timesteps: Any
    sigmas: Any | None
    scheduler_name: str


def schedule_phase(
    resolved: ResolvedRun,
    *,
    steps: int | None = None,
    width: int | None = None,
    height: int | None = None,
    ctx: ImageCreateRunContext | None = None,
) -> ScheduleState:
    """Build scheduler timesteps from prepared create context or fail loud."""
    _ = resolved, steps, width, height
    if ctx is None:
        raise RuntimeError("schedule_phase requires ImageCreateRunContext (Phase 1b)")
    name = getattr(ctx.semantics, "scheduler_name", "unknown")
    return ScheduleState(
        scheduler=ctx.scheduler,
        timesteps=ctx.timesteps,
        sigmas=ctx.sigmas,
        scheduler_name=str(name),
    )
