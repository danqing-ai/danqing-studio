"""Bridge ImagePipeline / VideoPipeline progress callbacks to ExecutionContext."""
from __future__ import annotations

from backend.core.contracts import ExecutionContext, ProgressEvent


def emit_pipeline_progress(
    ctx: ExecutionContext,
    progress: float,
    step: int,
    total: int,
    message: str | None = None,
    phase: str | None = None,
) -> None:
    ctx.on_progress(
        ProgressEvent(
            progress=progress,
            step=step,
            total=total,
            message=message,
            phase=phase,
        )
    )


def make_pipeline_progress_callback(ctx: ExecutionContext):
    """Return ``on_progress`` for pipeline.run*(..., on_progress=...)."""

    def on_progress(p, s, t, msg=None, phase=None):
        emit_pipeline_progress(ctx, p, s, t, msg, phase)

    return on_progress
