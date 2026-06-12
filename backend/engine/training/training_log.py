"""Task log + progress helpers for LoRA training."""

from __future__ import annotations

from backend.core.contracts import ExecutionContext, LogEvent, ProgressEvent


def training_log(ctx: ExecutionContext, level: str, message: str) -> None:
    ctx.on_log(LogEvent(level=level, message=message))  # type: ignore[arg-type]


def training_progress(
    ctx: ExecutionContext,
    *,
    step: int,
    total: int,
    message: str = "",
    loss: float | None = None,
    phase: str = "training",
    progress: float | None = None,
) -> None:
    meta = f" loss={loss:.4f}" if loss is not None else ""
    frac = progress if progress is not None else min(1.0, step / max(total, 1))
    ctx.on_progress(
        ProgressEvent(
            progress=frac,
            step=step if phase == "training" else None,
            total=total if phase == "training" else None,
            message=(message or f"Training iteration {step}/{total}") + meta,
            phase=phase,
        )
    )
