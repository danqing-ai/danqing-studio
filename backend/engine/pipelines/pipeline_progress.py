"""Shared progress + graph-step helpers for image/video pipelines."""
from __future__ import annotations

from typing import Callable

# Denoise uses most of the bar; VAE decode + save use the tail (queue ETA uses 1 - progress).
DENOISE_PROGRESS_SHARE = 0.88
POST_PROGRESS_SHARE = 0.12


def pipeline_graph_step(
    node_id: str,
    on_log: Callable[[str, str], None] | None,
    *,
    message: str = "start",
) -> None:
    if on_log is not None:
        on_log("info", f"[{node_id}] {message}")


def emit_denoise_progress(
    on_progress: Callable[..., None] | None,
    step_1based: int,
    n_steps: int,
) -> None:
    if on_progress is None:
        return
    n = max(1, int(n_steps))
    s = min(max(1, int(step_1based)), n)
    on_progress(DENOISE_PROGRESS_SHARE * (s / n), s, n, "denoise", "denoising")


def emit_post_progress(
    on_progress: Callable[..., None] | None,
    *,
    n_steps: int,
    within_post: float,
) -> None:
    if on_progress is None:
        return
    n = max(1, int(n_steps))
    w = min(1.0, max(0.0, float(within_post)))
    on_progress(DENOISE_PROGRESS_SHARE + POST_PROGRESS_SHARE * w, n, n, "post", "decoding")


def emit_complete(on_progress: Callable[..., None] | None, n_steps: int) -> None:
    if on_progress is None:
        return
    n = max(1, int(n_steps))
    on_progress(1.0, n, n, None, "saving")


def emit_phase(
    on_progress: Callable[..., None] | None,
    *,
    phase: str,
    progress: float,
    n_steps: int,
) -> None:
    if on_progress is None:
        return
    n = max(1, int(n_steps))
    on_progress(min(1.0, max(0.0, float(progress))), 0, n, phase, phase)
