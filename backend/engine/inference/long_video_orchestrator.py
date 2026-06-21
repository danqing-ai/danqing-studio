"""Long-video orchestrator — strategy dispatch without duplicating VideoPipeline."""
from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Callable

from backend.core.contracts import VideoLongGenerationRequest
from backend.engine.inference.strategies.latent_extend import run_latent_extend_strategy
from backend.engine.inference.strategies.segmented_i2v import run_segmented_i2v_strategy


def run_long_video_orchestrator(
    *,
    request: VideoLongGenerationRequest,
    exec_ctx: Any,
    image_dispatch: dict[str, Any],
    video_dispatch: dict[str, Any],
    on_progress: Callable | None = None,
    on_log: Callable | None = None,
) -> tuple[str, dict[str, Any]] | None:
    """Run long-video generation; returns (output_path, metadata) or None if cancelled."""
    if exec_ctx.cancel_token.is_cancelled():
        return None

    spec = request.long_video
    trace = getattr(exec_ctx, "trace", None)

    def span_factory(name: str):
        if trace is None:
            return nullcontext()
        return trace.span_ctx(name, kind="phase")

    with span_factory("resolve"):
        strategy = spec.strategy or "segmented_i2v"

    if strategy == "segmented_i2v":
        return run_segmented_i2v_strategy(
            request=request,
            ctx_exec=exec_ctx,
            on_progress=on_progress,
            on_log=on_log,
            span_factory=span_factory,
        )

    if strategy == "latent_extend":
        return run_latent_extend_strategy(
            request=request,
            ctx_exec=ctx_exec,
            video_dispatch=video_dispatch,
            on_progress=on_progress,
            on_log=on_log,
        )

    raise RuntimeError(f"unknown long_video strategy {strategy!r}")
