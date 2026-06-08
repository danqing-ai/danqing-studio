"""Phase: VAE / frame decode."""

from __future__ import annotations

from typing import Any

from backend.engine.pipelines.image_create_phases import ImageCreateRunContext, decode_create_latents
from backend.engine.pipelines.image_edit_phases import ImageEditRunContext, decode_image_edit_latents
from backend.engine.pipelines.pipeline_progress import emit_post_progress
from backend.engine.sessions._context import ResolvedRun
from backend.engine.sessions._phases.trace import phase_trace_span


def decode_phase(
    resolved: ResolvedRun,
    latents: Any,
    *,
    pipeline: Any | None = None,
    create_ctx: ImageCreateRunContext | None = None,
) -> Any:
    """Decode latents to pixels via legacy pipeline VAE path."""
    if create_ctx is None or pipeline is None:
        raise RuntimeError("decode_phase requires pipeline + ImageCreateRunContext (Phase 1b)")
    with phase_trace_span(resolved, "decode"):
        image = decode_create_latents(create_ctx, latents)
        emit_post_progress(
            create_ctx.on_progress,
            n_steps=len(create_ctx.timesteps),
            within_post=0.5,
        )
        return image


def decode_image_edit_phase(
    resolved: ResolvedRun,
    latents: Any,
    *,
    edit_ctx: ImageEditRunContext | None = None,
) -> Any:
    if edit_ctx is None:
        raise RuntimeError("decode_image_edit_phase requires ImageEditRunContext")
    with phase_trace_span(resolved, "decode"):
        return decode_image_edit_latents(edit_ctx, latents)
