"""Phase: L2 inference (denoise / flow / job)."""

from __future__ import annotations

from typing import Any

from backend.engine.inference import DiffusionInference, run_diffusion_denoise
from backend.engine.inference._runtime import inference_span
from backend.engine.sessions._context import MediaRunContext, ResolvedRun
from backend.engine.sessions._phases.schedule import ScheduleState


def infer_phase(
    resolved: ResolvedRun,
    conditioning: dict[str, Any],
    schedule: ScheduleState,
    *,
    runtime_ctx: Any,
    pipeline: Any | None = None,
    run_ctx: MediaRunContext | None = None,
    batch_seed: int | None = None,
    batch_idx: int = 0,
    batch_on_progress: Any | None = None,
    **denoise_kwargs: Any,
) -> Any | None:
    """Dispatch infer via ``MediaRunContext.session_infer`` or generic diffusion."""
    _ = conditioning, schedule, pipeline
    if run_ctx is not None:
        return run_ctx.session_infer(
            pipeline=pipeline,
            batch_seed=batch_seed,
            batch_idx=batch_idx,
            batch_on_progress=batch_on_progress,
        )

    plugin = resolved.plugin
    paradigm = plugin.select_paradigm() if plugin else "diffusion"
    if paradigm not in ("diffusion",):
        raise NotImplementedError(
            f"paradigm {paradigm!r} requires a media CreateRunContext on infer_phase "
            "(image/video/audio/upscale phased helpers)"
        )
    if plugin is not None and "model" not in denoise_kwargs:
        backbone = plugin.backbone
        model = getattr(backbone, "_model", None)
        if model is None and hasattr(backbone, "model"):
            try:
                model = backbone.model
            except RuntimeError:
                model = None
        if model is not None:
            denoise_kwargs = {**denoise_kwargs, "model": model}

    with inference_span(resolved.exec_ctx, "diffusion_paradigm"):
        return run_diffusion_denoise(runtime_ctx, **denoise_kwargs)


def infer_via_diffusion(bundle: Any, *, kernels: Any) -> Any | None:
    """Direct ``DiffusionInference`` entry for tests."""
    return DiffusionInference(kernels).run(bundle)
