"""Image upscale — registry job runner."""

from __future__ import annotations

from typing import Any

from backend.engine.inference._runtime import inference_span
from backend.engine.inference.job import JobBundle, run_job
from backend.engine.upscale_job_registry import get_upscale_job_runner


def run_upscale_job(ctx: Any) -> dict[str, Any]:
    pipeline = ctx.pipeline
    run_upscale_job_fn = get_upscale_job_runner(ctx.family)

    def _log(level: str, msg: str) -> None:
        if ctx.on_log:
            ctx.on_log(level, msg)

    bundle = JobBundle(
        run_fn=run_upscale_job_fn,
        kwargs={
            "bundle_path": ctx.bundle_root,
            "model_key": ctx.model_key,
            "source_image": ctx.src_path,
            "scale": ctx.scale,
            "softness": float(ctx.request.denoise),
            "seed": ctx.seed,
            "output_png": ctx.out_path,
            "on_log": _log,
            "pipeline": ctx.upscale_pipeline,
            "tile_size": int(getattr(ctx.request, "tile_size", 0) or 0),
        },
    )
    with inference_span(ctx.exec_ctx, "job_paradigm"):
        return run_job(bundle)
