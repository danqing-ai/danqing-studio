"""Video upscale — registry video SR runner."""

from __future__ import annotations

from typing import Any

from backend.engine.inference._runtime import inference_span
from backend.engine.inference.job import JobBundle, run_job
from backend.engine.video_upscale_registry import get_video_upscale_runner


def run_video_upscale_job(ctx: Any) -> tuple[str, dict[str, Any]]:
    pipeline = ctx.pipeline
    runner = get_video_upscale_runner(ctx.kind)
    bundle = JobBundle(
        run_fn=runner,
        kwargs={
            "ctx": pipeline.ctx,
            "request": ctx.request,
            "ctx_exec": ctx.exec_ctx,
            "entry": ctx.entry,
            "version_key": ctx.version_key,
            "model_key": ctx.model_key,
            "asset_store": pipeline._asset_store,
            "model_registry": pipeline._registry,
            "model_cache": pipeline._cache,
            "project_root": pipeline._project_root,
            "on_progress": ctx.on_progress,
            "on_log": ctx.on_log,
        },
    )
    with inference_span(ctx.exec_ctx, "job_paradigm"):
        result = run_job(bundle)
    if result is None:
        raise RuntimeError("Video upscale runner returned None")
    return result
