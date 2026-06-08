"""Video upscale phased helpers (``VideoUpscaleSession``)."""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import Any, Callable

from backend.core.contracts import ExecutionContext, VideoUpscaleRequest
from backend.engine.inference.video_upscale_job import run_video_upscale_job
from backend.engine.pipelines.pipeline_progress import pipeline_graph_step, validate_bundle_graph_step
from backend.engine.protocols.plugin import FamilyPlugin
from backend.engine.sessions._context import MediaRunContext, ResolvedRun, require_resolved_bundle
from backend.engine.video_upscale_registry import resolve_video_upscale_kind

PhaseCmFactory = Callable[[str], AbstractContextManager[Any]]


@dataclass
class VideoUpscaleCreateRunContext(MediaRunContext):
    """State for one video upscale job (load → infer → persist)."""

    pipeline: Any
    request: VideoUpscaleRequest
    exec_ctx: ExecutionContext
    entry: Any
    family: str
    model_key: str
    version_key: str | None
    kind: str
    on_progress: Callable | None = None
    on_log: Callable | None = None

    def session_infer(self, **_ignored: Any) -> tuple[str, dict[str, Any]]:
        return execute_video_upscale_job(self)


def build_video_upscale_create_context(
    pipeline: Any,
    request: VideoUpscaleRequest,
    ctx_exec: ExecutionContext,
    *,
    resolved: ResolvedRun,
    on_progress: Callable | None = None,
    on_log: Callable | None = None,
    phase_cm: PhaseCmFactory | None = None,
    plugin: FamilyPlugin | None = None,
) -> VideoUpscaleCreateRunContext | None:
    _ = plugin
    phase_cm = phase_cm or (lambda _name: nullcontext())

    if ctx_exec.cancel_token.is_cancelled():
        return None

    model_key = resolved.model_id
    version_key = resolved.version_key
    entry = resolved.registry_entry
    if getattr(entry, "media", None) != "video":
        raise RuntimeError(
            f"Video upscale model {model_key!r} is not a video model "
            f"(media={getattr(entry, 'media', None)!r})."
        )

    family = resolved.family_id
    kind = resolve_video_upscale_kind(entry, version_key)
    bundle_root = require_resolved_bundle(resolved)
    validate_bundle_graph_step(bundle_root, family=family, model_id=model_key, on_log=on_log)

    with phase_cm("load"):
        pipeline_graph_step("load_transformer", on_log, message="video_upscale")

    return VideoUpscaleCreateRunContext(
        pipeline=pipeline,
        request=request,
        exec_ctx=ctx_exec,
        entry=entry,
        family=family,
        model_key=model_key,
        version_key=version_key,
        kind=kind,
        on_progress=on_progress,
        on_log=on_log,
    )


def execute_video_upscale_job(ctx: VideoUpscaleCreateRunContext) -> tuple[str, dict[str, Any]]:
    """Run video SR via ``JobParadigm``."""
    pipeline_graph_step("denoise", ctx.on_log, message="video_upscale")
    return run_video_upscale_job(ctx)


def persist_video_upscale_create(
    ctx: VideoUpscaleCreateRunContext,
    result: tuple[str, dict[str, Any]],
) -> tuple[str, dict[str, Any]] | None:
    if ctx.exec_ctx.cancel_token.is_cancelled():
        return None
    pipeline_graph_step("save_asset", ctx.on_log)
    if ctx.on_progress:
        ctx.on_progress({"phase": "complete", "progress": 1.0})
    return result


