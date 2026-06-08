"""Image upscale phased helpers (``UpscaleSession``)."""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from backend.core.contracts import ExecutionContext, ImageUpscaleRequest
from backend.engine.families._upscale_backbone import plugin_upscale_pipeline_if_ready
from backend.engine.inference.upscale_job import run_upscale_job
from backend.engine.pipelines.pipeline_progress import pipeline_graph_step, validate_bundle_graph_step
from backend.engine.pipelines.upscale_model_load import load_upscale_pipeline
from backend.engine.protocols.plugin import FamilyPlugin
from backend.engine.sessions._context import MediaRunContext, ResolvedRun, require_resolved_bundle

PhaseCmFactory = Callable[[str], AbstractContextManager[Any]]


@dataclass
class UpscaleCreateRunContext(MediaRunContext):
    """State for one upscale job (load → infer → persist)."""

    pipeline: Any
    request: ImageUpscaleRequest
    exec_ctx: ExecutionContext
    entry: Any
    family: str
    model_key: str
    version_key: str | None
    bundle_root: Path
    src_path: Path
    scale: int
    seed: int | None
    out_path: Path
    upscale_pipeline: Any
    on_progress: Callable | None = None
    on_log: Callable | None = None

    def session_infer(self, **_ignored: Any) -> dict[str, Any]:
        return execute_upscale_job(self)


def build_upscale_create_context(
    pipeline: Any,
    request: ImageUpscaleRequest,
    ctx_exec: ExecutionContext,
    *,
    resolved: ResolvedRun,
    on_progress: Callable | None = None,
    on_log: Callable | None = None,
    phase_cm: PhaseCmFactory | None = None,
    plugin: FamilyPlugin | None = None,
) -> UpscaleCreateRunContext | None:
    phase_cm = phase_cm or (lambda _name: nullcontext())

    model_key = resolved.model_id
    version_key = resolved.version_key
    entry = resolved.registry_entry
    family = resolved.family_id
    if ctx_exec.cancel_token.is_cancelled():
        return None

    bundle_root = require_resolved_bundle(resolved)
    validate_bundle_graph_step(bundle_root, family=family, model_id=model_key, on_log=on_log)

    src_path = ctx_exec.asset_store.get_file_path(request.source_asset_id)
    if not src_path.is_file():
        raise RuntimeError(f"Source asset file missing: {src_path}")

    scale = int(request.scale)
    seed = (request.metadata or {}).get("seed")
    if seed is not None:
        seed = int(seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    work = Path(ctx_exec.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    out_path = work / f"{model_key}_up_{timestamp}.png"

    upscale_pipeline = plugin_upscale_pipeline_if_ready(plugin)

    def _log(level: str, msg: str) -> None:
        if on_log:
            on_log(level, msg)

    with phase_cm("load"):
        if upscale_pipeline is None:
            if pipeline._cache is None or pipeline._cache.get(
                f"upscale:image:{entry.id}:{version_key or 'default'}"
            ) is None:
                pipeline_graph_step("load_transformer", on_log)
            upscale_pipeline = load_upscale_pipeline(
                family=family,
                bundle_path=bundle_root,
                model_key=model_key,
                entry=entry,
                version_key=version_key,
                model_cache=pipeline._cache,
                on_log=_log,
            )

    return UpscaleCreateRunContext(
        pipeline=pipeline,
        request=request,
        exec_ctx=ctx_exec,
        entry=entry,
        family=family,
        model_key=model_key,
        version_key=version_key,
        bundle_root=bundle_root,
        src_path=src_path,
        scale=scale,
        seed=seed,
        out_path=out_path,
        upscale_pipeline=upscale_pipeline,
        on_progress=on_progress,
        on_log=on_log,
    )


def execute_upscale_job(ctx: UpscaleCreateRunContext) -> dict[str, Any]:
    """Run upscale SR job via ``JobParadigm``."""
    pipeline_graph_step("denoise", ctx.on_log, message="upscale")
    return run_upscale_job(ctx)


def persist_upscale_create(
    ctx: UpscaleCreateRunContext,
    extra: dict[str, Any],
) -> tuple[str, dict[str, Any]] | None:
    from PIL import Image

    if ctx.exec_ctx.cancel_token.is_cancelled():
        return None

    pipeline_graph_step("save_asset", ctx.on_log)
    with Image.open(ctx.out_path) as pil:
        w, h = pil.size
    if ctx.on_progress:
        ctx.on_progress(1.0, 1, 1, None)

    meta = {
        "model": ctx.request.model,
        "width": w,
        "height": h,
        "mime_type": "image/png",
        "scale": ctx.scale,
        "denoise": float(ctx.request.denoise),
    }
    meta.update(extra)
    return str(ctx.out_path), meta


