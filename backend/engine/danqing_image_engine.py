"""
DanQingImageEngine — IImageEngine 实现。

MLX 操作在 TaskScheduler 线程同步执行（全局串行队列保证互斥）。
"""
from __future__ import annotations

from pathlib import Path
from typing import ClassVar, List, Any

from backend.core.contracts import (
    EngineResult, ExecutionContext, ImageEditRequest,
    ImageGenerationRequest, ImageUpscaleRequest, parse_model_version,
)
from backend.core.media_interfaces import IImageEngine
from backend.core.interfaces import IPathResolver
from .common.cache import ModelCache
from .pipelines.image_pipeline import ImagePipeline
from .pipelines.image_upscale_pipeline import ImageUpscalePipeline
from .progress_bridge import make_pipeline_progress_callback
from .runtime._base import RuntimeContext


class DanQingImageEngine(IImageEngine):
    media_type: ClassVar[str] = "image"
    engine_id: ClassVar[str] = "danqing-image"

    def __init__(
        self,
        path_resolver: IPathResolver,
        registry: Any,
        runtimes: dict[str, RuntimeContext],
        model_cache: ModelCache | None = None,
    ):
        self._paths = path_resolver
        self._registry = registry
        self._runtimes = runtimes
        self._cache = model_cache

    def is_available(self) -> bool:
        return len(self._runtimes) > 0

    def is_model_ready(self, model_name: str, version: str = "") -> bool:
        m, v = parse_model_version(model_name) if ":" in model_name else (model_name, version)
        try:
            entry = self._registry.require(m)
            backends = entry.backends
            for b in backends:
                if b in self._runtimes:
                    return True
        except Exception:
            pass
        return False

    def get_supported_models(self) -> List[str]:
        return [mid for mid, entry in self._registry.entries.items() if entry.media == "image"]

    def supports(self, model_id: str, action: str) -> bool:
        e = self._registry.get(model_id)
        if not e or e.media != "image":
            return False
        return action in e.actions

    def _resolve_runtime(self, model_id: str) -> RuntimeContext:
        entry = self._registry.require(model_id)
        backends = entry.backends
        for b in backends:
            if b in self._runtimes:
                return self._runtimes[b]
        raise RuntimeError(f"No available backend for model {model_id} (backends: {backends})")

    async def generate(self, request: ImageGenerationRequest,
                       ctx: ExecutionContext) -> EngineResult:
        import asyncio
        if not self.supports(request.model, "generate"):
            mid = request.model.split(":", 1)[0]
            raise RuntimeError(
                f"Model {mid!r} does not support text-to-image (create) for this engine; "
                "see config/models_registry.json actions."
            )
        runtime = self._resolve_runtime(request.model)
        pipeline = ImagePipeline(
            runtime,
            self._registry,
            ctx.asset_store,
            model_cache=self._cache,
            project_root=self._paths.get_project_root(),
        )

        on_progress = make_pipeline_progress_callback(ctx)

        def on_log(lvl, msg):
            from backend.core.contracts import LogEvent
            ctx.on_log(LogEvent(level=lvl, message=msg))

        result = await asyncio.to_thread(
            pipeline.run, request, ctx, on_progress=on_progress, on_log=on_log,
        )
        if result is None:
            return EngineResult(primary_asset_id="", metadata={"status": "cancelled"})

        if isinstance(result, list):
            asset_ids: list[str] = []
            output_paths: list[str] = []
            for output_path, metadata in result:
                aid = ctx.asset_store.create_from_file(
                    Path(output_path), kind="image", mime_type="image/png",
                    source_task_id=ctx.task_id, metadata=metadata, source_action="create",
                )
                asset_ids.append(aid)
                output_paths.append(output_path)
            return EngineResult(
                primary_asset_id=asset_ids[0] if asset_ids else "",
                asset_ids=asset_ids,
                output_paths=output_paths,
            )

        output_path, metadata = result
        aid = ctx.asset_store.create_from_file(
            Path(output_path), kind="image", mime_type="image/png",
            source_task_id=ctx.task_id, metadata=metadata, source_action="create",
        )
        return EngineResult(primary_asset_id=aid, asset_ids=[aid], output_paths=[output_path])

    async def edit(self, request: ImageEditRequest, ctx: ExecutionContext) -> EngineResult:
        import asyncio
        if not self.supports(request.model, "edit"):
            mid = request.model.split(":", 1)[0]
            raise RuntimeError(
                f"Model {mid!r} does not support image edit (rewrite/retouch/extend) for this engine; "
                "see config/models_registry.json actions."
            )
        runtime = self._resolve_runtime(request.model)
        pipeline = ImagePipeline(
            runtime, self._registry, ctx.asset_store,
            model_cache=self._cache,
            project_root=self._paths.get_project_root(),
        )

        on_progress = make_pipeline_progress_callback(ctx)

        def on_log(lvl, msg):
            from backend.core.contracts import LogEvent
            ctx.on_log(LogEvent(level=lvl, message=msg))

        result = await asyncio.to_thread(
            pipeline.run_edit, request, ctx, on_progress=on_progress, on_log=on_log,
        )
        if result is None:
            return EngineResult(primary_asset_id="", metadata={"status": "cancelled"})

        output_path, metadata = result
        aid = ctx.asset_store.create_from_file(
            Path(output_path), kind="image", mime_type="image/png",
            source_task_id=ctx.task_id, metadata=metadata, source_action="rewrite",
        )
        return EngineResult(primary_asset_id=aid, asset_ids=[aid], output_paths=[output_path])

    async def upscale(self, request: ImageUpscaleRequest, ctx: ExecutionContext) -> EngineResult:
        if not self.supports(request.model, "upscale"):
            mid = request.model.split(":", 1)[0]
            raise RuntimeError(
                f"Model {mid!r} does not support upscale for this engine; "
                "see config/models_registry.json actions."
            )
        runtime = self._resolve_runtime(request.model)
        pipeline = ImageUpscalePipeline(
            runtime,
            self._registry,
            ctx.asset_store,
            model_cache=self._cache,
            project_root=self._paths.get_project_root(),
        )
        import asyncio

        on_progress = make_pipeline_progress_callback(ctx)

        def on_log(lvl, msg):
            from backend.core.contracts import LogEvent
            ctx.on_log(LogEvent(level=lvl, message=msg))

        result = await asyncio.to_thread(
            pipeline.run, request, ctx, on_progress=on_progress, on_log=on_log,
        )
        if result is None:
            return EngineResult(primary_asset_id="", metadata={"status": "cancelled"})

        output_path, metadata = result
        aid = ctx.asset_store.create_from_file(
            Path(output_path), kind="image", mime_type="image/png",
            source_task_id=ctx.task_id, metadata=metadata, source_action="upscale",
        )
        return EngineResult(primary_asset_id=aid, asset_ids=[aid], output_paths=[output_path])

    async def cancel(self, task_id: str) -> bool:
        return True
