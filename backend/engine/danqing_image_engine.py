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
            backends = getattr(entry, "backends", [list(self._runtimes.keys())[0]])
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
        backends = getattr(entry, "backends", ["mlx"])
        for b in backends:
            if b in self._runtimes:
                return self._runtimes[b]
        raise RuntimeError(f"No available backend for model {model_id} (backends: {backends})")

    async def generate(self, request: ImageGenerationRequest,
                       ctx: ExecutionContext) -> EngineResult:
        import asyncio
        runtime = self._resolve_runtime(request.model)
        from .image_pipeline import ImagePipeline
        pipeline = ImagePipeline(
            runtime,
            self._registry,
            ctx.asset_store,
            model_cache=self._cache,
            project_root=self._paths.get_project_root(),
        )

        def on_progress(p, s, t, msg):
            from backend.core.contracts import ProgressEvent
            ctx.on_progress(ProgressEvent(progress=p, step=s, total=t, message=msg))

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
            source_task_id=ctx.task_id, metadata=metadata, source_action="create",
        )
        return EngineResult(primary_asset_id=aid, asset_ids=[aid], output_paths=[output_path])

    async def edit(self, request: ImageEditRequest, ctx: ExecutionContext) -> EngineResult:
        import asyncio
        runtime = self._resolve_runtime(request.model)
        from .image_pipeline import ImagePipeline
        pipeline = ImagePipeline(
            runtime, self._registry, ctx.asset_store,
            model_cache=self._cache,
            project_root=self._paths.get_project_root(),
        )
        result = await asyncio.to_thread(
            pipeline.run_edit, request, ctx, on_progress=None, on_log=None,
        )
        if result is None:
            return EngineResult(primary_asset_id="")
        output_path, _ = result
        from backend.core.contracts import new_asset_id
        return EngineResult(primary_asset_id=new_asset_id(), output_paths=[output_path or ""])

    async def upscale(self, request: ImageUpscaleRequest, ctx: ExecutionContext) -> EngineResult:
        raise NotImplementedError("upscale not implemented")

    async def cancel(self, task_id: str) -> bool:
        return True
