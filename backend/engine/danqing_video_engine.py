"""
DanQingVideoEngine — IVideoEngine 的丹青实现。

聚合多个 RuntimeContext (MLX/CUDA)，按模型注册表自动选择后端。
"""
from __future__ import annotations

from typing import ClassVar, List

from backend.core.contracts import (
    EngineResult, ExecutionContext, VideoEditRequest,
    VideoGenerationRequest, VideoUpscaleRequest, parse_model_version,
)
from backend.core.media_interfaces import IVideoEngine
from backend.core.interfaces import IPathResolver

from .common.cache import ModelCache
from .runtime._base import RuntimeContext


class DanQingVideoEngine(IVideoEngine):
    """丹青视频引擎。

    后端无关：通过 Registry 选择模型 → 自动路由到 MLX/CUDA RuntimeContext。
    """

    media_type: ClassVar[str] = "video"
    engine_id: ClassVar[str] = "danqing-video"  # matches registry entries

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
        return [
            mid for mid, entry in self._registry.entries.items()
            if entry.media == "video"
        ]

    def supports(self, model_id: str, action: str) -> bool:
        e = self._registry.get(model_id)
        if not e or e.media != "video":
            return False
        return action in e.actions

    def _resolve_runtime(self, model_id: str) -> RuntimeContext:
        entry = self._registry.require(model_id)
        backends = getattr(entry, "backends", ["mlx"])
        for b in backends:
            if b in self._runtimes:
                return self._runtimes[b]
        raise RuntimeError(f"No available backend for model {model_id} (backends: {backends})")

    async def generate(self, request: VideoGenerationRequest,
                       ctx: ExecutionContext) -> EngineResult:
        runtime = self._resolve_runtime(request.model)
        from .video_pipeline import VideoPipeline
        pipeline = VideoPipeline(
            runtime, self._registry, ctx.asset_store,
            model_cache=self._cache,
            text_encoders_path=str(self._paths.get_models_dir()),
        )
        return await pipeline.generate(request, ctx)

    async def edit(self, request: VideoEditRequest,
                   ctx: ExecutionContext) -> EngineResult:
        runtime = self._resolve_runtime(request.model)
        from .video_pipeline import VideoPipeline
        pipeline = VideoPipeline(
            runtime, self._registry, ctx.asset_store,
            model_cache=self._cache,
        )
        return await pipeline.edit(request, ctx)

    async def upscale(self, request: VideoUpscaleRequest,
                      ctx: ExecutionContext) -> EngineResult:
        runtime = self._resolve_runtime(request.model)
        from .video_pipeline import VideoPipeline
        pipeline = VideoPipeline(
            runtime, self._registry, ctx.asset_store,
            model_cache=self._cache,
        )
        return await pipeline.upscale(request, ctx)

    async def cancel(self, task_id: str) -> bool:
        return True
