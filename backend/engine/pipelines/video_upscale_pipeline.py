"""VideoUpscalePipeline — 视频超分装配线（与 ``VideoPipeline`` 平级）。"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from backend.core.contracts import ExecutionContext, VideoUpscaleRequest, parse_model_version
from backend.engine.common.cache import ModelCache
from backend.engine.runtime._base import RuntimeContext


class VideoUpscalePipeline:
    """注册表驱动的视频超分占位；具体后端接入前显式失败。"""

    def __init__(
        self,
        ctx: RuntimeContext,
        model_registry: Any,
        asset_store: Any,
        model_cache: ModelCache | None = None,
        project_root: Path | None = None,
    ) -> None:
        self.ctx = ctx
        self._registry = model_registry
        self._asset_store = asset_store
        self._cache = model_cache
        self._project_root = project_root or Path.cwd()

    def run(
        self,
        request: VideoUpscaleRequest,
        ctx_exec: ExecutionContext,
        *,
        on_progress: Callable | None = None,
        on_log: Callable | None = None,
    ):
        if ctx_exec.cancel_token.is_cancelled():
            return None

        model_key, _vk = parse_model_version(request.model)
        entry = self._registry.require(model_key)
        if getattr(entry, "media", None) != "video":
            raise RuntimeError(
                f"Video upscale model {model_key!r} is not a video model (media={getattr(entry, 'media', None)!r})."
            )

        raise RuntimeError(
            "Video upscale is not implemented in DanQing Studio yet; "
            "VideoUpscalePipeline is reserved for future video super-resolution backends."
        )
