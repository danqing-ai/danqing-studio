"""VideoUpscalePipeline — registry-driven video SR (与 ``VideoPipeline`` 平级）。"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from backend.engine.cache import ModelCache
from backend.engine.runtime._base import RuntimeContext


class VideoUpscalePipeline:
    """注册表驱动的视频超分装配线 holder（执行经 ``VideoUpscaleSession``）。"""

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
