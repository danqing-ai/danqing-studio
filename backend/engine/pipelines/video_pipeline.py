"""VideoPipeline — runtime holder for video phased helpers (ops live in ``video_run_common``)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from backend.engine.cache import ModelCache
from backend.engine.common.codecs.text_encoders import T5Encoder
from backend.engine.runtime._base import RuntimeContext


class VideoPipeline:
    """Registry / asset / runtime context bundle for video create + edit."""

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
        self._t5: T5Encoder | None = None
        self._t5_bundle_root: Path | None = None
        self._t5_max_seq_len: int | None = None
        self._video_config: Any | None = None
