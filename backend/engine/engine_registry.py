"""按 model_id 解析媒体引擎 — plan 7.7。"""

from __future__ import annotations

from backend.core.media_interfaces import IAudioEngine, IImageEngine, IVideoEngine
from backend.core.model_registry import ModelRegistry


class EngineRegistry:
    def __init__(self, model_registry: ModelRegistry) -> None:
        self._model_registry = model_registry
        self._by_engine_id: dict[str, object] = {}

    def register(self, engine: IImageEngine | IVideoEngine | IAudioEngine) -> None:
        self._by_engine_id[engine.engine_id] = engine

    def get_image(self, model_id: str) -> IImageEngine:
        cfg = self._model_registry.require(model_id)
        if cfg.media != "image":
            raise ValueError(f"model {model_id} is not an image model (media={cfg.media})")
        eng = self._by_engine_id.get(cfg.engine)
        if not isinstance(eng, IImageEngine):
            raise TypeError(f"engine {cfg.engine!r} is not configured as IImageEngine")
        return eng

    def get_video(self, model_id: str) -> IVideoEngine:
        cfg = self._model_registry.require(model_id)
        if cfg.media != "video":
            raise ValueError(f"model {model_id} is not a video model (media={cfg.media})")
        eng = self._by_engine_id.get(cfg.engine)
        if not isinstance(eng, IVideoEngine):
            raise TypeError(f"engine {cfg.engine!r} is not configured as IVideoEngine")
        return eng

    def get_audio(self, model_id: str) -> IAudioEngine:
        cfg = self._model_registry.require(model_id)
        if cfg.media != "audio":
            raise ValueError(f"model {model_id} is not an audio model (media={cfg.media})")
        eng = self._by_engine_id.get(cfg.engine)
        if not isinstance(eng, IAudioEngine):
            raise TypeError(f"engine {cfg.engine!r} is not configured as IAudioEngine")
        return eng
