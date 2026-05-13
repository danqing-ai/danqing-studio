"""DanQing 音频引擎 — 音乐生成管线接入，支持 ACE-Step DiT + VAE。"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, List

from backend.core.contracts import (
    AudioEditRequest,
    AudioGenerationRequest,
    EngineResult,
    ExecutionContext,
    LogEvent,
    parse_model_version,
)
from backend.core.interfaces import IPathResolver
from backend.core.media_interfaces import IAudioEngine
from backend.core.model_registry import ModelRegistry


def _version_config(entry_raw: dict[str, Any], version_key: str) -> dict[str, Any]:
    versions = entry_raw.get("versions")
    if not isinstance(versions, dict) or not versions:
        return {}
    if version_key and version_key in versions:
        cfg = versions[version_key]
        return cfg if isinstance(cfg, dict) else {}
    for _k, cfg in versions.items():
        if isinstance(cfg, dict) and cfg.get("default"):
            return cfg
    first = next(iter(versions.values()))
    return first if isinstance(first, dict) else {}


def _resolve_models_subpath(path_resolver: IPathResolver, local_path: str) -> Path:
    if local_path.startswith("models/"):
        rel = local_path[len("models/"):]
        return path_resolver.get_models_dir() / rel
    return Path(local_path).expanduser()


def _dir_ready(p: Path) -> bool:
    return p.is_dir() and any(p.iterdir())


class DanQingAudioEngine(IAudioEngine):
    media_type = "audio"
    engine_id = "danqing-audio"

    def __init__(
        self,
        path_resolver: IPathResolver,
        registry: ModelRegistry,
        runtimes: dict[str, Any],
        model_cache: Any = None,
    ):
        self._paths = path_resolver
        self._registry = registry
        self._runtimes = runtimes
        self._cache = model_cache

    def is_available(self) -> bool:
        return True

    def is_model_ready(self, model_name: str, version: str = "") -> bool:
        m, v = parse_model_version(model_name) if ":" in model_name else (model_name, version)
        entry = self._registry.get(m)
        if not entry or entry.media != "audio":
            return False
        raw = entry.raw
        if isinstance(raw, dict) and raw.get("stub_no_download"):
            return True
        ver_cfg = _version_config(raw, v)
        lp = ver_cfg.get("local_path")
        if not lp:
            return False
        dit_path = _resolve_models_subpath(self._paths, str(lp))
        if not _dir_ready(dit_path):
            return False
        clp = ver_cfg.get("companion_local_path")
        if isinstance(clp, str) and clp.strip():
            lm_path = _resolve_models_subpath(self._paths, clp.strip())
            if not _dir_ready(lm_path):
                return False
        return True

    def get_supported_models(self) -> List[str]:
        return [mid for mid, e in self._registry.all().items() if e.media == "audio"]

    def supports(self, model_id: str, action: str) -> bool:
        e = self._registry.get(model_id)
        if not e or e.media != "audio":
            return False
        return action in e.actions

    def _resolve_runtime(self, entry: Any) -> Any:
        """Select the appropriate RuntimeContext for this model."""
        backends = getattr(entry, "backends", None) or entry.raw.get("backends", ["mlx"])
        for b in backends:
            rt = self._runtimes.get(b)
            if rt is not None:
                return rt
        if self._runtimes:
            return next(iter(self._runtimes.values()))
        raise RuntimeError("No runtime available")

    async def generate(
        self, request: AudioGenerationRequest, ctx: ExecutionContext
    ) -> EngineResult:
        if not self.supports(request.model, "create_music"):
            mid = request.model.split(":", 1)[0]
            raise RuntimeError(
                f"Model {mid!r} does not support text-to-music (create); "
                "see config/models_registry.json actions."
            )

        entry = self._registry.get(request.model)
        if entry is None:
            raise RuntimeError(f"Model {request.model!r} not found in registry")

        is_stub = entry.raw.get("stub_no_download", False)
        if is_stub:
            ctx.on_log(LogEvent(level="error", message="Audio stub model has no backend"))
            raise RuntimeError(
                "Audio generation is not implemented for the stub model. "
                "Use a real audio model (e.g. ace-step-xl-sft)."
            )

        ctx.on_log(LogEvent(level="info", message=f"Initializing audio pipeline for {request.model}"))

        runtime = self._resolve_runtime(entry)

        from backend.engine.pipelines.music_pipeline import MusicPipeline
        pipeline = MusicPipeline(
            ctx=runtime,
            model_registry=self._registry,
            asset_store=ctx.asset_store,
            model_cache=self._cache,
            project_root=self._paths.get_project_root() if hasattr(self._paths, "get_project_root") else None,
        )

        result = await asyncio.to_thread(pipeline.run, request, ctx)
        return result

    async def edit(self, request: AudioEditRequest, ctx: ExecutionContext) -> EngineResult:
        if not self.supports(request.model, "edit"):
            raise RuntimeError(
                f"Model {request.model!r} does not declare audio edit (cover/repaint); "
                "see config/models_registry.json."
            )
        ctx.on_log(
            LogEvent(
                level="error",
                message="Audio edit is not implemented; only the API surface is retained.",
            )
        )
        raise RuntimeError(
            "Audio edit is not implemented (音频编辑未实现). "
            "Only the HTTP API contract is retained."
        )

    async def cancel(self, task_id: str) -> bool:
        return False
