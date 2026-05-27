"""
CLI 共享初始化 — 提取 main.py 中的引擎构建逻辑。

CLI 和 REST API 都通过此模块初始化 Engine，确保路径一致。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from backend.core.contracts import CancelToken, ExecutionContext
from backend.core.model_registry import ModelRegistry
from backend.engine.common.cache import ModelCache
from backend.engine.memory_policy import (
    apply_memory_settings,
    build_gpu_runtimes,
    build_shared_model_cache,
)
from backend.engine.danqing_audio_engine import DanQingAudioEngine
from backend.engine.danqing_image_engine import DanQingImageEngine
from backend.engine.danqing_video_engine import DanQingVideoEngine
from backend.engine.engine_registry import EngineRegistry
from backend.engine.platform import PlatformInfo
from backend.persistence.asset_store import SQLiteAssetStore
from backend.persistence.stores import JsonConfigStore
from backend.utils.path_utils import PathResolver


@dataclass
class EngineContext:
    """Engine 运行所需的依赖集合。"""
    path_resolver: PathResolver
    model_registry: ModelRegistry
    engine_registry: EngineRegistry
    image_engine: DanQingImageEngine
    video_engine: DanQingVideoEngine
    audio_engine: DanQingAudioEngine
    runtimes: dict[str, Any]
    asset_store: SQLiteAssetStore
    model_cache: ModelCache


def build_engine_context(project_root: Path | None = None) -> EngineContext:
    """构建 Engine 上下文。CLI 和 REST API 都复用此函数。"""
    path_resolver = PathResolver(project_root)
    root = path_resolver.get_project_root()

    registry_json = path_resolver.get_models_registry_path()
    model_registry = ModelRegistry.load(registry_json)
    config_store = JsonConfigStore(path_resolver)

    app_settings = config_store.load()
    shared_cache = build_shared_model_cache(config_store.load)
    platforms = PlatformInfo.detect()
    runtimes = build_gpu_runtimes(app_settings)
    if not runtimes:
        raise RuntimeError("No GPU backend available (need MLX on Apple Silicon or CUDA on NVIDIA)")
    apply_memory_settings(app_settings, runtimes, shared_cache)

    image_engine = DanQingImageEngine(
        path_resolver, model_registry, runtimes, model_cache=shared_cache,
    )
    video_engine = DanQingVideoEngine(
        path_resolver, model_registry, runtimes, model_cache=shared_cache,
    )
    audio_engine = DanQingAudioEngine(
        path_resolver, model_registry, runtimes, model_cache=shared_cache,
    )

    engine_registry = EngineRegistry(model_registry)
    engine_registry.register(image_engine)
    engine_registry.register(video_engine)
    engine_registry.register(audio_engine)


    v3_db = root / "db" / "studio.db"
    asset_root = root / "outputs" / "assets"
    asset_store = SQLiteAssetStore(v3_db, asset_root)

    return EngineContext(
        path_resolver=path_resolver,
        model_registry=model_registry,
        engine_registry=engine_registry,
        image_engine=image_engine,
        video_engine=video_engine,
        audio_engine=audio_engine,
        runtimes=runtimes,
        asset_store=asset_store,
        model_cache=shared_cache,
    )


def build_exec_context(
    task_id: str = "cli",
    work_dir: Path | None = None,
    asset_store: SQLiteAssetStore | None = None,
    on_progress=None,
    on_log=None,
) -> ExecutionContext:
    """构建 ExecutionContext。CLI 使用简化的回调。"""
    if work_dir is None:
        work_dir = Path.cwd() / "outputs" / "cli_tmp"
    work_dir.mkdir(parents=True, exist_ok=True)

    if on_progress is None:
        on_progress = lambda ev: None
    if on_log is None:
        on_log = lambda ev: None

    return ExecutionContext(
        task_id=task_id,
        cancel_token=CancelToken(),
        on_progress=on_progress,
        on_log=on_log,
        work_dir=work_dir,
        asset_store=asset_store or _NullAssetStore(),
    )


async def run_engine_task(
    *,
    ctx: EngineContext,
    kind: str,
    model_id: str,
    request: Any,
) -> Any:
    """Shared async engine dispatch for CLI (mirrors TaskScheduler dispatch table)."""
    from backend.scheduler.task_dispatch import TASK_DISPATCH

    spec = TASK_DISPATCH.get(kind)
    if spec is None:
        raise RuntimeError(f"unknown task kind {kind!r}")
    if not isinstance(request, spec.request_cls):
        request = spec.request_cls.model_validate(request)
    exec_ctx = build_exec_context(asset_store=ctx.asset_store)
    getter = getattr(ctx.engine_registry, spec.engine_getter)
    engine = getter(model_id)
    method = getattr(engine, spec.method_name)
    return await method(request, exec_ctx)


class _NullAssetStore:
    """CLI 不需要持久化资产时的空实现。"""
    def create_from_file(self, *a, **kw): return "cli_ast"
    def get(self, *a, **kw): return None
    def list_assets(self, *a, **kw): return []
