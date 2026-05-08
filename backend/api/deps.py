"""Plan 7.10：路由层统一 Depends 入口。"""

from __future__ import annotations

from backend.core.container import get_container
from backend.core.model_registry import ModelRegistry
from backend.engine.engine_registry import EngineRegistry
from backend.persistence.asset_store import SQLiteAssetStore
from backend.scheduler.task_scheduler import TaskScheduler


def get_task_scheduler() -> TaskScheduler:
    return get_container().resolve(TaskScheduler)


def get_engine_registry() -> EngineRegistry:
    return get_container().resolve(EngineRegistry)


def get_asset_store() -> SQLiteAssetStore:
    return get_container().resolve(SQLiteAssetStore)


def get_model_registry() -> ModelRegistry:
    return get_container().resolve(ModelRegistry)
