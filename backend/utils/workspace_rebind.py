"""Rebind in-process services after custom workspace relocation (no full restart)."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from backend.core.container import get_container
from backend.core.interfaces import IPathResolver
from backend.persistence.asset_store import SQLiteAssetStore
from backend.scheduler.task_scheduler import TaskScheduler
from backend.utils.path_utils import PathResolver

_logger = logging.getLogger(__name__)


def rebind_workspace_after_relocation(path_resolver: IPathResolver) -> Path:
    """Update PathResolver + SQLite stores to the new workspace; refresh task queue."""
    if not isinstance(path_resolver, PathResolver):
        raise TypeError("rebind_workspace_after_relocation requires PathResolver")
    new_root = path_resolver.reload_workspace_root()
    db_path = new_root / "db" / "studio.db"
    asset_root = new_root / "outputs" / "assets"

    container = get_container()
    asset_store = container.try_resolve(SQLiteAssetStore)
    if asset_store is not None:
        asset_store.rebind(db_path, asset_root)

    scheduler = container.try_resolve(TaskScheduler)
    if scheduler is not None:
        scheduler.rebind_task_store(db_path)
        _schedule_queue_rebuild(scheduler)

    _logger.info("Rebound persistence to workspace root %s", new_root)
    return new_root


def _schedule_queue_rebuild(scheduler: TaskScheduler) -> None:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(scheduler.rebuild_queued_heap_sync())
        return
    loop.create_task(scheduler.rebuild_queued_heap_sync())
