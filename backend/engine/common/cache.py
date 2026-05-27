"""
模型缓存 — LRU + TTL，后端无关。

个人工具默认 ``max_entries=1``：同时最多保留一份重模型。
空闲超过 ``ttl_minutes`` 后由 ``get`` / 后台 ``purge_idle`` 自动卸载并释放 GPU 缓存。
"""
from __future__ import annotations

import asyncio
import gc
import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable, Optional

_logger = logging.getLogger(__name__)


@dataclass
class _CacheEntry:
    model: Any
    size_gb: float
    last_used: datetime
    cache_key: str


class ModelCache:
    """LRU + TTL 模型缓存；``max_entries=1`` 时为全局单槽。

    参数:
        get_memory_limit: 返回内存上限 (GB) 的回调（``max_entries > 1`` 时用于 LRU 预算）
        reserve_gb: 为系统和运行时预留的 GB
        ttl_minutes: 缓存条目空闲多久后主动淘汰（设置 → ``model_cache_ttl_minutes``）
        max_entries: 最多条目数；``1`` = 个人工具单槽
        release_fn: 自定义释放函数
    """

    def __init__(
        self,
        get_memory_limit: Callable[[], float],
        *,
        reserve_gb: float = 20.0,
        ttl_minutes: int = 30,
        max_entries: int = 1,
        release_fn: Optional[Callable[[Any], None]] = None,
    ):
        self._get_memory_limit = get_memory_limit
        self._reserve = reserve_gb
        self._ttl_minutes = ttl_minutes
        self._max_entries = max(1, int(max_entries))
        self._release_fn = release_fn or self._default_release
        self._cache: OrderedDict[str, _CacheEntry] = OrderedDict()
        self._lock = threading.Lock()
        self._total_gb = 0.0
        self._cleanup_task: Optional[asyncio.Task] = None
        self._cleanup_thread: Optional[threading.Thread] = None
        self._cleanup_stop = threading.Event()

    @staticmethod
    def _default_release(model: Any) -> None:
        del model
        gc.collect()

    def set_ttl_minutes(self, minutes: int) -> None:
        self._ttl_minutes = max(1, int(minutes))

    @property
    def max_entries(self) -> int:
        return self._max_entries

    @property
    def ttl_minutes(self) -> int:
        return self._ttl_minutes

    @property
    def _limit_gb(self) -> float:
        try:
            limit = self._get_memory_limit()
        except Exception:
            limit = 120.0
        return max(float(limit) - self._reserve, 10.0)

    @property
    def _ttl(self) -> timedelta:
        return timedelta(minutes=self._ttl_minutes)

    @property
    def cleanup_interval_seconds(self) -> float:
        """Background idle purge cadence (~¼ TTL, clamped 15s–5min)."""
        ttl_s = float(self._ttl_minutes) * 60.0
        return max(15.0, min(300.0, ttl_s / 4.0))

    def _is_expired(self, entry: _CacheEntry, now: datetime | None = None) -> bool:
        now = now or datetime.now()
        return now - entry.last_used > self._ttl

    def _idle_minutes(self, entry: _CacheEntry, now: datetime | None = None) -> float:
        now = now or datetime.now()
        return (now - entry.last_used).total_seconds() / 60.0

    def _pop_entry_unlocked(self, key: str) -> _CacheEntry | None:
        entry = self._cache.pop(key, None)
        if entry is None:
            return None
        self._total_gb -= entry.size_gb
        return entry

    def _release_entry(self, key: str, entry: _CacheEntry, *, reason: str) -> None:
        idle = self._idle_minutes(entry)
        _logger.info(
            "ModelCache unload key=%s reason=%s idle_minutes=%.1f ttl_minutes=%d",
            key,
            reason,
            idle,
            self._ttl_minutes,
        )
        self._release_fn(entry.model)

    def get(self, key: str) -> Optional[Any]:
        stale: _CacheEntry | None = None
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            if self._is_expired(entry):
                stale = self._pop_entry_unlocked(key)
            else:
                self._cache.move_to_end(key)
                entry.last_used = datetime.now()
                return entry.model
        if stale is not None:
            self._release_entry(key, stale, reason="ttl_on_get")
        return None

    def put(self, key: str, model: Any, size_gb: float) -> None:
        evicted: list[tuple[str, _CacheEntry, str]] = []
        with self._lock:
            if key in self._cache:
                old = self._cache.pop(key)
                self._total_gb -= old.size_gb
            elif self._max_entries == 1 and self._cache:
                for old_key in list(self._cache.keys()):
                    stale = self._pop_entry_unlocked(old_key)
                    if stale is not None:
                        evicted.append((old_key, stale, "single_slot_replace"))
            elif self._max_entries > 1 and len(self._cache) >= self._max_entries:
                while len(self._cache) >= self._max_entries and self._cache:
                    oldest = next(iter(self._cache))
                    stale = self._pop_entry_unlocked(oldest)
                    if stale is not None:
                        evicted.append((oldest, stale, "max_entries"))

            if self._max_entries > 1:
                limit = self._limit_gb
                while self._total_gb + size_gb > limit and self._cache:
                    oldest = next(iter(self._cache))
                    stale = self._pop_entry_unlocked(oldest)
                    if stale is not None:
                        evicted.append((oldest, stale, "memory_budget"))

            self._cache[key] = _CacheEntry(
                model=model,
                size_gb=size_gb,
                last_used=datetime.now(),
                cache_key=key,
            )
            self._cache.move_to_end(key)
            self._total_gb += size_gb
        for evict_key, entry, reason in evicted:
            self._release_entry(evict_key, entry, reason=reason)

    def evict(self, key: str) -> None:
        with self._lock:
            entry = self._pop_entry_unlocked(key)
        if entry is not None:
            self._release_entry(key, entry, reason="explicit_evict")

    def purge_idle(self) -> list[str]:
        """Unload all entries idle longer than ``ttl_minutes``. Returns evicted keys."""
        now = datetime.now()
        to_release: list[tuple[str, _CacheEntry]] = []
        with self._lock:
            for key, entry in list(self._cache.items()):
                if self._is_expired(entry, now):
                    popped = self._pop_entry_unlocked(key)
                    if popped is not None:
                        to_release.append((key, popped))
        for key, entry in to_release:
            self._release_entry(key, entry, reason="ttl_idle")
        return [k for k, _ in to_release]

    def unload_all(self) -> None:
        with self._lock:
            keys = list(self._cache.keys())
        for k in keys:
            self.evict(k)
        gc.collect()

    def start_cleanup(self, loop: asyncio.AbstractEventLoop | None = None) -> None:
        """Start periodic idle purge (async task when *loop* given, else daemon thread)."""
        self.stop_cleanup()
        if loop is not None:
            self._cleanup_task = loop.create_task(self._cleanup_loop())
        else:
            self._cleanup_stop.clear()
            self._cleanup_thread = threading.Thread(
                target=self._thread_cleanup_loop,
                name="ModelCache-idle-purge",
                daemon=True,
            )
            self._cleanup_thread.start()

    def stop_cleanup(self) -> None:
        if self._cleanup_task is not None:
            self._cleanup_task.cancel()
            self._cleanup_task = None
        self._cleanup_stop.set()
        if self._cleanup_thread is not None:
            self._cleanup_thread.join(timeout=2.0)
            self._cleanup_thread = None

    @property
    def stats(self) -> dict:
        with self._lock:
            return {
                "cached_models": len(self._cache),
                "max_entries": self._max_entries,
                "total_gb": round(self._total_gb, 2),
                "limit_gb": self._limit_gb,
                "ttl_minutes": self._ttl_minutes,
                "cleanup_interval_seconds": round(self.cleanup_interval_seconds, 1),
                "models": [
                    {
                        "key": k,
                        "size_gb": v.size_gb,
                        "idle_minutes": round(self._idle_minutes(v), 1),
                        "expires_in_minutes": round(
                            max(0.0, self._ttl_minutes - self._idle_minutes(v)), 1
                        ),
                    }
                    for k, v in self._cache.items()
                ],
            }

    def _thread_cleanup_loop(self) -> None:
        while not self._cleanup_stop.wait(self.cleanup_interval_seconds):
            try:
                self.purge_idle()
            except Exception:
                _logger.exception("ModelCache idle purge failed")

    async def _cleanup_loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(self.cleanup_interval_seconds)
                self.purge_idle()
        except asyncio.CancelledError:
            return
