"""
模型缓存 — LRU + TTL，后端无关。

共享给 MLX 和 CUDA 后端使用。
"""
from __future__ import annotations

import asyncio
import gc
import threading
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable, Optional


@dataclass
class _CacheEntry:
    model: Any
    size_gb: float
    last_used: datetime
    cache_key: str


class ModelCache:
    """LRU + TTL 模型缓存。

    参数:
        get_memory_limit: 返回内存上限 (GB) 的回调
        reserve_gb: 为系统和运行时预留的 GB
        ttl_minutes: 缓存条目空闲多久后主动淘汰
        release_fn: 自定义释放函数
    """

    def __init__(
        self,
        get_memory_limit: Callable[[], float],
        *,
        reserve_gb: float = 20.0,
        ttl_minutes: int = 30,
        release_fn: Optional[Callable[[Any], None]] = None,
    ):
        self._get_memory_limit = get_memory_limit
        self._reserve = reserve_gb
        self._ttl_minutes = ttl_minutes
        self._release_fn = release_fn or self._default_release
        self._cache: OrderedDict[str, _CacheEntry] = OrderedDict()
        self._lock = threading.Lock()
        self._total_gb = 0.0
        self._cleanup_task: Optional[asyncio.Task] = None

    @staticmethod
    def _default_release(model: Any) -> None:
        del model
        gc.collect()

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

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            if datetime.now() - entry.last_used > self._ttl:
                self._evict(key)
                return None
            self._cache.move_to_end(key)
            entry.last_used = datetime.now()
            return entry.model

    def put(self, key: str, model: Any, size_gb: float) -> None:
        with self._lock:
            if key in self._cache:
                old = self._cache.pop(key)
                self._total_gb -= old.size_gb

            limit = self._limit_gb
            while self._total_gb + size_gb > limit and self._cache:
                oldest = next(iter(self._cache))
                self._evict(oldest)

            self._cache[key] = _CacheEntry(
                model=model, size_gb=size_gb,
                last_used=datetime.now(), cache_key=key,
            )
            self._cache.move_to_end(key)
            self._total_gb += size_gb

    def unload_all(self) -> None:
        with self._lock:
            keys = list(self._cache.keys())
        for k in keys:
            self._evict(k)
        gc.collect()

    def start_cleanup(self, loop: asyncio.AbstractEventLoop) -> None:
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = loop.create_task(self._cleanup_loop())

    def stop_cleanup(self) -> None:
        if self._cleanup_task:
            self._cleanup_task.cancel()
            self._cleanup_task = None

    @property
    def stats(self) -> dict:
        with self._lock:
            return {
                "cached_models": len(self._cache),
                "total_gb": round(self._total_gb, 2),
                "limit_gb": self._limit_gb,
                "ttl_minutes": self._ttl_minutes,
                "models": [
                    {
                        "key": k,
                        "size_gb": v.size_gb,
                        "idle_minutes": round(
                            (datetime.now() - v.last_used).total_seconds() / 60, 1
                        ),
                    }
                    for k, v in self._cache.items()
                ],
            }

    def _evict(self, key: str) -> None:
        entry = self._cache.pop(key, None)
        if entry is None:
            return
        self._total_gb -= entry.size_gb
        self._release_fn(entry.model)

    def _evict_expired(self) -> None:
        now = datetime.now()
        with self._lock:
            expired = [
                k for k, v in self._cache.items()
                if now - v.last_used > self._ttl
            ]
        for k in expired:
            self._evict(k)

    async def _cleanup_loop(self) -> None:
        while True:
            await asyncio.sleep(60)
            self._evict_expired()
