"""MemoryGuard — 步进式内存守卫。

借鉴 OminiX-MLX ``mlx-rs-core/src/memory.rs`` MemoryGuard:
- ``eval_interval``: 每 N 步强制 eval（防止 MLX lazy graph 堆积）
- ``clear_interval``: 每 M 步清理缓存（释放已弃用的 MLX arrays）
- ``pressure_threshold``: active/peak > 阈值时主动清理

替代分散在 pipeline / model 中的 ``ctx.eval()`` / ``ctx.clear_cache()`` 调用。
注意：block 级 eval（Flux1/LTX transformer 内部）留在 Model (L3) —
MemoryGuard 只管 step 级别。
"""
from __future__ import annotations

from typing import Any


class MemoryGuard:
    """步进式内存守卫 — 统一管理 denoise loop 中的 eval + clear_cache。"""

    def __init__(
        self,
        ctx: Any,
        *,
        eval_interval: int = 1,
        clear_interval: int = 0,
        pressure_threshold: float = 0.0,
    ) -> None:
        self._ctx = ctx
        self._eval_interval = eval_interval
        self._clear_interval = clear_interval
        self._pressure_threshold = pressure_threshold
        self._step = 0

    def step(self, *arrays: Any) -> bool:
        """每步调用。返回 ``True`` 表示执行了 cache clear。"""
        self._step += 1
        # 1) 强制 eval — MLX: 刷新 lazy graph, 防止 Metal 超时
        if arrays and self._step % self._eval_interval == 0:
            self._ctx.eval(*arrays)
        # 2) 定期清理
        cleared = False
        if self._clear_interval and self._step % self._clear_interval == 0:
            self._ctx.clear_cache()
            cleared = True
        # 3) 压力清理
        elif self._pressure_threshold > 0:
            active = self._ctx.active_memory_gb()
            peak = getattr(self._ctx, "peak_memory_gb", lambda: 0)()
            if peak > 0 and active / peak > self._pressure_threshold:
                self._ctx.clear_cache()
                cleared = True
        return cleared
