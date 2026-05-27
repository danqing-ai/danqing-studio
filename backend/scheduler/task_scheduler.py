"""Global single-worker task scheduling — plan TaskScheduler."""

from __future__ import annotations

import asyncio
import itertools
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Optional

from backend.core.contracts import (
    AudioEditRequest,
    AudioGenerationRequest,
    CancelToken,
    ExecutionContext,
    ImageEditRequest,
    ImageGenerationRequest,
    ImageUpscaleRequest,
    LogEvent,
    ProgressEvent,
    VideoEditRequest,
    VideoGenerationRequest,
    new_task_id,
)
from backend.core.interfaces import IConfigStore, IV3TaskStore, TaskStatus
import backend.core.task_kinds as TK
from backend.engine.engine_registry import EngineRegistry
from backend.persistence.asset_store import SQLiteAssetStore


class TaskScheduler:
    def __init__(
        self,
        *,
        path_resolver,
        task_store: IV3TaskStore,
        asset_store: SQLiteAssetStore,
        engine_registry: EngineRegistry,
        config_store: Optional[IConfigStore] = None,
    ):
        self._paths = path_resolver
        self._tasks = task_store
        self._assets = asset_store
        self._engines = engine_registry
        self._config = config_store
        self._pq: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self._seq = itertools.count()
        self._heap_lock = asyncio.Lock()
        self._in_flight: set[str] = set()
        self._tokens: dict[str, CancelToken] = {}
        self._worker: Optional[asyncio.Task] = None
        self._shutdown = False
        self._durations: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=24))
        self._progress_meta: dict[str, dict[str, Any]] = {}
        self._realtime_queues: dict[str, asyncio.Queue] = {}

    async def start(self) -> None:
        if self._worker is None or self._worker.done():
            self._shutdown = False
            await self._recover_orphaned_running()
            await self._rebuild_queued_heap()
            self._worker = asyncio.create_task(self._worker_loop())

    def rebind_task_store(self, db_path: Path) -> None:
        """Point task store at migrated workspace DB."""
        self._tasks.rebind(db_path)

    async def rebuild_queued_heap_sync(self) -> None:
        await self._rebuild_queued_heap()

    @staticmethod
    def _paginate_task_rows(
        list_fn,
        *,
        status: str | None = None,
        page_size: int = 500,
    ) -> list[dict[str, Any]]:
        """Load all matching tasks (``list_tasks`` is capped per page)."""
        out: list[dict[str, Any]] = []
        offset = 0
        while True:
            rows = list_fn(
                limit=page_size,
                offset=offset,
                status=status,
            )
            if not rows:
                break
            out.extend(rows)
            if len(rows) < page_size:
                break
            offset += page_size
        return out

    async def _recover_orphaned_running(self) -> None:
        """On restart: mark previously orphaned running tasks as failed (context lost, unrecoverable)."""
        rows = self._paginate_task_rows(
            self._tasks.list_tasks,
            status=TaskStatus.RUNNING.value,
        )
        for row in rows:
            tid = row["id"]
            self._tasks.mark_failed(tid, "Process restarted while task was running")
            self._tasks.append_log(tid, "Task was interrupted by process restart and marked as failed", "error")
            self._progress_meta.pop(tid, None)
            self._tokens.pop(tid, None)
            self._realtime_queues.pop(tid, None)

    async def shutdown(self) -> None:
        self._shutdown = True
        if self._worker:
            self._worker.cancel()
            try:
                await self._worker
            except asyncio.CancelledError:
                pass

    def _work_dir(self, task_id: str) -> Path:
        d = self._paths.get_outputs_dir() / "work" / task_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _queue_band(self, *, kind: str, priority: str) -> int:
        if priority == "high":
            return 0
        if self._config:
            try:
                if self._config.load().queue_image_first:
                    if TK.is_video_kind(kind):
                        return 2
                    return 1
            except Exception:
                pass
        return 1

    def _avg_seconds(self, kind: str) -> float:
        d = self._durations.get(kind)
        if not d:
            return 120.0
        return max(15.0, sum(d) / len(d))

    async def submit(
        self,
        *,
        kind: str,
        model_id: str,
        params: dict[str, Any],
        priority: str = "normal",
    ) -> dict[str, Any]:
        if kind not in TK.ALL_KINDS:
            raise ValueError(f"unknown task kind: {kind}")
        tid = new_task_id()
        tok = CancelToken()
        self._tokens[tid] = tok
        band = self._queue_band(kind=kind, priority=priority)
        self._tasks.insert_task(
            tid,
            kind,
            model_id,
            params,
            priority=50 if priority == "high" else 100,
            status=TaskStatus.QUEUED,
        )
        async with self._heap_lock:
            await self._pq.put((band, next(self._seq), tid))
        return {
            "id": tid,
            "kind": kind,
            "status": "queued",
            "queue_position": self._pq.qsize(),
            "links": {
                "self": f"/api/tasks/{tid}",
                "stream": f"/api/tasks/{tid}/stream",
                "cancel": f"/api/tasks/{tid}",
                "patch": f"/api/tasks/{tid}",
            },
        }

    async def cancel(self, task_id: str) -> bool:
        tok = self._tokens.get(task_id)
        if tok:
            tok.cancel()
        row = self._tasks.get_task(task_id)
        if not row:
            return False
        st = row["status"]
        if st in ("completed", "failed", "cancelled"):
            return False
        if st == "queued":
            self._tasks.mark_cancelled(task_id)
            await self._rebuild_queued_heap()
            return True
        if st == "running":
            k = row.get("kind") or ""
            mid = row.get("model_id") or ""
            if TK.is_image_kind(k):
                await self._engines.get_image(mid).cancel(task_id)
            elif TK.is_video_kind(k):
                await self._engines.get_video(mid).cancel(task_id)
            elif TK.is_audio_kind(k):
                await self._engines.get_audio(mid).cancel(task_id)
            self._tasks.mark_cancelled(task_id)
            return True
        return False

    def get_task(self, task_id: str) -> Optional[dict[str, Any]]:
        return self._tasks.get_task(task_id)

    def list_tasks(
        self,
        limit: int = 200,
        offset: int = 0,
        *,
        kind: Optional[str] = None,
        status: Optional[str] = None,
        since: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        return self._tasks.list_tasks(
            limit=limit, offset=offset, kind=kind, status=status, since=since
        )

    def get_task_logs(self, task_id: str, offset: int = 0, limit: int = 500) -> list[dict[str, Any]]:
        return self._tasks.get_logs(task_id, offset=offset, limit=limit)

    def get_progress_meta(self, task_id: str) -> dict[str, Any]:
        return dict(self._progress_meta.get(task_id) or {})

    def get_realtime_queue(self, task_id: str) -> asyncio.Queue | None:
        return self._realtime_queues.get(task_id)

    def queue_index_maps(self) -> tuple[dict[str, int], dict[str, Optional[int]]]:
        snap = self.queue_snapshot()
        pos: dict[str, int] = {}
        wait: dict[str, Optional[int]] = {}
        for i, q in enumerate(snap.get("queued") or [], start=1):
            tid = q.get("id")
            if not tid:
                continue
            pos[tid] = i
            wait[tid] = q.get("estimated_wait_seconds")
        return pos, wait

    def public_task_view(
        self,
        row: dict[str, Any],
        *,
        index_maps: Optional[tuple[dict[str, int], dict[str, Optional[int]]]] = None,
    ) -> dict[str, Any]:
        """Plan §6.3 task object: queue position, wait/remaining ETA, model summary, error mirror."""
        out = dict(row)
        tid = out.get("id") or ""
        st = out.get("status") or ""
        if index_maps is None:
            index_maps = self.queue_index_maps()
        pos_map, wait_map = index_maps
        if st == TaskStatus.QUEUED.value:
            qp = pos_map.get(tid)
            if qp is not None:
                out["queue_position"] = qp
            ew = wait_map.get(tid)
            if ew is not None:
                out["estimated_wait_seconds"] = ew
        elif st == TaskStatus.RUNNING.value:
            rk = out.get("kind") or TK.IMAGE_GENERATION
            avg = self._avg_seconds(rk)
            prog = float(out.get("progress") or 0.0)
            out["estimated_remaining_seconds"] = int(avg * max(0.05, 1.0 - min(1.0, prog)))
        mid = out.get("model_id") or ""
        out["model"] = {"id": mid, "name": mid}
        em = (out.get("error_message") or "").strip()
        out["error"] = em if em else None
        meta = self.get_progress_meta(tid)
        if meta:
            out["progress_detail"] = meta
        return out

    def _queued_row_sort_key(self, r: dict[str, Any]) -> tuple[Any, ...]:
        pr = "high" if (r.get("priority") or 100) <= 50 else "normal"
        band = self._queue_band(kind=r["kind"], priority=pr)
        return (band, r.get("priority") or 100, r.get("created_at") or "")

    async def _rebuild_queued_heap(self) -> None:
        """Rebuild in-memory heap based on DB ``queued`` entries; exclude already-dequeued but not-yet-finished tids (``_in_flight``)."""
        async with self._heap_lock:
            while True:
                try:
                    self._pq.get_nowait()
                except asyncio.QueueEmpty:
                    break
            rows = self._paginate_task_rows(
                self._tasks.list_tasks,
                status=TaskStatus.QUEUED.value,
            )
            rows = [r for r in rows if r["id"] not in self._in_flight]

            rows.sort(key=self._queued_row_sort_key)
            for r in rows:
                pr = "high" if (r.get("priority") or 100) <= 50 else "normal"
                band = self._queue_band(kind=r["kind"], priority=pr)
                await self._pq.put((band, next(self._seq), r["id"]))

    async def update_queued_priority(
        self, task_id: str, priority: str
    ) -> Literal["ok", "not_found", "not_queued", "noop"]:
        row = self._tasks.get_task(task_id)
        if not row:
            return "not_found"
        if row["status"] != TaskStatus.QUEUED.value:
            return "not_queued"
        pri_int = 50 if priority == "high" else 100
        if (row.get("priority") or 100) == pri_int:
            return "noop"
        if not self._tasks.update_task_priority(task_id, pri_int):
            return "not_queued"
        await self._rebuild_queued_heap()
        return "ok"

    def queue_snapshot(self) -> dict[str, Any]:
        rows = self._tasks.list_tasks(500, 0)
        running = [r for r in rows if r["status"] == TaskStatus.RUNNING.value]
        queued_all = [r for r in rows if r["status"] == TaskStatus.QUEUED.value]
        queued = sorted(queued_all, key=self._queued_row_sort_key)[:50]

        cum = 0.0
        if running:
            rk = running[0].get("kind") or TK.IMAGE_GENERATION
            prog = float(running[0].get("progress") or 0.0)
            cum += self._avg_seconds(rk) * max(0.05, 1.0 - min(1.0, prog))

        enriched: list[dict[str, Any]] = []
        for r in queued:
            cum += self._avg_seconds(r.get("kind") or TK.IMAGE_GENERATION)
            d = dict(r)
            d["estimated_wait_seconds"] = int(cum)
            enriched.append(d)

        return {
            "running": running[:5],
            "queued": enriched,
            "counts": {"queued": len(queued_all), "running": len(running)},
        }

    async def _worker_loop(self) -> None:
        while not self._shutdown:
            try:
                _, __, tid = await self._pq.get()
            except asyncio.CancelledError:
                break
            self._in_flight.add(tid)
            try:
                row = self._tasks.get_task(tid)
                if not row or row["status"] == TaskStatus.CANCELLED.value:
                    continue
                await self._execute(tid)
            finally:
                self._in_flight.discard(tid)
                self._pq.task_done()

    def _record_duration(self, tid: str, kind: str) -> None:
        row = self._tasks.get_task(tid)
        if not row or not row.get("started_at"):
            return
        try:
            st = datetime.fromisoformat(row["started_at"])
            sec = (datetime.now() - st).total_seconds()
            if 1.0 < sec < 6 * 3600:
                self._durations[kind].append(sec)
        except Exception:
            pass

    async def _execute(self, tid: str) -> None:
        row = self._tasks.get_task(tid)
        if not row or row["status"] != TaskStatus.QUEUED.value:
            return
        tok = self._tokens.setdefault(tid, CancelToken())
        kind = row["kind"]
        params = row["params"]
        model_id = row["model_id"]
        loop = asyncio.get_running_loop()
        rt_queue: asyncio.Queue = asyncio.Queue(maxsize=200)
        self._realtime_queues[tid] = rt_queue

        def _put_rt(event_type: str, data: Any) -> None:
            q = self._realtime_queues.get(tid)
            if not q:
                return
            try:
                q.put_nowait((event_type, data))
            except asyncio.QueueFull:
                pass

        def on_progress(ev: ProgressEvent) -> None:
            self._tasks.update_progress(tid, ev.progress)
            self._progress_meta[tid] = {
                "step": ev.step,
                "total": ev.total,
                "eta_seconds": ev.eta_seconds,
                "message": ev.message,
            }
            loop.call_soon_threadsafe(_put_rt, "progress", ev)

        def on_log(ev: LogEvent) -> None:
            # Persist only — ``GET /api/tasks/{id}/stream`` replays from DB (step 1).
            # Pushing the same log onto ``rt_queue`` duplicated every line (DB + queue).
            self._tasks.append_log(tid, ev.message, ev.level)

        ctx = ExecutionContext(
            task_id=tid,
            cancel_token=tok,
            on_progress=on_progress,
            on_log=on_log,
            work_dir=self._work_dir(tid),
            asset_store=self._assets,
        )

        self._tasks.mark_running(tid)
        try:
            if tok.is_cancelled():
                self._tasks.mark_cancelled(tid)
                return
            if kind == TK.IMAGE_GENERATION:
                req = ImageGenerationRequest.model_validate(params)
                res = await self._engines.get_image(model_id).generate(req, ctx)
            elif kind == TK.IMAGE_EDIT:
                req = ImageEditRequest.model_validate(params)
                res = await self._engines.get_image(model_id).edit(req, ctx)
            elif kind == TK.IMAGE_UPSCALE:
                req = ImageUpscaleRequest.model_validate(params)
                res = await self._engines.get_image(model_id).upscale(req, ctx)
            elif kind == TK.VIDEO_GENERATION:
                req = VideoGenerationRequest.model_validate(params)
                res = await self._engines.get_video(model_id).generate(req, ctx)
            elif kind == TK.VIDEO_EDIT:
                req = VideoEditRequest.model_validate(params)
                res = await self._engines.get_video(model_id).edit(req, ctx)
            elif kind == TK.AUDIO_GENERATION:
                req = AudioGenerationRequest.model_validate(params)
                res = await self._engines.get_audio(model_id).generate(req, ctx)
            elif kind == TK.AUDIO_EDIT:
                req = AudioEditRequest.model_validate(params)
                res = await self._engines.get_audio(model_id).edit(req, ctx)
            else:
                raise RuntimeError(f"unknown kind {kind}")

            self._tasks.mark_completed(
                tid,
                {
                    "primary_asset_id": res.primary_asset_id,
                    "asset_ids": res.asset_ids,
                    "output_paths": res.output_paths,
                    "metadata": res.metadata,
                },
            )
            self._record_duration(tid, kind)
        except asyncio.CancelledError:
            self._tasks.mark_cancelled(tid)
        except Exception as e:
            self._tasks.mark_failed(tid, str(e))
            self._tasks.append_log(tid, str(e), "error")
        finally:
            self._progress_meta.pop(tid, None)
            self._tokens.pop(tid, None)
            self._realtime_queues.pop(tid, None)
