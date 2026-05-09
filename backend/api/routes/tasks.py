"""GET/DELETE/SSE /api/tasks/* — plan tasks.py (§6.2)"""

import asyncio
import json
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from backend.api.deps import get_task_scheduler
from backend.scheduler.task_scheduler import TaskScheduler

router = APIRouter(prefix="/api/tasks", tags=["tasks"])


class TaskPriorityPatch(BaseModel):
    priority: Literal["normal", "high"]


async def _list_tasks_query(
    limit: int,
    offset: int,
    kind: str | None,
    status: str | None,
    since: str | None,
    sched: TaskScheduler,
):
    rows = sched.list_tasks(limit=limit, offset=offset, kind=kind, status=status, since=since)
    idx = sched.queue_index_maps()
    return {"tasks": [sched.public_task_view(r, index_maps=idx) for r in rows]}


@router.get("")
async def list_tasks_root(
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    kind: str | None = Query(
        None,
        description="Exact kind (e.g. image.generation) or family prefix without dot (image → image.*)",
    ),
    status: str | None = Query(None, description="queued | running | completed | failed | cancelled"),
    since: str | None = Query(None, description="ISO8601 lower bound on created_at (inclusive)"),
    sched: TaskScheduler = Depends(get_task_scheduler),
):
    """Plan：``GET /api/tasks`` — 与 ``/list`` 等价。"""
    return await _list_tasks_query(limit, offset, kind, status, since, sched)


@router.get("/list")
async def list_tasks(
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    kind: str | None = Query(
        None,
        description="Exact kind (e.g. image.generation) or family prefix without dot (image → image.*)",
    ),
    status: str | None = Query(None, description="queued | running | completed | failed | cancelled"),
    since: str | None = Query(None, description="ISO8601 lower bound on created_at (inclusive)"),
    sched: TaskScheduler = Depends(get_task_scheduler),
):
    return await _list_tasks_query(limit, offset, kind, status, since, sched)


@router.get("/{task_id}")
async def get_task(task_id: str, sched: TaskScheduler = Depends(get_task_scheduler)):
    row = sched.get_task(task_id)
    if not row:
        raise HTTPException(404, "task not found")
    return sched.public_task_view(row)


@router.patch("/{task_id}")
async def patch_task_priority(
    task_id: str,
    body: TaskPriorityPatch,
    sched: TaskScheduler = Depends(get_task_scheduler),
):
    out = await sched.update_queued_priority(task_id, body.priority)
    if out == "not_found":
        raise HTTPException(404, "task not found")
    if out == "not_queued":
        raise HTTPException(409, "only queued tasks can change priority")
    row = sched.get_task(task_id)
    if not row:
        raise HTTPException(404, "task not found")
    return sched.public_task_view(row)


@router.get("/{task_id}/logs")
async def get_task_logs(
    task_id: str,
    offset: int = Query(0, ge=0),
    limit: int = Query(500, ge=1, le=2000),
    sched: TaskScheduler = Depends(get_task_scheduler),
):
    if not sched.get_task(task_id):
        raise HTTPException(404, "task not found")
    return {"items": sched.get_task_logs(task_id, offset=offset, limit=limit)}


@router.delete("/{task_id}")
async def delete_task(task_id: str, sched: TaskScheduler = Depends(get_task_scheduler)):
    ok = await sched.cancel(task_id)
    return {"ok": ok}


@router.get("/{task_id}/stream")
async def stream_task(task_id: str, sched: TaskScheduler = Depends(get_task_scheduler)):
    async def gen():
        last_log = 0
        last_progress_key: tuple[Any, ...] | None = None
        while True:
            row = sched.get_task(task_id)
            if not row:
                yield f"event: done\ndata: {json.dumps({'status': 'not_found'})}\n\n"
                break
            st = row["status"]
            # 1. flush DB logs
            for log in sched.get_task_logs(task_id, offset=last_log, limit=200):
                last_log += 1
                payload = {
                    "message": log["message"],
                    "level": log["level"],
                    "ts": log.get("time"),
                }
                yield f"event: log\ndata: {json.dumps(payload)}\n\n"
            # 2. flush realtime queue events (progress + logs emitted from worker thread)
            rt_queue = sched.get_realtime_queue(task_id)
            if rt_queue:
                try:
                    while True:
                        ev_type, ev_data = rt_queue.get_nowait()
                        if ev_type == "log" and hasattr(ev_data, "message"):
                            payload = {
                                "message": ev_data.message,
                                "level": ev_data.level,
                                "ts": None,
                            }
                            yield f"event: log\ndata: {json.dumps(payload)}\n\n"
                        elif ev_type == "progress" and hasattr(ev_data, "progress"):
                            prog = float(ev_data.progress or 0.0)
                            pkey = (prog, ev_data.step, ev_data.total, ev_data.eta_seconds)
                            if pkey != last_progress_key:
                                last_progress_key = pkey
                                yield (
                                    "event: progress\ndata: "
                                    + json.dumps(
                                        {
                                            "progress": prog,
                                            "step": ev_data.step,
                                            "total": ev_data.total,
                                            "eta_seconds": ev_data.eta_seconds,
                                        }
                                    )
                                    + "\n\n"
                                )
                except asyncio.QueueEmpty:
                    pass
            # 3. status
            meta = sched.get_progress_meta(task_id)
            prog = float(row.get("progress") or 0.0)
            pkey = (prog, meta.get("step"), meta.get("total"), meta.get("eta_seconds"))
            if pkey != last_progress_key:
                last_progress_key = pkey
                yield (
                    "event: progress\ndata: "
                    + json.dumps(
                        {
                            "progress": prog,
                            "step": meta.get("step"),
                            "total": meta.get("total"),
                            "eta_seconds": meta.get("eta_seconds"),
                        }
                    )
                    + "\n\n"
                )
            status_payload: dict[str, Any] = {
                "status": st,
                "progress": prog,
                "started_at": row.get("started_at"),
            }
            yield f"event: status\ndata: {json.dumps(status_payload)}\n\n"
            if st in ("completed", "failed", "cancelled"):
                if st == "completed" and row.get("result"):
                    yield f"event: result\ndata: {json.dumps(row['result'])}\n\n"
                yield f"event: done\ndata: {json.dumps({'status': st})}\n\n"
                break
            # 4. wait: event-driven when realtime queue exists, else short poll
            if rt_queue:
                try:
                    ev_type, ev_data = await asyncio.wait_for(rt_queue.get(), timeout=0.3)
                    # put back so next loop processes it
                    await rt_queue.put((ev_type, ev_data))
                except asyncio.TimeoutError:
                    pass
            else:
                await asyncio.sleep(0.1)

    return StreamingResponse(gen(), media_type="text/event-stream")
