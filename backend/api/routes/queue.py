"""GET /api/queue — plan queue.py"""

from fastapi import APIRouter, Depends

from backend.api.deps import get_task_scheduler
from backend.scheduler.task_scheduler import TaskScheduler

router = APIRouter(prefix="/api", tags=["queue"])


@router.get("/queue")
async def get_queue(sched: TaskScheduler = Depends(get_task_scheduler)):
    return sched.queue_snapshot()
