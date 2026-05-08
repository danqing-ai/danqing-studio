"""POST /api/videos/* — plan videos.py"""

from fastapi import APIRouter, Depends, HTTPException

from backend.api.deps import get_engine_registry, get_task_scheduler
from backend.core import task_kinds as TK
from backend.core.contracts import VideoEditRequest, VideoGenerationRequest
from backend.engine.engine_registry import EngineRegistry
from backend.scheduler.task_scheduler import TaskScheduler

router = APIRouter(prefix="/api/videos", tags=["videos"])


@router.post("/generations", status_code=202)
async def post_video_generation(
    body: VideoGenerationRequest,
    sched: TaskScheduler = Depends(get_task_scheduler),
    engines: EngineRegistry = Depends(get_engine_registry),
):
    eng = engines.get_video(body.model)
    if not eng.supports(body.model, "generate"):
        raise HTTPException(409, detail={"code": "unsupported", "message": "model does not support video generate"})
    r = await sched.submit(
        kind=TK.VIDEO_GENERATION,
        model_id=body.model,
        params=body.model_dump(),
        priority=body.priority,
    )
    return {"task": r}


@router.post("/edits", status_code=202)
async def post_video_edit(
    body: VideoEditRequest,
    sched: TaskScheduler = Depends(get_task_scheduler),
    engines: EngineRegistry = Depends(get_engine_registry),
):
    eng = engines.get_video(body.model)
    if not eng.supports(body.model, "edit"):
        raise HTTPException(409, detail={"code": "unsupported", "message": "model does not support video edit"})
    r = await sched.submit(
        kind=TK.VIDEO_EDIT,
        model_id=body.model,
        params=body.model_dump(),
        priority=body.priority,
    )
    return {"task": r}
