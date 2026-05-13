"""POST /api/audios/* — plan audios.py"""

from fastapi import APIRouter, Depends, HTTPException

from backend.api.deps import get_engine_registry, get_task_scheduler
from backend.core import task_kinds as TK
from backend.core.contracts import AudioEditRequest, AudioGenerationRequest
from backend.engine.engine_registry import EngineRegistry
from backend.scheduler.task_scheduler import TaskScheduler

router = APIRouter(prefix="/api/audios", tags=["audios"])


@router.post("/generations", status_code=202)
async def post_audio_generation(
    body: AudioGenerationRequest,
    sched: TaskScheduler = Depends(get_task_scheduler),
    engines: EngineRegistry = Depends(get_engine_registry),
):
    eng = engines.get_audio(body.model)
    if not eng.supports(body.model, "create_music"):
        raise HTTPException(409, detail={"code": "unsupported", "message": "model does not support create_music"})
    r = await sched.submit(
        kind=TK.AUDIO_GENERATION,
        model_id=body.model,
        params=body.model_dump(),
        priority=body.priority,
    )
    return {"task": r}


@router.post("/edits", status_code=202)
async def post_audio_edit(
    body: AudioEditRequest,
    sched: TaskScheduler = Depends(get_task_scheduler),
    engines: EngineRegistry = Depends(get_engine_registry),
):
    eng = engines.get_audio(body.model)
    if not eng.supports(body.model, "edit"):
        raise HTTPException(409, detail={"code": "unsupported", "message": "model does not support audio edit"})
    r = await sched.submit(
        kind=TK.AUDIO_EDIT,
        model_id=body.model,
        params=body.model_dump(),
        priority=body.priority,
    )
    return {"task": r}
