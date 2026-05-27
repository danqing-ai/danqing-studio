"""POST /api/audios/* — plan audios.py"""

from fastapi import APIRouter, Depends

from backend.api.deps import get_engine_registry, get_task_scheduler
from backend.api.routes.submit_helpers import submit_media_task
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
    return await submit_media_task(
        body=body,
        media="audio",
        api_action="create_music",
        task_kind=TK.AUDIO_GENERATION,
        sched=sched,
        engines=engines,
    )


@router.post("/edits", status_code=202)
async def post_audio_edit(
    body: AudioEditRequest,
    sched: TaskScheduler = Depends(get_task_scheduler),
    engines: EngineRegistry = Depends(get_engine_registry),
):
    return await submit_media_task(
        body=body,
        media="audio",
        api_action="edit",
        task_kind=TK.AUDIO_EDIT,
        sched=sched,
        engines=engines,
    )
