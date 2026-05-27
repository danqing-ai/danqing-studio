"""POST /api/images/* — plan images.py"""

from fastapi import APIRouter, Depends

from backend.api.deps import get_engine_registry, get_task_scheduler
from backend.api.routes.submit_helpers import submit_media_task
from backend.core import task_kinds as TK
from backend.core.contracts import ImageEditRequest, ImageGenerationRequest, ImageUpscaleRequest
from backend.engine.engine_registry import EngineRegistry
from backend.scheduler.task_scheduler import TaskScheduler

router = APIRouter(prefix="/api/images", tags=["images"])


@router.post("/generations", status_code=202)
async def post_image_generation(
    body: ImageGenerationRequest,
    sched: TaskScheduler = Depends(get_task_scheduler),
    engines: EngineRegistry = Depends(get_engine_registry),
):
    return await submit_media_task(
        body=body,
        media="image",
        api_action="generate",
        task_kind=TK.IMAGE_GENERATION,
        sched=sched,
        engines=engines,
    )


@router.post("/edits", status_code=202)
async def post_image_edit(
    body: ImageEditRequest,
    sched: TaskScheduler = Depends(get_task_scheduler),
    engines: EngineRegistry = Depends(get_engine_registry),
):
    return await submit_media_task(
        body=body,
        media="image",
        api_action="edit",
        task_kind=TK.IMAGE_EDIT,
        sched=sched,
        engines=engines,
    )


@router.post("/upscales", status_code=202)
async def post_image_upscale(
    body: ImageUpscaleRequest,
    sched: TaskScheduler = Depends(get_task_scheduler),
    engines: EngineRegistry = Depends(get_engine_registry),
):
    return await submit_media_task(
        body=body,
        media="image",
        api_action="upscale",
        task_kind=TK.IMAGE_UPSCALE,
        sched=sched,
        engines=engines,
    )
