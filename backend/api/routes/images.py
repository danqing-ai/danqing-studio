"""POST /api/images/* — plan images.py"""

from fastapi import APIRouter, Depends, HTTPException

from backend.api.deps import get_engine_registry, get_task_scheduler
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
    eng = engines.get_image(body.model)
    if not eng.supports(body.model, "generate"):
        raise HTTPException(409, detail={"code": "unsupported", "message": "model does not support generate"})
    r = await sched.submit(
        kind=TK.IMAGE_GENERATION,
        model_id=body.model,
        params=body.model_dump(),
        priority=body.priority,
    )
    return {"task": r}


@router.post("/edits", status_code=202)
async def post_image_edit(
    body: ImageEditRequest,
    sched: TaskScheduler = Depends(get_task_scheduler),
    engines: EngineRegistry = Depends(get_engine_registry),
):
    eng = engines.get_image(body.model)
    if not eng.supports(body.model, "edit"):
        raise HTTPException(409, detail={"code": "unsupported", "message": "model does not support edit"})
    r = await sched.submit(
        kind=TK.IMAGE_EDIT,
        model_id=body.model,
        params=body.model_dump(),
        priority=body.priority,
    )
    return {"task": r}


@router.post("/upscales", status_code=202)
async def post_image_upscale(
    body: ImageUpscaleRequest,
    sched: TaskScheduler = Depends(get_task_scheduler),
    engines: EngineRegistry = Depends(get_engine_registry),
):
    eng = engines.get_image(body.model)
    if not eng.supports(body.model, "upscale"):
        raise HTTPException(409, detail={"code": "unsupported", "message": "model does not support upscale"})
    r = await sched.submit(
        kind=TK.IMAGE_UPSCALE,
        model_id=body.model,
        params=body.model_dump(),
        priority=body.priority,
    )
    return {"task": r}
