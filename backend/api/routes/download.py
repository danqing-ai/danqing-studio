"""
API routes - download related
Supports model download, LoRA download, CivitAI search, progress SSE
"""

import asyncio
import uuid
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional

from backend.core.container import get_container
from backend.core.interfaces import IDownloadService, ISettingsService, DownloadProgress, ConversionTask
from backend.core.i18n import t, resolve_locale
from backend.third_party.civitai_client import CivitAIClient

router = APIRouter(prefix="/api/download", tags=["download"])


class DownloadModelRequest(BaseModel):
    model_name: Optional[str] = None
    version: Optional[str] = None


class DownloadLoraRequest(BaseModel):
    url: str
    filename: str


class DownloadProgressResponse(BaseModel):
    task_id: str
    status: str
    progress: float
    total_size: int
    downloaded_size: int
    speed: str
    error_message: str
    filename: str


class CivitAIModelResponse(BaseModel):
    id: int
    name: str
    description: str
    type: str
    nsfw: bool
    tags: List[str]
    creator: dict
    stats: dict
    model_versions: List[dict]


def get_download_service():
    return get_container().resolve(IDownloadService)


def get_settings_service():
    return get_container().resolve(ISettingsService)


async def start_model_install(model_name: str, *, version: Optional[str] = None) -> DownloadProgressResponse:
    """Start HuggingFace model file download; progress via GET /api/download/progress/{task_id}/stream."""
    service = get_download_service()
    progress_queue: asyncio.Queue = asyncio.Queue()

    async def on_progress(progress: DownloadProgress):
        await progress_queue.put(progress)

    async def do_download():
        try:
            await service.download_model(model_name, version=version, progress_callback=on_progress)
        except Exception as e:
            await progress_queue.put(
                DownloadProgress(
                    task_id="",
                    status="failed",
                    progress=0,
                    error_message=str(e),
                    filename=model_name,
                )
            )

    asyncio.create_task(do_download())
    first_progress = await progress_queue.get()
    task_id = first_progress.task_id
    await progress_queue.put(first_progress)
    return DownloadProgressResponse(
        task_id=task_id,
        status="running",
        progress=0,
        total_size=0,
        downloaded_size=0,
        speed="",
        error_message="",
        filename=model_name,
    )


@router.post("/lora", response_model=DownloadProgressResponse)
async def download_lora(request: DownloadLoraRequest):
    """Download LoRA file"""
    service = get_download_service()

    progress_queue: asyncio.Queue = asyncio.Queue()

    async def on_progress(progress: DownloadProgress):
        await progress_queue.put(progress)

    async def do_download():
        try:
            await service.download_lora(request.url, request.filename, progress_callback=on_progress)
        except Exception as e:
            await progress_queue.put(DownloadProgress(
                task_id="",
                status="failed",
                progress=0,
                error_message=str(e),
                filename=request.filename
            ))

    task = asyncio.create_task(do_download())

    first_progress = await progress_queue.get()
    task_id = first_progress.task_id
    await progress_queue.put(first_progress)

    return DownloadProgressResponse(
        task_id=task_id,
        status="running",
        progress=0,
        total_size=0,
        downloaded_size=0,
        speed="",
        error_message="",
        filename=request.filename
    )


@router.get("/progress/{task_id}/stream")
async def stream_progress(task_id: str):
    """SSE streaming download progress"""
    service = get_download_service()

    async def event_generator():
        while True:
            progress = service.get_progress(task_id)
            if progress:
                import json
                data = json.dumps({
                    'task_id': progress.task_id,
                    'status': progress.status,
                    'progress': progress.progress,
                    'total_size': progress.total_size,
                    'downloaded_size': progress.downloaded_size,
                    'speed': progress.speed,
                    'error_message': progress.error_message,
                    'filename': progress.filename
                })
                yield f"data: {data}\n\n"

                if progress.status in ("completed", "failed", "cancelled"):
                    break

            await asyncio.sleep(0.5)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )


@router.get("/progress/{task_id}", response_model=DownloadProgressResponse)
async def get_progress(task_id: str, req: Request = None):
    locale = resolve_locale(req.headers.get("Accept-Language")) if req else "zh"
    service = get_download_service()
    progress = service.get_progress(task_id)

    if not progress:
        return DownloadProgressResponse(
            task_id=task_id,
            status="unknown",
            progress=0,
            total_size=0,
            downloaded_size=0,
            speed="",
            error_message=t("error.download_task_not_found", locale),
            filename=""
        )

    return DownloadProgressResponse(
        task_id=progress.task_id,
        status=progress.status,
        progress=progress.progress,
        total_size=progress.total_size,
        downloaded_size=progress.downloaded_size,
        speed=progress.speed,
        error_message=progress.error_message,
        filename=progress.filename
    )


@router.post("/cancel/{task_id}")
async def cancel_download(task_id: str):
    """Cancel download task"""
    service = get_download_service()
    success = await service.cancel_download(task_id)
    return {"success": success}


@router.delete("/tasks/{task_id}")
async def delete_download(task_id: str):
    """Delete download task"""
    service = get_download_service()
    success = service.delete_download(task_id)
    return {"success": success}


@router.post("/resume/{task_id}", response_model=DownloadProgressResponse)
async def resume_download(task_id: str):
    """Resume download task (after process restart)"""
    service = get_download_service()

    progress_queue: asyncio.Queue = asyncio.Queue()

    async def on_progress(progress: DownloadProgress):
        await progress_queue.put(progress)

    async def do_resume():
        try:
            await service.resume_download(task_id, progress_callback=on_progress)
        except Exception as e:
            await progress_queue.put(DownloadProgress(
                task_id=task_id,
                status="failed",
                progress=0,
                error_message=str(e),
                filename=""
            ))

    asyncio.create_task(do_resume())

    # Wait for first real progress (max 10 seconds)
    try:
        first_progress = await asyncio.wait_for(progress_queue.get(), timeout=10.0)
    except asyncio.TimeoutError:
        first_progress = DownloadProgress(
            task_id=task_id,
            status="running",
            progress=0,
            total_size=0,
            downloaded_size=0,
            speed="",
            error_message="",
            filename=""
        )

    return DownloadProgressResponse(
        task_id=first_progress.task_id,
        status=first_progress.status,
        progress=first_progress.progress,
        total_size=first_progress.total_size,
        downloaded_size=first_progress.downloaded_size,
        speed=first_progress.speed,
        error_message=first_progress.error_message,
        filename=first_progress.filename
    )


@router.get("/tasks")
async def list_downloads():
    """List all download tasks"""
    service = get_download_service()
    tasks = service.list_downloads()
    result = []
    for t in tasks:
        progress = service.get_progress(t.id)
        result.append({
            "id": t.id,
            "url": t.url,
            "target_path": t.target_path,
            "status": t.status.value,
            "progress": t.progress,
            "total_size": progress.total_size if progress else 0,
            "downloaded_size": progress.downloaded_size if progress else 0,
            "filename": progress.filename if progress else t.url,
            "error_message": progress.error_message if progress else t.error_message
        })
    return result


# ===== CivitAI search routes =====

class CivitAISearchResponse(BaseModel):
    items: List[CivitAIModelResponse]
    next_cursor: Optional[str] = None
    prev_cursor: Optional[str] = None


@router.get("/civitai/search", response_model=CivitAISearchResponse)
async def search_civitai(
    q: str = Query("", description="Search keyword"),
    types: str = Query("LORA", description="Model type, comma-separated e.g. LORA,Checkpoint"),
    limit: int = Query(20, ge=1, le=100),
    page: int = Query(1, ge=1),
    cursor: Optional[str] = Query(None, description="Pagination cursor (used during search)"),
    sort: str = Query("Highest Rated", description="Sort method")
):
    """Search CivitAI models (no NSFW filtering)"""
    settings = get_settings_service().get_settings()
    client = CivitAIClient(api_key=settings.civitai_token or None)

    try:
        type_list = [t.strip() for t in types.split(",")] if types else ["LORA"]
        result = await client.search(
            query=q,
            types=type_list,
            limit=limit,
            page=page,
            cursor=cursor,
            sort=sort,
            nsfw=True if (settings.civitai_token and settings.nsfw_enabled) else None
        )

        models = result["items"]
        metadata = result["metadata"]

        return CivitAISearchResponse(
            items=[
                CivitAIModelResponse(
                    id=m.id,
                    name=m.name,
                    description=m.description,
                    type=m.type,
                    nsfw=m.nsfw,
                    tags=m.tags,
                    creator=m.creator,
                    stats=m.stats,
                    model_versions=[
                        {
                            "id": v.id,
                            "name": v.name,
                            "base_model": v.base_model,
                            "download_url": v.download_url,
                            "trained_words": v.trained_words,
                            "files": [
                                {
                                    "name": f.name,
                                    "download_url": f.download_url,
                                    "size_kb": f.size_kb,
                                    "format": f.format,
                                    "primary": f.primary
                                }
                                for f in v.files
                            ]
                        }
                        for v in m.model_versions
                    ]
                )
                for m in models
            ],
            next_cursor=metadata.get("nextCursor"),
            prev_cursor=metadata.get("prevCursor")
        )
    finally:
        await client.close()


@router.get("/civitai/model/{model_id}", response_model=CivitAIModelResponse)
async def get_civitai_model(model_id: int):
    """Get CivitAI model details"""
    settings = get_settings_service().get_settings()
    client = CivitAIClient(api_key=settings.civitai_token or None)

    try:
        model = await client.get_model(model_id)
        return CivitAIModelResponse(
            id=model.id,
            name=model.name,
            description=model.description,
            type=model.type,
            nsfw=model.nsfw,
            tags=model.tags,
            creator=model.creator,
            stats=model.stats,
            model_versions=[
                {
                    "id": v.id,
                    "name": v.name,
                    "base_model": v.base_model,
                    "download_url": v.download_url,
                    "trained_words": v.trained_words,
                    "files": [
                        {
                            "name": f.name,
                            "download_url": f.download_url,
                            "size_kb": f.size_kb,
                            "format": f.format,
                            "primary": f.primary
                        }
                        for f in v.files
                    ]
                }
                for v in model.model_versions
            ]
        )
    finally:
        await client.close()


# ===== Model conversion (generate quantized version) =====

class ConvertModelRequest(BaseModel):
    model_name: str
    from_version: str
    to_version: str


class ConversionProgressResponse(BaseModel):
    task_id: str
    status: str
    progress: float
    stage: str
    error_message: str


@router.post("/convert")
async def convert_model(request: ConvertModelRequest, http_request: Request):
    """Start model conversion (int4|int8 MLX quantization of transformer weights)."""
    locale = resolve_locale(http_request.headers.get("Accept-Language"))
    service = get_download_service()
    cfg = service.get_model_download_config(request.model_name)
    if not cfg:
        raise HTTPException(status_code=404, detail=t("error.model_not_in_registry", locale, name=request.model_name))
    versions = cfg.get("versions") or {}
    to_ver = versions.get(request.to_version)
    if not to_ver:
        raise HTTPException(status_code=404, detail=t("error.target_version_not_found", locale, version=request.to_version))

    progress_queue: asyncio.Queue = asyncio.Queue()

    async def on_progress(task: ConversionTask):
        await progress_queue.put(task)

    async def do_conversion():
        try:
            await service.convert_model(
                request.model_name,
                request.from_version,
                request.to_version,
                progress_callback=on_progress
            )
        except Exception as e:
            await progress_queue.put(ConversionTask(
                id="",
                model_name=request.model_name,
                from_version=request.from_version,
                to_version=request.to_version,
                status="failed",
                progress=0,
                stage="error",
                error_message=str(e)
            ))

    task = asyncio.create_task(do_conversion())

    first_progress = await progress_queue.get()
    task_id = first_progress.id or str(uuid.uuid4())
    await progress_queue.put(first_progress)

    return ConversionProgressResponse(
        task_id=task_id,
        status="running",
        progress=0,
        stage="pending",
        error_message=""
    )


@router.get("/convert/{task_id}/stream")
async def stream_conversion_progress(task_id: str):
    """SSE streaming conversion progress"""
    service = get_download_service()

    async def event_generator():
        while True:
            task = service.get_conversion_progress(task_id)
            if task:
                import json
                data = json.dumps({
                    'task_id': task.id,
                    'status': task.status.value if hasattr(task.status, 'value') else task.status,
                    'progress': task.progress,
                    'stage': task.stage,
                    'error_message': task.error_message,
                    'model_name': task.model_name,
                    'from_version': task.from_version,
                    'to_version': task.to_version
                })
                yield f"data: {data}\n\n"

                status_str = task.status.value if hasattr(task.status, 'value') else task.status
                if status_str in ("completed", "failed", "cancelled"):
                    break

            await asyncio.sleep(0.5)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )


@router.post("/convert/{task_id}/cancel")
async def cancel_conversion(task_id: str):
    """Cancel conversion task"""
    service = get_download_service()
    success = await service.cancel_conversion(task_id)
    return {"success": success}


@router.get("/conversions")
async def list_conversions():
    """List all conversion tasks"""
    service = get_download_service()
    tasks = service.list_conversions()
    return [
        {
            "id": t.id,
            "model_name": t.model_name,
            "from_version": t.from_version,
            "to_version": t.to_version,
            "status": t.status.value if hasattr(t.status, 'value') else t.status,
            "progress": t.progress,
            "stage": t.stage,
            "error_message": t.error_message
        }
        for t in tasks
    ]


