"""Plan models：注册表索引 + 模型安装 / 批量安装 / 按版本删除。"""

import asyncio
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

from backend.api.deps import get_model_registry
from backend.core.container import get_container
from backend.core.interfaces import ISettingsService
from backend.api.routes.download import (
    DownloadModelRequest,
    DownloadProgressResponse,
    get_download_service,
    get_settings_service,
    start_model_install,
)
from backend.core.i18n import resolve_locale, t
from backend.core.interfaces import DownloadProgress
from backend.core.model_registry import ModelRegistry

router = APIRouter(prefix="/api/models", tags=["models"])


def _settings_service() -> ISettingsService:
    return get_container().resolve(ISettingsService)


class BatchInstallRequest(BaseModel):
    """注册表模型 id 列表；逐项启动 HF 下载（与单模型安装相同进度 SSE）。"""

    model_ids: List[str] = Field(..., min_length=1)


@router.get("")
def list_models_index(
    media: Optional[str] = Query(None, description="image | video"),
    action: Optional[str] = Query(None, description="create | rewrite | … 等注册表 action 键"),
    installed: Optional[bool] = Query(None, description="true=至少一版本就绪；false=未就绪"),
    reg: ModelRegistry = Depends(get_model_registry),
):
    detailed = _settings_service().get_models_detailed_status()
    out: dict[str, dict] = {}
    for mid, e in reg.all().items():
        if media and e.media != media:
            continue
        if action and action not in e.actions:
            continue
        st = detailed.get(mid) or {}
        ready = bool(st.get("ready"))
        if installed is True and not ready:
            continue
        if installed is False and ready:
            continue
        out[mid] = {
            "media": e.media,
            "family": e.family,
            "engine": e.engine,
            "actions": sorted(e.actions),
            "installed": ready,
        }
    return {"models": out}


@router.post("/install-batch")
async def install_registry_models_batch(
    body: BatchInstallRequest,
    http_request: Request,
    reg: ModelRegistry = Depends(get_model_registry),
):
    """批量启动注册表模型下载；响应项与旧 `/api/download/batch` 结构一致（`results[].model_name` / `task_id`）。"""
    locale = resolve_locale(http_request.headers.get("Accept-Language"))
    service = get_download_service()
    settings = get_settings_service()
    detailed = settings.get_models_detailed_status()
    results: list[dict] = []

    for model_id in body.model_ids:
        if reg.get(model_id) is None:
            results.append({"model_name": model_id, "status": "failed", "error": "model not found in registry"})
            continue
        try:
            config = service.get_model_download_config(model_id)
            if config and config.get("dependencies"):
                missing_deps: list[str] = []
                for dep in config["dependencies"]:
                    if not detailed.get(dep, {}).get("ready"):
                        missing_deps.append(dep)
                if missing_deps:
                    results.append(
                        {
                            "model_name": model_id,
                            "status": "skipped",
                            "reason": t("error.missing_dependencies", locale, deps=", ".join(missing_deps)),
                        }
                    )
                    continue

            progress_queue: asyncio.Queue = asyncio.Queue()

            async def on_progress(progress: DownloadProgress):
                await progress_queue.put(progress)

            asyncio.create_task(service.download_model(model_id, progress_callback=on_progress))
            first_progress = await asyncio.wait_for(progress_queue.get(), timeout=5.0)
            results.append(
                {
                    "model_name": model_id,
                    "status": "started",
                    "task_id": first_progress.task_id,
                }
            )
        except Exception as e:
            results.append({"model_name": model_id, "status": "failed", "error": str(e)})

    return {"results": results}


@router.post("/{model_id}/install", response_model=DownloadProgressResponse)
async def install_registry_model(
    model_id: str,
    request: Optional[DownloadModelRequest] = None,
    reg: ModelRegistry = Depends(get_model_registry),
):
    """安装注册表中的模型权重；进度 SSE 与取消仍用 /api/download/progress|cancel|resume。"""
    if reg.get(model_id) is None:
        raise HTTPException(status_code=404, detail="model not found in registry")
    version = request.version if request else None
    return await start_model_install(model_id, version=version)


@router.delete("/{model_id}/versions/{version_key}")
async def delete_registry_model_version(
    model_id: str,
    version_key: str,
    reg: ModelRegistry = Depends(get_model_registry),
):
    """删除注册表中某一版本的本地权重目录（磁盘）。"""
    entry = reg.get(model_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="model not found in registry")
    versions = entry.raw.get("versions") if isinstance(entry.raw, dict) else None
    if not isinstance(versions, dict) or version_key not in versions:
        raise HTTPException(status_code=404, detail="version not in registry")
    service = get_download_service()
    return await service.delete_model(model_id, version_key)


@router.get("/{model_id}")
def get_model_index(model_id: str, reg: ModelRegistry = Depends(get_model_registry)):
    e = reg.get(model_id)
    if not e:
        raise HTTPException(404, "model not found in registry")
    return {
        "id": e.id,
        "media": e.media,
        "family": e.family,
        "engine": e.engine,
        "actions": sorted(e.actions),
        "config": e.raw,
    }
