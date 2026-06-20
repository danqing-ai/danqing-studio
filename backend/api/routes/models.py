"""Plan models：注册表索引 + 模型安装 / 批量安装 / 按版本删除。"""

import asyncio
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field, model_validator

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
from backend.core.model_registry import ModelRegistry
from backend.services.model_install_batch import (
    normalize_batch_install_items,
    run_batch_model_install,
)

router = APIRouter(prefix="/api/models", tags=["models"])


def _settings_service() -> ISettingsService:
    return get_container().resolve(ISettingsService)


class BatchInstallItem(BaseModel):
    model_id: str
    version: Optional[str] = None


class BatchInstallRequest(BaseModel):
    """注册表模型 id 列表；逐项启动 HF 下载（与单模型安装相同进度 SSE）。"""

    model_ids: Optional[List[str]] = None
    items: Optional[List[BatchInstallItem]] = None

    @model_validator(mode="after")
    def _require_payload(self):
        if not self.model_ids and not self.items:
            raise ValueError("model_ids or items is required")
        return self


@router.get("")
def list_models_index(
    media: Optional[str] = Query(None, description="image | video"),
    action: Optional[str] = Query(None, description="create | rewrite | … 等注册表 action 键"),
    installed: Optional[bool] = Query(None, description="true=至少一版本就绪；false=未就绪"),
    commercial_use_allowed: Optional[bool] = Query(
        None, description="true=注册表标注可商用；false=明确不可商用或未标注"
    ),
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
        raw = e.raw if isinstance(e.raw, dict) else {}
        cup = raw.get("commercial_use_allowed")
        if commercial_use_allowed is True and cup is not True:
            continue
        if commercial_use_allowed is False and cup is True:
            continue
        out[mid] = {
            "media": e.media,
            "family": e.family,
            "engine": e.engine,
            "actions": sorted(e.actions),
            "installed": ready,
            "commercial_use_allowed": raw.get("commercial_use_allowed"),
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
    raw_items = normalize_batch_install_items(
        model_ids=body.model_ids,
        items=[item.model_dump() for item in body.items] if body.items else None,
    )
    if not raw_items:
        raise HTTPException(status_code=400, detail="no models to install")

    results = await run_batch_model_install(
        items=raw_items,
        registry=reg,
        download_service=get_download_service(),
        settings_service=get_settings_service(),
        locale=locale,
    )
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
