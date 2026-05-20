"""CRUD /api/assets — plan assets.py"""

import uuid
from pathlib import Path

from fastapi import APIRouter, Body, Depends, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from backend.api.deps import get_asset_store
from backend.persistence.asset_store import SQLiteAssetStore

router = APIRouter(prefix="/api/assets", tags=["assets"])


class AssetReconcileRequest(BaseModel):
    """磁盘对账：主文件路径在磁盘上不存在时视为孤儿（可仅报告或从 DB 删除行）。"""

    dry_run: bool = Field(
        True,
        description="为 true 时只返回 missing 列表，不修改数据库；为 false 时删除对应 assets 行",
    )


@router.post("", status_code=201)
async def upload_asset(
    file: UploadFile = File(...),
    store: SQLiteAssetStore = Depends(get_asset_store),
):
    suffix = Path(file.filename or "upload.bin").suffix or ".bin"
    tid = "upl_" + uuid.uuid4().hex[:12]
    tmp = store.files_root / f"_tmp_{tid}{suffix}"
    try:
        content = await file.read()
        tmp.write_bytes(content)
        mime = file.content_type or "application/octet-stream"
        kind = "image" if mime.startswith("image/") else "file"
        aid = store.create_from_file(
            tmp,
            kind=kind,
            mime_type=mime,
            source_task_id="",
            metadata={"original_name": file.filename},
            source_action="upload",
        )
        tmp.unlink(missing_ok=True)
        return {"id": aid, "kind": kind}
    except Exception as e:
        tmp.unlink(missing_ok=True)
        raise HTTPException(500, str(e)) from e


class BatchDeleteRequest(BaseModel):
    asset_ids: list[str] = Field(..., description="要批量删除的资产 ID 列表")


@router.get("")
async def list_assets(
    kind: str | None = Query(None),
    source_task_id: str | None = Query(None, description="仅列出该任务产出的资产"),
    created_after: str | None = Query(None, description="ISO8601：created_at 下界（含）"),
    created_before: str | None = Query(None, description="ISO8601：created_at 上界（含）"),
    model: str | None = Query(None, description="按 metadata.model 筛选"),
    search: str | None = Query(None, description="搜索 metadata 中的关键词"),
    exclude_upload_refs: bool = Query(False, description="排除创作页上传的参考图/蒙版（source_task_id 为空且 source_action 为 upload）"),
    sort_by: str = Query("created_at", description="排序字段：created_at|name|width|height"),
    sort_order: str = Query("desc", description="排序方向：asc|desc"),
    limit: int = Query(40, ge=1, le=500),
    offset: int = Query(0, ge=0),
    store: SQLiteAssetStore = Depends(get_asset_store),
):
    return {
        "items": store.list_assets(
            kind=kind,
            source_task_id=source_task_id,
            created_after=created_after,
            created_before=created_before,
            model=model,
            search=search,
            exclude_upload_refs=exclude_upload_refs,
            sort_by=sort_by,
            sort_order=sort_order,
            limit=limit,
            offset=offset,
        ),
        "limit": limit,
        "offset": offset,
    }


@router.post("/reconcile")
async def reconcile_assets_disk(
    body: AssetReconcileRequest | None = Body(default=None),
    store: SQLiteAssetStore = Depends(get_asset_store),
):
    """比对 DB 登记的主文件路径与磁盘；默认 dry_run 不删库。须声明 `dry_run: false` 才会移除孤儿行。"""
    opts = body or AssetReconcileRequest()
    return store.reconcile_disk_vs_db(dry_run=opts.dry_run)


@router.get("/{asset_id}/thumbnail")
async def get_asset_thumbnail(asset_id: str, store: SQLiteAssetStore = Depends(get_asset_store)):
    tp = store.get_thumbnail_path(asset_id)
    if tp and tp.exists():
        suf = tp.suffix.lower()
        media = "image/png" if suf == ".png" else "image/webp" if suf == ".webp" else "image/jpeg"
        return FileResponse(str(tp), media_type=media)
    try:
        main = store.get_file_path(asset_id)
    except FileNotFoundError as e:
        raise HTTPException(404, "asset not found") from e
    if not main.exists():
        raise HTTPException(404, "file missing")
    suf = main.suffix.lower()
    if suf in (".png", ".jpg", ".jpeg", ".webp", ".gif"):
        media = "image/png" if suf == ".png" else "image/jpeg" if suf in (".jpg", ".jpeg") else "image/webp"
        return FileResponse(str(main), media_type=media)
    raise HTTPException(404, "no thumbnail")


_AUDIO_MEDIA = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".flac": "audio/flac",
    ".ogg": "audio/ogg",
    ".m4a": "audio/mp4",
    ".aac": "audio/aac",
    ".opus": "audio/opus",
}


@router.head("/{asset_id}/file")
@router.get("/{asset_id}/file")
async def get_asset_file(asset_id: str, store: SQLiteAssetStore = Depends(get_asset_store)):
    try:
        p = store.get_file_path(asset_id)
    except FileNotFoundError as e:
        raise HTTPException(404, "asset not found") from e
    if not p.exists():
        raise HTTPException(404, "file missing")
    media = _AUDIO_MEDIA.get(p.suffix.lower())
    if media:
        return FileResponse(str(p), media_type=media)
    return FileResponse(str(p))


@router.post("/batch-delete")
async def batch_delete_assets(
    body: BatchDeleteRequest,
    store: SQLiteAssetStore = Depends(get_asset_store),
):
    return store.delete_batch(body.asset_ids)


@router.delete("/{asset_id}")
async def delete_asset(asset_id: str, store: SQLiteAssetStore = Depends(get_asset_store)):
    return {"ok": store.delete(asset_id)}
