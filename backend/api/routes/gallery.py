"""
API 路由 - 图库（Plan B）：列表**仅** SQLite `assets`；媒体字节走 `/api/assets`。

- `GET /api/gallery/images` — 与 `SQLiteAssetStore.list_assets` 对齐（无 outputs 合并、无旧路径）。
- `DELETE /api/gallery/image?path=asset:{id}` — 等同 `DELETE /api/assets/{id}`。
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from pydantic import BaseModel
from PIL import Image

from backend.core.container import get_container
from backend.persistence.asset_store import SQLiteAssetStore

router = APIRouter(prefix="/api/gallery", tags=["gallery"])


class GalleryItemResponse(BaseModel):
    path: str
    name: str
    width: int
    height: int
    created_at: str
    prompt: str
    model: str
    thumbnail: Optional[str] = None
    duration_seconds: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


def _mime_for_suffix(suf: str) -> str:
    s = suf.lower()
    if s == ".png":
        return "image/png"
    if s in (".jpg", ".jpeg"):
        return "image/jpeg"
    if s == ".webp":
        return "image/webp"
    if s == ".mp4":
        return "video/mp4"
    return "application/octet-stream"


def _kind_for_suffix(suf: str) -> str:
    if suf.lower() == ".mp4":
        return "video"
    return "image"


@router.get("/images", response_model=List[GalleryItemResponse])
def list_images(limit: int = 50, offset: int = 0):
    store = get_container().try_resolve(SQLiteAssetStore)
    if not store:
        raise HTTPException(status_code=503, detail="asset store unavailable")
    cap = min(500, max((limit + offset) * 2, 80))
    rows: list[GalleryItemResponse] = []
    for a in store.list_assets(kind=None, limit=cap, offset=0):
        # 跳过创作页面上传的参考图/蒙版（source_task_id 为空且 source_action 为 upload）
        if (a.get("source_task_id") or "") == "" and (a.get("source_action") or "") == "upload":
            continue
        aid = a["id"]
        meta = dict(a.get("metadata") or {})
        p = Path(str(a.get("path", "")))
        w = int(a.get("width") or meta.get("width") or 0)
        h = int(a.get("height") or meta.get("height") or 0)
        raw_dur = a.get("duration_seconds")
        if raw_dur is None:
            raw_dur = meta.get("duration_seconds")
        dur_val: float | None
        try:
            dur_val = float(raw_dur) if raw_dur is not None else None
        except (TypeError, ValueError):
            dur_val = None
        rows.append(
            GalleryItemResponse(
                path=f"asset:{aid}",
                name=p.name or aid,
                width=w,
                height=h,
                created_at=str(a.get("created_at") or ""),
                prompt=str(meta.get("prompt") or ""),
                model=str(meta.get("model") or ""),
                thumbnail=str(a.get("thumbnail_url") or f"/api/assets/{aid}/thumbnail"),
                duration_seconds=dur_val,
                metadata={
                    **meta,
                    "asset_kind": a.get("kind", "image"),
                },
            )
        )
    rows.sort(key=lambda r: r.created_at or "", reverse=True)
    return rows[offset : offset + limit]


@router.delete("/image")
def delete_gallery_image(path: str = Query(..., description="Must be asset:{id}")):
    if not path.startswith("asset:"):
        raise HTTPException(status_code=400, detail="expected asset:id")
    aid = path[len("asset:") :]
    store = get_container().try_resolve(SQLiteAssetStore)
    if not store:
        raise HTTPException(status_code=503, detail="asset store unavailable")
    if not store.delete(aid):
        raise HTTPException(status_code=404, detail="asset not found")
    return {"ok": True}


@router.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    store = get_container().try_resolve(SQLiteAssetStore)
    if not store:
        raise HTTPException(status_code=503, detail="asset store unavailable")

    suffix = Path(file.filename or "image.png").suffix or ".png"
    content = await file.read()
    tmp_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(content)
            tmp_path = Path(tmp.name)
        mime = _mime_for_suffix(suffix)
        kind = _kind_for_suffix(suffix)
        meta: Dict[str, Any] = {"source": "gallery_upload", "original_name": file.filename or ""}
        if kind == "image":
            try:
                with Image.open(tmp_path) as im:
                    meta["width"], meta["height"] = im.size
            except Exception:
                pass
        aid = store.create_from_file(
            tmp_path,
            kind=kind,
            mime_type=mime,
            source_task_id="gallery_upload",
            metadata=meta,
            source_action="upload",
        )
    finally:
        if tmp_path is not None:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass

    return {"path": f"asset:{aid}", "filename": aid, "asset_id": aid}
