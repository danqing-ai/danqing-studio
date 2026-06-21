"""REST API for long-video workbench projects."""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from backend.api.deps import get_long_video_project_store
from backend.persistence.long_video_project_store import LongVideoProjectStore

router = APIRouter(prefix="/api/long-video", tags=["long-video"])


class LongVideoProjectCreateRequest(BaseModel):
    title: str = ""
    state: Optional[dict[str, Any]] = None


class LongVideoProjectUpdateRequest(BaseModel):
    title: Optional[str] = None
    state: Optional[dict[str, Any]] = None


@router.get("/projects")
async def list_long_video_projects(
    limit: int = 100,
    store: LongVideoProjectStore = Depends(get_long_video_project_store),
):
    return {"items": store.list_projects(limit=limit)}


@router.post("/projects", status_code=201)
async def create_long_video_project(
    body: LongVideoProjectCreateRequest,
    store: LongVideoProjectStore = Depends(get_long_video_project_store),
):
    return store.create_project(title=body.title, state=body.state)


@router.get("/projects/{project_id}")
async def get_long_video_project(
    project_id: str,
    store: LongVideoProjectStore = Depends(get_long_video_project_store),
):
    row = store.get_project(project_id)
    if not row:
        raise HTTPException(status_code=404, detail="long video project not found")
    return row


@router.put("/projects/{project_id}")
async def update_long_video_project(
    project_id: str,
    body: LongVideoProjectUpdateRequest,
    store: LongVideoProjectStore = Depends(get_long_video_project_store),
):
    row = store.update_project(project_id, title=body.title, state=body.state)
    if not row:
        raise HTTPException(status_code=404, detail="long video project not found")
    return row


@router.delete("/projects/{project_id}", status_code=204)
async def delete_long_video_project(
    project_id: str,
    store: LongVideoProjectStore = Depends(get_long_video_project_store),
):
    if not store.delete_project(project_id):
        raise HTTPException(status_code=404, detail="long video project not found")
