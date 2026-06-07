"""REST API for Studio canvas sessions."""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from backend.api.deps import get_canvas_session_store
from backend.persistence.canvas_session_store import CanvasSessionStore

router = APIRouter(prefix="/api/canvas", tags=["canvas"])


class CanvasSessionState(BaseModel):
    items: dict[str, Any] = Field(default_factory=dict)
    viewport: dict[str, Any] = Field(default_factory=lambda: {"zoom": 1, "panX": 0, "panY": 0})
    staging: dict[str, Any] = Field(
        default_factory=lambda: {"x": 240, "y": 180, "width": 512, "height": 512, "visible": True}
    )
    active_asset_path: str = ""
    overlays: dict[str, Any] = Field(default_factory=lambda: {"reference": None, "control": None})
    edges: list[dict[str, Any]] = Field(default_factory=list)
    composer_snapshot: dict[str, Any] = Field(default_factory=dict)


class CanvasSessionCreateRequest(BaseModel):
    media: str = "image"
    title: str = ""
    state: Optional[CanvasSessionState] = None


class CanvasSessionUpdateRequest(BaseModel):
    title: Optional[str] = None
    state: Optional[CanvasSessionState] = None


@router.get("/sessions")
async def list_canvas_sessions(
    media: str = Query("image"),
    limit: int = Query(50, ge=1, le=200),
    store: CanvasSessionStore = Depends(get_canvas_session_store),
):
    return {"items": store.list_sessions(media=media, limit=limit)}


@router.get("/sessions/{session_id}")
async def get_canvas_session(
    session_id: str,
    store: CanvasSessionStore = Depends(get_canvas_session_store),
):
    row = store.get_session(session_id)
    if not row:
        raise HTTPException(status_code=404, detail="canvas session not found")
    return row


@router.post("/sessions", status_code=201)
async def create_canvas_session(
    body: CanvasSessionCreateRequest,
    store: CanvasSessionStore = Depends(get_canvas_session_store),
):
    state = body.state.model_dump() if body.state is not None else None
    return store.create_session(media=body.media, title=body.title, state=state)


@router.put("/sessions/{session_id}")
async def update_canvas_session(
    session_id: str,
    body: CanvasSessionUpdateRequest,
    store: CanvasSessionStore = Depends(get_canvas_session_store),
):
    state = body.state.model_dump() if body.state is not None else None
    row = store.update_session(session_id, title=body.title, state=state)
    if not row:
        raise HTTPException(status_code=404, detail="canvas session not found")
    return row


@router.delete("/sessions/{session_id}", status_code=204)
async def delete_canvas_session(
    session_id: str,
    store: CanvasSessionStore = Depends(get_canvas_session_store),
):
    if not store.delete_session(session_id):
        raise HTTPException(status_code=404, detail="canvas session not found")
