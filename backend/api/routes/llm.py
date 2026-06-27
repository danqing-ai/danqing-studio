"""
LLM routes — OpenAI-compatible /v1/chat/completions + DanQing-specific endpoints.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from backend.api.deps import get_asset_store, get_llm_service
from backend.api.routes.settings import get_settings_service
from backend.core.contracts import (
    ChatCompletionRequest,
    DescribeNodeResponse,
    EnhanceRequest,
    ImageToPromptRequest,
    ImageToPromptResponse,
    LongVideoChapterAnalyzeRequest,
    LongVideoChapterAnalyzeResponse,
    LongVideoStoryboardRequest,
    LongVideoStoryboardResponse,
    VisualAnalyzeRequest,
    VisualAnalyzeResponse,
)
from backend.engine.llm import LLMService
from backend.core.i18n import resolve_locale
from backend.persistence.asset_store import SQLiteAssetStore

router = APIRouter()


def _resolve_storyboard_locale(http_request: Request, body_locale: str) -> str:
    loc = (body_locale or "").strip().lower().split("-")[0]
    if loc in ("zh", "en"):
        return loc
    try:
        settings = get_settings_service().get_settings()
        lang = getattr(settings, "language", None)
        if lang in ("zh", "en"):
            return str(lang)
    except Exception:
        pass
    return resolve_locale(http_request.headers.get("Accept-Language"))


def _resolve_asset_image_path(
    store: SQLiteAssetStore,
    asset_id: str,
    *,
    prefer_vision: bool = True,
) -> tuple[dict, Path | None, str]:
    row = store.get_asset_record(asset_id)
    if not row:
        raise HTTPException(status_code=404, detail=f"asset not found: {asset_id}")
    kind = str(row.get("kind") or "")
    image_path: Path | None = None
    if kind in ("image", "video") and prefer_vision:
        try:
            if kind == "image":
                image_path = store.get_file_path(asset_id)
            else:
                image_path = store.get_thumbnail_path(asset_id)
                if image_path is None:
                    image_path = store.get_file_path(asset_id)
        except FileNotFoundError:
            image_path = None
    return row, image_path, kind


@router.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    service: LLMService = Depends(get_llm_service),
):
    """OpenAI-compatible chat completions endpoint.

    Supports both streaming (SSE) and non-streaming responses.
    """
    if not service.is_available():
        raise HTTPException(
            status_code=503,
            detail="LLM model not installed. Install via Models page.",
        )

    if request.stream:
        return StreamingResponse(
            service.chat_completion_stream(request),
            media_type="text/event-stream",
        )

    result = await asyncio.to_thread(service.chat_completion, request)
    return result


@router.post("/api/chat/enhance")
async def enhance_prompt(
    request: EnhanceRequest,
    service: LLMService = Depends(get_llm_service),
):
    """Enhance a user prompt for image generation."""
    if not service.is_available():
        raise HTTPException(
            status_code=503,
            detail="LLM model not installed. Install via Models page.",
        )

    result = await asyncio.to_thread(service.enhance_prompt, request)
    return result


@router.post("/api/chat/long-video-storyboard")
async def long_video_storyboard(
    request: LongVideoStoryboardRequest,
    http_request: Request,
    service: LLMService = Depends(get_llm_service),
):
    """Multi-round long-video storyboard (Plan → Expand → optional Continuity)."""
    if not service.is_available():
        raise HTTPException(
            status_code=503,
            detail="LLM model not installed. Install via Models page.",
        )
    locale = _resolve_storyboard_locale(http_request, request.locale)
    req = request.model_copy(update={"locale": locale})
    try:
        return await asyncio.to_thread(service.generate_long_video_storyboard, req)
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.post("/api/chat/long-video-chapter-analyze")
async def long_video_chapter_analyze(
    request: LongVideoChapterAnalyzeRequest,
    http_request: Request,
    service: LLMService = Depends(get_llm_service),
):
    """Analyze a novel chapter into synopsis, cast anchor, and visual scene beats."""
    if not service.is_available():
        raise HTTPException(
            status_code=503,
            detail="LLM model not installed. Install via Models page.",
        )
    locale = _resolve_storyboard_locale(http_request, request.locale)
    req = request.model_copy(update={"locale": locale})
    try:
        return await asyncio.to_thread(service.analyze_long_video_chapter, req)
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.post("/api/chat/lyrics")
async def generate_lyrics(
    request: EnhanceRequest,
    service: LLMService = Depends(get_llm_service),
):
    """Generate ACE-Step formatted lyrics from a music description."""
    if not service.is_available():
        raise HTTPException(
            status_code=503,
            detail="LLM model not installed. Install via Models page.",
        )

    try:
        lyrics = await asyncio.to_thread(
            service.generate_lyrics,
            prompt=request.prompt,
            style=request.style_positive,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    if not (lyrics or "").strip():
        raise HTTPException(
            status_code=502,
            detail="LLM returned empty lyrics. Check Settings → Default LLM Model or try again.",
        )
    return {"lyrics": lyrics}


@router.post("/api/chat/describe-node")
async def describe_canvas_node(
    request: ImageToPromptRequest,
    service: LLMService = Depends(get_llm_service),
    store: SQLiteAssetStore = Depends(get_asset_store),
):
    """Generate a canvas node note (vision when available, else text LLM metadata)."""
    if not service.is_available():
        raise HTTPException(
            status_code=503,
            detail="LLM model not installed. Install via Models page.",
        )
    asset_id = (request.asset_id or "").strip()
    if not asset_id:
        raise HTTPException(status_code=400, detail="asset_id is required")
    row, image_path, _kind = _resolve_asset_image_path(
        store, asset_id, prefer_vision=request.prefer_vision,
    )

    note, vision_used = await asyncio.to_thread(
        service.describe_node,
        row,
        image_path=image_path,
        prefer_vision=request.prefer_vision,
    )
    return DescribeNodeResponse(note=note, vision_used=vision_used)


@router.post("/api/chat/image-to-prompt")
async def image_to_prompt(
    request: ImageToPromptRequest,
    service: LLMService = Depends(get_llm_service),
    store: SQLiteAssetStore = Depends(get_asset_store),
):
    """Extract a generation prompt from a gallery/canvas image or video keyframe."""
    if not service.is_available():
        raise HTTPException(
            status_code=503,
            detail="LLM model not installed. Install via Models page.",
        )
    asset_id = (request.asset_id or "").strip()
    if not asset_id:
        raise HTTPException(status_code=400, detail="asset_id is required")
    row, image_path, kind = _resolve_asset_image_path(
        store, asset_id, prefer_vision=True,
    )
    if image_path is None:
        raise HTTPException(
            status_code=400,
            detail="asset must be an image or video with a readable preview",
        )
    try:
        prompt, vision_used = await asyncio.to_thread(
            service.image_to_prompt,
            row,
            image_path=image_path,
            media=kind if kind == "video" else "image",
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return ImageToPromptResponse(prompt=prompt, vision_used=vision_used)


@router.post("/api/chat/visual-analyze")
async def visual_analyze(
    request: VisualAnalyzeRequest,
    service: LLMService = Depends(get_llm_service),
    store: SQLiteAssetStore = Depends(get_asset_store),
):
    """Analyze a reference image for style, palette, subject, or custom creative questions."""
    if not service.is_available():
        raise HTTPException(
            status_code=503,
            detail="LLM model not installed. Install via Models page.",
        )
    asset_id = (request.asset_id or "").strip()
    question = (request.question or "").strip()
    if not asset_id:
        raise HTTPException(status_code=400, detail="asset_id is required")
    if not question:
        raise HTTPException(status_code=400, detail="question is required")
    row, image_path, _kind = _resolve_asset_image_path(
        store, asset_id, prefer_vision=True,
    )
    if image_path is None:
        raise HTTPException(
            status_code=400,
            detail="asset must be an image or video with a readable preview",
        )
    try:
        answer, vision_used = await asyncio.to_thread(
            service.analyze_reference,
            row,
            image_path=image_path,
            question=question,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return VisualAnalyzeResponse(answer=answer, vision_used=vision_used)


@router.get("/api/chat/model")
def get_llm_model_info(
    service: LLMService = Depends(get_llm_service),
):
    """Get current LLM model info; includes nested ``vision`` block when configured."""
    info = service.get_model_info()
    info["vision"] = service.get_vision_model_info()
    return info
