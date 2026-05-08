"""LTX / Wan 共用媒体管线（底层按 model 名分支）。"""

from __future__ import annotations

from backend.core.contracts import EngineResult, ExecutionContext, VideoEditRequest, VideoGenerationRequest

from ..pipeline import MlxVideoMediaPipeline
from ._base import VideoFamilyAdapter


class WiredVideoFamilyAdapter(VideoFamilyAdapter):
    __slots__ = ("_pipeline",)

    def __init__(self, pipeline: MlxVideoMediaPipeline) -> None:
        self._pipeline = pipeline

    async def generate(self, req: VideoGenerationRequest, ctx: ExecutionContext) -> EngineResult:
        return await self._pipeline.generate(req, ctx)

    async def animate(self, req: VideoEditRequest, ctx: ExecutionContext) -> EngineResult:
        return await self._pipeline.animate(req, ctx)
