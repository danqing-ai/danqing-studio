"""多数图像族共用同一 mflux 管线，后续可按族拆分实现。"""

from __future__ import annotations

from backend.core.contracts import (
    EngineResult,
    ExecutionContext,
    ImageEditRequest,
    ImageGenerationRequest,
    ImageUpscaleRequest,
)

from ..pipeline import MfluxImagePipeline
from ._base import FamilyAdapter


class WiredImageFamilyAdapter(FamilyAdapter):
    __slots__ = ("_pipeline",)

    def __init__(self, pipeline: MfluxImagePipeline) -> None:
        self._pipeline = pipeline

    async def generate(self, req: ImageGenerationRequest, ctx: ExecutionContext) -> EngineResult:
        return await self._pipeline.generate(req, ctx)

    async def rewrite(self, req: ImageEditRequest, ctx: ExecutionContext) -> EngineResult:
        return await self._pipeline.rewrite(req, ctx)

    async def retouch(self, req: ImageEditRequest, ctx: ExecutionContext) -> EngineResult:
        return await self._pipeline.retouch(req, ctx)

    async def extend(self, req: ImageEditRequest, ctx: ExecutionContext) -> EngineResult:
        return await self._pipeline.extend(req, ctx)

    async def upscale(self, req: ImageUpscaleRequest, ctx: ExecutionContext) -> EngineResult:
        return await self._pipeline.upscale(req, ctx)
