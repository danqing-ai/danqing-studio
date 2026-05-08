"""SeedVR2 — 仅放大。"""

from __future__ import annotations

from backend.core.contracts import (
    EngineResult,
    ExecutionContext,
    ImageEditRequest,
    ImageGenerationRequest,
    ImageUpscaleRequest,
)

from ..pipeline import MfluxImagePipeline
from ._base import FamilyAdapter, UnsupportedOperation


class SeedVR2Adapter(FamilyAdapter):
    __slots__ = ("_pipeline",)

    def __init__(self, pipeline: MfluxImagePipeline) -> None:
        self._pipeline = pipeline

    async def generate(self, req: ImageGenerationRequest, ctx: ExecutionContext) -> EngineResult:
        raise UnsupportedOperation("SeedVR2Adapter", "generate")

    async def rewrite(self, req: ImageEditRequest, ctx: ExecutionContext) -> EngineResult:
        raise UnsupportedOperation("SeedVR2Adapter", "rewrite")

    async def retouch(self, req: ImageEditRequest, ctx: ExecutionContext) -> EngineResult:
        raise UnsupportedOperation("SeedVR2Adapter", "retouch")

    async def extend(self, req: ImageEditRequest, ctx: ExecutionContext) -> EngineResult:
        raise UnsupportedOperation("SeedVR2Adapter", "extend")

    async def upscale(self, req: ImageUpscaleRequest, ctx: ExecutionContext) -> EngineResult:
        return await self._pipeline.upscale(req, ctx)
