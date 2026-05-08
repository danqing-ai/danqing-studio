"""模型族内部 ABC — plan 7.5 FamilyAdapter。"""

from __future__ import annotations

from abc import ABC, abstractmethod

from backend.core.contracts import (
    EngineResult,
    ExecutionContext,
    ImageEditRequest,
    ImageGenerationRequest,
    ImageUpscaleRequest,
)


class UnsupportedOperation(Exception):
    def __init__(self, family: str, op: str) -> None:
        super().__init__(f"{family} does not support {op}")


class FamilyAdapter(ABC):
    @abstractmethod
    async def generate(self, req: ImageGenerationRequest, ctx: ExecutionContext) -> EngineResult: ...

    async def rewrite(self, req: ImageEditRequest, ctx: ExecutionContext) -> EngineResult:
        raise UnsupportedOperation(self.__class__.__name__, "rewrite")

    async def retouch(self, req: ImageEditRequest, ctx: ExecutionContext) -> EngineResult:
        raise UnsupportedOperation(self.__class__.__name__, "retouch")

    async def extend(self, req: ImageEditRequest, ctx: ExecutionContext) -> EngineResult:
        raise UnsupportedOperation(self.__class__.__name__, "extend")

    async def upscale(self, req: ImageUpscaleRequest, ctx: ExecutionContext) -> EngineResult:
        raise UnsupportedOperation(self.__class__.__name__, "upscale")
