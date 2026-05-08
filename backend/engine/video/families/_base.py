"""视频模型族内部 ABC。"""

from __future__ import annotations

from abc import ABC, abstractmethod

from backend.core.contracts import EngineResult, ExecutionContext, VideoEditRequest, VideoGenerationRequest


class UnsupportedVideoOperation(Exception):
    def __init__(self, family: str, op: str) -> None:
        super().__init__(f"{family} does not support {op}")


class VideoFamilyAdapter(ABC):
    @abstractmethod
    async def generate(self, req: VideoGenerationRequest, ctx: ExecutionContext) -> EngineResult: ...

    async def animate(self, req: VideoEditRequest, ctx: ExecutionContext) -> EngineResult:
        raise UnsupportedVideoOperation(self.__class__.__name__, "animate")
