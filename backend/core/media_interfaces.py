"""
v3 媒体引擎接口：与 REST 端点 1:1 对齐。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar, List

from backend.core.contracts import (
    EngineResult,
    ExecutionContext,
    ImageEditRequest,
    ImageGenerationRequest,
    ImageUpscaleRequest,
    VideoEditRequest,
    VideoGenerationRequest,
)


class IImageEngine(ABC):
    media_type: ClassVar[str] = "image"
    engine_id: ClassVar[str]

    @abstractmethod
    def is_available(self) -> bool:
        pass

    @abstractmethod
    def is_model_ready(self, model_name: str, version: str = "") -> bool:
        pass

    @abstractmethod
    def get_supported_models(self) -> List[str]:
        pass

    @abstractmethod
    def supports(self, model_id: str, action: str) -> bool:
        """action: generate | edit | upscale"""
        ...

    @abstractmethod
    async def generate(
        self, request: ImageGenerationRequest, ctx: ExecutionContext
    ) -> EngineResult:
        pass

    @abstractmethod
    async def edit(self, request: ImageEditRequest, ctx: ExecutionContext) -> EngineResult:
        pass

    @abstractmethod
    async def upscale(self, request: ImageUpscaleRequest, ctx: ExecutionContext) -> EngineResult:
        pass

    @abstractmethod
    async def cancel(self, task_id: str) -> bool:
        pass


class IVideoEngine(ABC):
    media_type: ClassVar[str] = "video"
    engine_id: ClassVar[str]

    @abstractmethod
    def is_available(self) -> bool:
        pass

    @abstractmethod
    def is_model_ready(self, model_name: str, version: str = "") -> bool:
        pass

    @abstractmethod
    def get_supported_models(self) -> List[str]:
        pass

    @abstractmethod
    def supports(self, model_id: str, action: str) -> bool:
        """action: generate | edit"""
        ...

    @abstractmethod
    async def generate(
        self, request: VideoGenerationRequest, ctx: ExecutionContext
    ) -> EngineResult:
        pass

    @abstractmethod
    async def edit(self, request: VideoEditRequest, ctx: ExecutionContext) -> EngineResult:
        pass

    @abstractmethod
    async def cancel(self, task_id: str) -> bool:
        pass
