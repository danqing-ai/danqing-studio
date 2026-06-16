"""IToolsEngine — offline model tools (merge, export, …)."""
from __future__ import annotations

from abc import ABC, abstractmethod

from backend.core.contracts import EngineResult, ExecutionContext, ZImageMergeRequest


class IToolsEngine(ABC):
    engine_id: str

    @abstractmethod
    async def merge_z_image(self, request: ZImageMergeRequest, ctx: ExecutionContext) -> EngineResult:
        ...
