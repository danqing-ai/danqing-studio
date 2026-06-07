"""资产存储抽象 — 供 ExecutionContext 引用。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, BinaryIO, Optional


class IAssetStore(ABC):
    @abstractmethod
    def create_from_file(
        self,
        src_path: Path,
        *,
        kind: str,
        mime_type: str,
        source_task_id: str,
        metadata: Optional[dict[str, Any]] = None,
        source_action: Optional[str] = None,
        parent_asset_id: Optional[str] = None,
        relation_type: Optional[str] = None,
    ) -> str:
        """复制/登记文件，返回 asset_id。"""
        pass

    @abstractmethod
    def get_file_path(self, asset_id: str) -> Path:
        pass

    @abstractmethod
    def read_bytes(self, asset_id: str) -> bytes:
        pass

    @abstractmethod
    def delete(self, asset_id: str) -> bool:
        pass
