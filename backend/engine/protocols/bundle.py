"""Bundle and opaque tensor contracts (engine v3)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


class TensorRef(Protocol):
    """Opaque tensor handle — inference/session layers pass through without framework imports."""

    ...


@dataclass(frozen=True)
class MediaBundle:
    """Installed model weights on disk (v3 bundle layout)."""

    family_id: str
    model_id: str
    root: Path
    version_key: str | None = None
    manifest: dict[str, Any] = field(default_factory=dict)

    def component_path(self, name: str) -> Path:
        return self.root / name
