"""Shared session resolution context (v3 phases)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from backend.core.contracts import ExecutionContext
from backend.engine.platform.session import PlatformSession
from backend.engine.protocols.plugin import FamilyPlugin


class MediaRunContext:
    """Base for phased create/edit run state — subclasses implement ``session_infer``."""

    def session_infer(
        self,
        *,
        pipeline: Any | None = None,
        batch_seed: int | None = None,
        batch_idx: int = 0,
        batch_on_progress: Callable | None = None,
        **_ignored: Any,
    ) -> Any:
        raise NotImplementedError(f"{type(self).__name__}.session_infer()")


@dataclass
class ResolvedRun:
    """Output of resolve + load_plugin phases."""

    model_id: str
    version_key: str | None
    family_id: str
    platform: PlatformSession
    plugin: FamilyPlugin | None
    bundle_root: Path | None
    registry_entry: Any
    exec_ctx: ExecutionContext
    request: Any
    extra: dict[str, Any] = field(default_factory=dict)


def require_resolved_bundle(resolved: ResolvedRun) -> Path:
    """Fail loud when session resolve did not yield an on-disk bundle root."""
    root = resolved.bundle_root
    if root is None or not root.is_dir():
        raise RuntimeError(f"Model bundle not found for {resolved.request.model!r}")
    return root
