"""Base class for image / video / audio session orchestrators."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from backend.core.contracts import ExecutionContext
from backend.engine.sessions._context import ResolvedRun
from backend.engine.sessions._prepare import session_prepare


class MediaSession:
    """Resolve plugin + construct media pipeline holder for phased create/edit."""

    media_label: str = "media"
    load_plugin: bool = True

    def __init__(
        self,
        runtime_ctx: Any,
        model_registry: Any,
        asset_store: Any,
        model_cache: Any | None = None,
        project_root: Path | None = None,
    ) -> None:
        self._runtime_ctx = runtime_ctx
        self._registry = model_registry
        self._asset_store = asset_store
        self._cache = model_cache
        self._project_root = project_root or Path.cwd()

    def _make_pipeline(self) -> Any:
        raise NotImplementedError(f"{type(self).__name__}._make_pipeline()")

    def _resolve_log(self, resolved: ResolvedRun, *, log_tag: str | None = None) -> str:
        label = self.media_label if not log_tag else f"{self.media_label} {log_tag}"
        status = "job_runner=ok" if not self.load_plugin else "plugin=ok"
        return (
            f"[resolve] {label} session family={resolved.family_id} "
            f"model={resolved.model_id} {status}"
        )

    def _prepare(
        self,
        request: Any,
        exec_ctx: ExecutionContext,
        on_log: Callable | None,
        *,
        log_tag: str | None = None,
    ) -> tuple[ResolvedRun, Any, Callable | None]:
        return session_prepare(
            self,
            request,
            exec_ctx,
            on_log,
            resolve_log=lambda r: self._resolve_log(r, log_tag=log_tag),
            make_pipeline=self._make_pipeline,
            load_plugin=self.load_plugin,
        )
