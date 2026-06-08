"""Shared resolve + optional plugin load + pipeline construction for media sessions."""

from __future__ import annotations

from typing import Any, Callable

from backend.core.contracts import ExecutionContext, LogEvent
from backend.engine.sessions._context import ResolvedRun
from backend.engine.sessions._phases import load_plugin_phase, resolve_phase


def session_prepare(
    session: Any,
    request: Any,
    exec_ctx: ExecutionContext,
    on_log: Callable | None,
    *,
    resolve_log: str | Callable[[ResolvedRun], str],
    make_pipeline: Callable[[], Any],
    load_plugin: bool = True,
) -> tuple[ResolvedRun, Any, Callable | None]:
    """Resolve registry entry, optionally load plugin, build pipeline, adapt log callback."""
    trace = getattr(exec_ctx, "trace", None)
    if trace is not None:
        with trace.span_ctx("resolve", kind="phase"):
            resolved = resolve_phase(session, request, exec_ctx, runtime_ctx=session._runtime_ctx)
    else:
        resolved = resolve_phase(session, request, exec_ctx, runtime_ctx=session._runtime_ctx)

    if load_plugin:
        if trace is not None:
            with trace.span_ctx(
                "load_plugin",
                kind="phase",
                family_id=resolved.family_id,
                model_id=resolved.model_id,
            ):
                load_plugin_phase(
                    resolved,
                    project_root=session._project_root,
                    model_cache=session._cache,
                )
        else:
            load_plugin_phase(
                resolved,
                project_root=session._project_root,
                model_cache=session._cache,
            )

    if on_log:
        msg = resolve_log(resolved) if callable(resolve_log) else resolve_log
        on_log(LogEvent(level="info", message=msg))

    pipeline = make_pipeline()

    def _on_log(level: str, msg: str) -> None:
        if on_log is not None:
            on_log(LogEvent(level=level, message=msg))  # type: ignore[arg-type]

    return resolved, pipeline, _on_log if on_log else None
