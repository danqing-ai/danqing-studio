"""Trace span helpers for session phases."""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from typing import Any

from backend.engine.sessions._context import ResolvedRun


def phase_trace_span(resolved: ResolvedRun, name: str) -> AbstractContextManager[Any]:
    trace = getattr(resolved.exec_ctx, "trace", None)
    if trace is None:
        return nullcontext()
    return trace.span_ctx(name, kind="phase", family_id=resolved.family_id)
