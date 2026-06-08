"""Live pipeline graph payloads for SSE and REST."""

from __future__ import annotations

from typing import Any

from backend.observability.graph_runtime import snapshot, snapshot_to_dict
from backend.observability.trace import RunTrace


def trace_graph_payload(trace: RunTrace | None, *, locale: str = "zh") -> dict[str, Any] | None:
    if trace is None:
        return None
    snap = snapshot(trace, trace.graph_id, locale=locale)
    return snapshot_to_dict(snap, locale=locale)
