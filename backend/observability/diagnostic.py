"""Build dev-time DiagnosticBundle for agents and scripts."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from backend.observability.error_codes import ErrorCode, classify_failed_span, failure_hints
from backend.observability.graph_runtime import graph_id_for_task_kind, snapshot, snapshot_to_dict
from backend.observability.trace import RunTrace

_GRAPH_STEP_RE = re.compile(r"^\[([^\]]+)\]\s*(.*)$")

def _parse_legacy_graph_steps(logs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    for row in logs:
        msg = str(row.get("message") or "").strip()
        m = _GRAPH_STEP_RE.match(msg)
        if not m:
            continue
        steps.append(
            {
                "node": m.group(1),
                "detail": (m.group(2) or "").strip(),
                "level": row.get("level"),
                "time": row.get("time"),
            }
        )
    return steps


def _reconstruct_trace_from_logs(task_id: str, logs: list[dict[str, Any]], graph_id: str) -> RunTrace:
    trace = RunTrace(task_id, graph_id=graph_id)
    for row in logs:
        trace.ingest_log_line(str(row.get("level") or "info"), str(row.get("message") or ""))
    return trace


def build_diagnostic_bundle(
    *,
    task_id: str,
    task_row: dict[str, Any],
    logs: list[dict[str, Any]],
    work_dir: Path | None,
    health: dict[str, Any] | None = None,
    locale: str = "zh",
) -> dict[str, Any]:
    kind = str(task_row.get("kind") or "")
    graph_id = graph_id_for_task_kind(kind)
    status = str(task_row.get("status") or "")
    error_message = str(task_row.get("error") or task_row.get("error_message") or "")

    trace_path = (work_dir / "trace.json") if work_dir else None
    trace_data = RunTrace.load(trace_path) if trace_path else None
    trace = _reconstruct_trace_from_logs(task_id, logs, graph_id)
    graph_snap = snapshot(trace, graph_id)
    legacy_steps = _parse_legacy_graph_steps(logs)

    failure_block = None
    if status in ("failed", "cancelled"):
        code: ErrorCode
        span_id: str | None = None
        span_name: str | None = None
        detail = error_message

        if trace_data and isinstance(trace_data.get("failure"), dict):
            fail = trace_data["failure"]
            try:
                code = ErrorCode(str(fail.get("code") or ErrorCode.INTERNAL_ERROR.value))
            except ValueError:
                code = ErrorCode.INTERNAL_ERROR
            span_id = fail.get("span_id")
            span_name = fail.get("span_name")
            detail = str(fail.get("detail") or error_message)
        elif trace.failure:
            code = trace.failure.code
            span_id = trace.failure.span_id
            span_name = trace.failure.span_name
            detail = trace.failure.detail or error_message
        elif status == "cancelled":
            code = ErrorCode.CANCELLED
        else:
            span_name = legacy_steps[-1]["node"] if legacy_steps else None
            code = classify_failed_span(span_name, error_message)

        hints = failure_hints(code, locale=locale)
        failure_block = {
            "code": code.value,
            "span_id": span_id,
            "span_name": span_name or (legacy_steps[-1]["node"] if legacy_steps else None),
            "detail": detail,
            "hints": hints["hints"],
            "recommended_checks": [
                c.replace("{id}", task_id).replace("{task_id}", task_id)
                for c in hints["recommended_checks"]
            ],
        }

    params = task_row.get("params") if isinstance(task_row.get("params"), dict) else {}
    model_id = task_row.get("model_id") or params.get("model") or ""

    return {
        "task_id": task_id,
        "status": status,
        "kind": kind,
        "model_id": model_id,
        "params_summary": {
            "prompt": params.get("prompt"),
            "size": params.get("size"),
            "steps": params.get("steps"),
            "guidance": params.get("guidance"),
            "seed": params.get("seed"),
        },
        "graph": snapshot_to_dict(graph_snap, locale=locale),
        "trace": trace_data or (trace.to_dict() if trace else {"spans": [], "events": []}),
        "legacy_graph_steps": legacy_steps,
        "failure": failure_block,
        "artifacts": {
            "work_dir": str(work_dir) if work_dir else None,
            "trace_file": str(trace_path) if trace_path else None,
            "preview_url": f"/api/tasks/{task_id}/preview",
        },
        "context": {
            "health": health,
            "log_count": len(logs),
            "error_message": error_message or None,
            "classified_without_trace_file": trace_data is None,
        },
    }
