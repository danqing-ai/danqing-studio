"""Structured run trace — replaces regex log scraping for dev diagnosis."""

from __future__ import annotations

import json
import re
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from collections.abc import Callable
from typing import Any, Iterator, Literal

from backend.observability.error_codes import ErrorCode, classify_failed_span

SpanKind = Literal["phase", "paradigm", "step", "hook", "system"]
SpanStatus = Literal["pending", "running", "ok", "failed", "skipped", "cancelled"]
EventLevel = Literal["debug", "info", "warning", "error"]

_LEGACY_GRAPH_STEP_RE = re.compile(r"^\[([^\]]+)\]\s*(.*)$")
_LEGACY_NODE_ALIASES: dict[str, str] = {
    "encode_prompt": "encode",
    "load_transformer": "load_backbone",
    "decode_vae": "decode",
    "save_asset": "persist",
}


@dataclass
class Span:
    id: str
    name: str
    kind: SpanKind
    status: SpanStatus
    parent_id: str | None = None
    started_at: float = 0.0
    ended_at: float | None = None
    attributes: dict[str, Any] = field(default_factory=dict)

    def duration_ms(self) -> float | None:
        if self.ended_at is None:
            return None
        return max(0.0, (self.ended_at - self.started_at) * 1000.0)


@dataclass
class TraceEvent:
    span_id: str
    level: EventLevel
    message: str
    fields: dict[str, Any] = field(default_factory=dict)
    ts: float = 0.0


@dataclass
class FailureRecord:
    code: ErrorCode
    span_id: str | None
    span_name: str | None
    detail: str
    fields: dict[str, Any] = field(default_factory=dict)


class RunTrace:
    """Per-task trace; persisted to ``work_dir/trace.json``."""

    def __init__(self, task_id: str, *, graph_id: str = "image_create") -> None:
        self.task_id = task_id
        self.graph_id = graph_id
        self._spans: dict[str, Span] = {}
        self._events: list[TraceEvent] = []
        self._open_stack: list[str] = []
        self.failure: FailureRecord | None = None
        self._legacy_open: dict[str, str] = {}
        self._on_update: Callable[[], None] | None = None

    def set_update_callback(self, callback: Callable[[], None] | None) -> None:
        """Optional hook — scheduler pushes graph snapshots to SSE ``event: trace``."""
        self._on_update = callback

    def _notify_update(self) -> None:
        if self._on_update is not None:
            try:
                self._on_update()
            except Exception:
                pass

    @property
    def spans(self) -> list[Span]:
        return sorted(self._spans.values(), key=lambda s: s.started_at)

    @property
    def events(self) -> list[TraceEvent]:
        return list(self._events)

    def span(
        self,
        name: str,
        *,
        kind: SpanKind = "phase",
        parent_id: str | None = None,
        **attributes: Any,
    ) -> Span:
        sid = f"{name}:{uuid.uuid4().hex[:8]}"
        sp = Span(
            id=sid,
            name=name,
            kind=kind,
            status="running",
            parent_id=parent_id or (self._open_stack[-1] if self._open_stack else None),
            started_at=time.time(),
            attributes=dict(attributes),
        )
        self._spans[sid] = sp
        self._open_stack.append(sid)
        self._notify_update()
        return sp

    def end_span(self, span_id: str, *, status: SpanStatus = "ok") -> None:
        sp = self._spans.get(span_id)
        if sp is None:
            return
        sp.status = status
        sp.ended_at = time.time()
        while self._open_stack and self._open_stack[-1] != span_id:
            self._open_stack.pop()
        if self._open_stack and self._open_stack[-1] == span_id:
            self._open_stack.pop()
        self._notify_update()

    @contextmanager
    def span_ctx(
        self,
        name: str,
        *,
        kind: SpanKind = "phase",
        **attributes: Any,
    ) -> Iterator[Span]:
        sp = self.span(name, kind=kind, **attributes)
        try:
            yield sp
            self.end_span(sp.id, status="ok")
        except Exception as exc:
            self.end_span(sp.id, status="failed")
            if self.failure is None:
                self.set_failure(
                    classify_failed_span(name, str(exc)),
                    span_id=sp.id,
                    detail=str(exc),
                )
            raise

    def event(
        self,
        span_id: str,
        level: EventLevel,
        message: str,
        **fields: Any,
    ) -> None:
        self._events.append(
            TraceEvent(span_id=span_id, level=level, message=message, fields=fields, ts=time.time())
        )

    def set_failure(
        self,
        code: ErrorCode,
        *,
        span_id: str | None = None,
        detail: str = "",
        **fields: Any,
    ) -> None:
        span_name = None
        if span_id and span_id in self._spans:
            span_name = self._spans[span_id].name
        elif self._open_stack:
            sid = self._open_stack[-1]
            span_name = self._spans[sid].name if sid in self._spans else None
            span_id = span_id or sid
        self.failure = FailureRecord(
            code=code,
            span_id=span_id,
            span_name=span_name,
            detail=detail,
            fields=dict(fields),
        )

    def ingest_log_line(self, level: str, message: str) -> None:
        """Dual-write bridge: legacy ``[node_id] msg`` logs → spans."""
        msg = (message or "").strip()
        m = _LEGACY_GRAPH_STEP_RE.match(msg)
        if not m:
            return
        raw_node = m.group(1).strip()
        detail = (m.group(2) or "").strip()
        node = _LEGACY_NODE_ALIASES.get(raw_node, raw_node)

        if node in self._legacy_open:
            self.end_span(self._legacy_open[node], status="ok")

        sp = self.span(node, kind="phase")
        self._legacy_open[node] = sp.id
        if detail:
            self.event(sp.id, "info" if level != "error" else "error", detail)
        if level == "error":
            self.end_span(sp.id, status="failed")
            self._legacy_open.pop(node, None)
            if self.failure is None:
                self.set_failure(
                    classify_failed_span(node, detail or msg),
                    span_id=sp.id,
                    detail=detail or msg,
                )

    def mark_cancelled(self) -> None:
        for sid in list(self._open_stack):
            self.end_span(sid, status="cancelled")
        if self.failure is None:
            self.set_failure(ErrorCode.CANCELLED, detail="task cancelled")

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "graph_id": self.graph_id,
            "spans": [asdict(s) for s in self.spans],
            "events": [asdict(e) for e in self.events],
            "failure": (
                {
                    "code": self.failure.code.value,
                    "span_id": self.failure.span_id,
                    "span_name": self.failure.span_name,
                    "detail": self.failure.detail,
                    "fields": self.failure.fields,
                }
                if self.failure
                else None
            ),
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> dict[str, Any] | None:
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
