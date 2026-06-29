"""SQLite persistence for long-video project activity / observability events."""

from __future__ import annotations

import json
import secrets
import sqlite3
import threading
from datetime import datetime
from typing import Any, Optional


def new_activity_id() -> str:
    return "lva_" + secrets.token_hex(12)


class LongVideoActivityStore:
    def __init__(self, db_path) -> None:
        self._db_path = db_path
        self._lock = threading.Lock()
        self._local = threading.local()
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            c = sqlite3.connect(str(self._db_path), check_same_thread=False)
            c.row_factory = sqlite3.Row
            c.execute("PRAGMA journal_mode=WAL")
            self._local.conn = c
        return self._local.conn

    def _init_db(self) -> None:
        with self._lock:
            conn = self._conn()
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS long_video_project_activity (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    category TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    phase TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT '',
                    summary TEXT NOT NULL DEFAULT '',
                    task_id TEXT,
                    parse_run_id TEXT,
                    shot_id TEXT NOT NULL DEFAULT '',
                    detail_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_lvp_activity_project_created
                    ON long_video_project_activity(project_id, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_lvp_activity_task
                    ON long_video_project_activity(task_id);
                CREATE INDEX IF NOT EXISTS idx_lvp_activity_parse_run
                    ON long_video_project_activity(parse_run_id, created_at ASC);
                CREATE INDEX IF NOT EXISTS idx_lvp_activity_category
                    ON long_video_project_activity(project_id, category, created_at DESC);
                """
            )
            conn.commit()

    def append_event(
        self,
        *,
        project_id: str,
        category: str,
        event_type: str,
        phase: str = "",
        status: str = "",
        summary: str = "",
        task_id: str | None = None,
        parse_run_id: str | None = None,
        shot_id: str = "",
        detail: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        pid = (project_id or "").strip()
        if not pid:
            raise ValueError("project_id is required")
        event_id = new_activity_id()
        now = datetime.now().isoformat()
        row = {
            "id": event_id,
            "project_id": pid,
            "category": category,
            "event_type": event_type,
            "phase": phase or "",
            "status": status or "",
            "summary": summary or "",
            "task_id": task_id,
            "parse_run_id": parse_run_id,
            "shot_id": shot_id or "",
            "detail": detail or {},
            "created_at": now,
        }
        with self._lock:
            conn = self._conn()
            conn.execute(
                """
                INSERT INTO long_video_project_activity (
                    id, project_id, category, event_type, phase, status, summary,
                    task_id, parse_run_id, shot_id, detail_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    pid,
                    category,
                    event_type,
                    row["phase"],
                    row["status"],
                    row["summary"],
                    task_id,
                    parse_run_id,
                    row["shot_id"],
                    json.dumps(row["detail"], ensure_ascii=False),
                    now,
                ),
            )
            conn.commit()
        return self._public_row(row)

    def list_events(
        self,
        project_id: str,
        *,
        limit: int = 200,
        offset: int = 0,
        category: str | None = None,
        phase: str | None = None,
        event_type: str | None = None,
        parse_run_id: str | None = None,
        task_id: str | None = None,
        shot_id: str | None = None,
    ) -> list[dict[str, Any]]:
        lim = max(1, min(int(limit), 500))
        off = max(0, int(offset))
        where = ["project_id = ?"]
        args: list[Any] = [project_id]
        if category:
            where.append("category = ?")
            args.append(category)
        if phase:
            where.append("phase = ?")
            args.append(phase)
        if event_type:
            where.append("event_type = ?")
            args.append(event_type)
        if parse_run_id:
            where.append("parse_run_id = ?")
            args.append(parse_run_id)
        if task_id:
            where.append("task_id = ?")
            args.append(task_id)
        if shot_id:
            where.append("shot_id = ?")
            args.append(shot_id)
        sql = (
            "SELECT * FROM long_video_project_activity WHERE "
            + " AND ".join(where)
            + " ORDER BY created_at DESC, id DESC LIMIT ? OFFSET ?"
        )
        args.extend([lim, off])
        with self._conn() as conn:
            rows = conn.execute(sql, args).fetchall()
        return [self._row_dict(r) for r in rows]

    def count_events(
        self,
        project_id: str,
        *,
        category: str | None = None,
        event_type: str | None = None,
        parse_run_id: str | None = None,
    ) -> int:
        where = ["project_id = ?"]
        args: list[Any] = [project_id]
        if category:
            where.append("category = ?")
            args.append(category)
        if event_type:
            where.append("event_type = ?")
            args.append(event_type)
        if parse_run_id:
            where.append("parse_run_id = ?")
            args.append(parse_run_id)
        sql = "SELECT COUNT(*) AS n FROM long_video_project_activity WHERE " + " AND ".join(where)
        with self._conn() as conn:
            row = conn.execute(sql, args).fetchone()
        return int(row["n"]) if row else 0

    def get_parse_run(self, project_id: str, parse_run_id: str) -> Optional[dict[str, Any]]:
        events = self.list_events(
            project_id,
            parse_run_id=parse_run_id,
            limit=500,
            offset=0,
        )
        if not events:
            return None
        events.sort(key=lambda r: r.get("created_at") or "")
        started = next((e for e in events if e["event_type"] == "parse_started"), None)
        completed = next((e for e in events if e["event_type"] == "parse_completed"), None)
        failed = next((e for e in events if e["event_type"] == "parse_failed"), None)
        phases = [e for e in events if e["event_type"] == "parse_phase"]
        status = "running"
        if completed:
            status = "completed"
        elif failed:
            status = "failed"
        return {
            "parse_run_id": parse_run_id,
            "project_id": project_id,
            "status": status,
            "started_at": started.get("created_at") if started else events[0].get("created_at"),
            "completed_at": (completed or failed or {}).get("created_at"),
            "summary": (completed or failed or started or {}).get("summary", ""),
            "detail": (completed or failed or started or {}).get("detail") or {},
            "phases": [
                {"phase": p.get("phase"), "message": (p.get("detail") or {}).get("message", ""), "at": p.get("created_at")}
                for p in phases
            ],
            "events": events,
        }

    def _row_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        detail_raw = row["detail_json"] or "{}"
        try:
            detail = json.loads(detail_raw)
        except json.JSONDecodeError:
            detail = {}
        return {
            "id": row["id"],
            "project_id": row["project_id"],
            "category": row["category"],
            "event_type": row["event_type"],
            "phase": row["phase"] or "",
            "status": row["status"] or "",
            "summary": row["summary"] or "",
            "task_id": row["task_id"],
            "parse_run_id": row["parse_run_id"],
            "shot_id": row["shot_id"] or "",
            "detail": detail if isinstance(detail, dict) else {},
            "created_at": row["created_at"],
        }

    @staticmethod
    def _public_row(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": row["id"],
            "project_id": row["project_id"],
            "category": row["category"],
            "event_type": row["event_type"],
            "phase": row["phase"],
            "status": row["status"],
            "summary": row["summary"],
            "task_id": row["task_id"],
            "parse_run_id": row["parse_run_id"],
            "shot_id": row["shot_id"],
            "detail": row["detail"],
            "created_at": row["created_at"],
        }
