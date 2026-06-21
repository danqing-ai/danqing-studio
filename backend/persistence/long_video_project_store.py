"""SQLite persistence for long-video workbench projects (independent of canvas sessions)."""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import datetime
from typing import Any, Optional


def _default_title() -> str:
    return "Long video"


def _shot_stats(state: dict[str, Any]) -> tuple[int, int, int, bool]:
    shots = state.get("shots") if isinstance(state.get("shots"), list) else []
    kf = sum(1 for s in shots if isinstance(s, dict) and s.get("keyframe_asset_id"))
    seg = sum(
        1
        for s in shots
        if isinstance(s, dict) and s.get("status") == "segment_ready" and s.get("segment_asset_id")
    )
    return len(shots), kf, seg, bool(state.get("final_asset_id"))


class LongVideoProjectStore:
    def __init__(self, db_path):
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
                CREATE TABLE IF NOT EXISTS long_video_projects (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    state_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_long_video_projects_updated
                    ON long_video_projects(updated_at);
                """
            )
            conn.commit()

    def list_projects(self, *, limit: int = 100) -> list[dict[str, Any]]:
        lim = max(1, min(int(limit), 200))
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT id, title, state_json, created_at, updated_at
                FROM long_video_projects
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (lim,),
            ).fetchall()
        out: list[dict[str, Any]] = []
        for row in rows:
            st = self._parse_state(row["state_json"])
            shot_count, keyframe_count, segment_count, has_final = _shot_stats(st)
            out.append(
                {
                    "id": row["id"],
                    "title": row["title"],
                    "shot_count": shot_count,
                    "keyframe_count": keyframe_count,
                    "segment_count": segment_count,
                    "has_final": has_final,
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                }
            )
        return out

    def get_project(self, project_id: str) -> Optional[dict[str, Any]]:
        with self._conn() as conn:
            row = conn.execute(
                """
                SELECT id, title, state_json, created_at, updated_at
                FROM long_video_projects WHERE id = ?
                """,
                (project_id,),
            ).fetchone()
        if not row:
            return None
        return self._row_to_detail(row)

    def create_project(
        self,
        *,
        title: str = "",
        state: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        pid = "lvp_" + uuid.uuid4().hex[:20]
        now = datetime.now().isoformat()
        st = dict(state or {})
        st.setdefault("version", 1)
        st.setdefault("shots", [])
        title_clean = (title or "").strip() or _default_title()
        with self._lock:
            conn = self._conn()
            conn.execute(
                """
                INSERT INTO long_video_projects (id, title, state_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (pid, title_clean, json.dumps(st, ensure_ascii=False), now, now),
            )
            conn.commit()
        return {
            "id": pid,
            "title": title_clean,
            "state": st,
            "created_at": now,
            "updated_at": now,
        }

    def update_project(
        self,
        project_id: str,
        *,
        title: Optional[str] = None,
        state: Optional[dict[str, Any]] = None,
    ) -> Optional[dict[str, Any]]:
        existing = self.get_project(project_id)
        if not existing:
            return None
        now = datetime.now().isoformat()
        new_title = (title.strip() if title is not None else existing["title"])
        new_state = state if state is not None else existing["state"]
        with self._lock:
            conn = self._conn()
            conn.execute(
                """
                UPDATE long_video_projects
                SET title = ?, state_json = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    new_title,
                    json.dumps(new_state, ensure_ascii=False),
                    now,
                    project_id,
                ),
            )
            conn.commit()
        return {
            "id": project_id,
            "title": new_title,
            "state": new_state,
            "created_at": existing["created_at"],
            "updated_at": now,
        }

    def delete_project(self, project_id: str) -> bool:
        with self._lock:
            conn = self._conn()
            cur = conn.execute("DELETE FROM long_video_projects WHERE id = ?", (project_id,))
            conn.commit()
            return cur.rowcount > 0

    @staticmethod
    def _parse_state(raw: str) -> dict[str, Any]:
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                if not isinstance(data.get("shots"), list):
                    data["shots"] = []
                return data
        except (json.JSONDecodeError, TypeError):
            pass
        return {"version": 1, "shots": []}

    def _row_to_detail(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "title": row["title"],
            "state": self._parse_state(row["state_json"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }
