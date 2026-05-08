"""
任务持久化存储 - SQLite 实现
支持任务状态、进度、日志的持久化记录
"""

import json
import sqlite3
import threading
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any, Union

from backend.core.interfaces import (
    ITaskStore,
    GenerationTask,
    GenerationParams,
    TaskStatus,
    VideoGenerationTask,
    VideoGenerationParams,
)


class SQLiteTaskStore(ITaskStore):
    """基于 SQLite 的任务持久化存储"""

    def __init__(self, db_path: Path):
        self._db_path = db_path
        self._lock = threading.Lock()
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """获取数据库连接"""
        conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """初始化数据库表"""
        with self._get_connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS tasks (
                    id TEXT PRIMARY KEY,
                    params TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    progress REAL NOT NULL DEFAULT 0.0,
                    output_path TEXT,
                    error_message TEXT,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT
                );

                CREATE TABLE IF NOT EXISTS task_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL,
                    message TEXT NOT NULL,
                    level TEXT NOT NULL DEFAULT 'info',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (task_id) REFERENCES tasks(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_logs_task_id ON task_logs(task_id);
                CREATE INDEX IF NOT EXISTS idx_logs_created ON task_logs(created_at);
                CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
                CREATE INDEX IF NOT EXISTS idx_tasks_created ON tasks(created_at);
            """)
            conn.commit()

    def save_task(self, task: GenerationTask | VideoGenerationTask) -> None:
        """保存或更新任务"""
        with self._lock:
            with self._get_connection() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO tasks
                    (id, params, status, progress, output_path, error_message,
                     created_at, started_at, completed_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        task.id,
                        json.dumps(self._params_to_dict_any(task.params)),
                        task.status.value,
                        task.progress,
                        task.output_path,
                        task.error_message,
                        task.created_at.isoformat() if task.created_at else datetime.now().isoformat(),
                        task.started_at.isoformat() if task.started_at else None,
                        task.completed_at.isoformat() if task.completed_at else None,
                    )
                )
                conn.commit()

    def get_task(self, task_id: str) -> Optional[Union[GenerationTask, VideoGenerationTask]]:
        """获取单个任务"""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM tasks WHERE id = ?", (task_id,)
            ).fetchone()

            if not row:
                return None

            return self._row_to_task(row)

    def list_tasks(
        self, limit: int = 100, offset: int = 0
    ) -> List[Union[GenerationTask, VideoGenerationTask]]:
        """列出任务，按创建时间倒序"""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM tasks ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (limit, offset)
            ).fetchall()

            return [self._row_to_task(row) for row in rows]

    def delete_task(self, task_id: str) -> bool:
        """删除任务"""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
                conn.commit()
                return cursor.rowcount > 0

    def append_log(self, task_id: str, message: str, level: str = "info") -> None:
        """追加任务日志"""
        with self._lock:
            with self._get_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO task_logs (task_id, message, level, created_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (task_id, message, level, datetime.now().isoformat())
                )
                conn.commit()

    def get_logs(self, task_id: str, offset: int = 0, limit: int = 1000) -> List[Dict[str, Any]]:
        """获取任务日志"""
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT message, level, created_at
                FROM task_logs
                WHERE task_id = ?
                ORDER BY id ASC
                LIMIT ? OFFSET ?
                """,
                (task_id, limit, offset)
            ).fetchall()

            return [
                {
                    "message": row["message"],
                    "level": row["level"],
                    "time": row["created_at"]
                }
                for row in rows
            ]

    def update_progress(self, task_id: str, progress: float) -> None:
        """更新任务进度"""
        with self._lock:
            with self._get_connection() as conn:
                conn.execute(
                    "UPDATE tasks SET progress = ? WHERE id = ?",
                    (progress, task_id)
                )
                conn.commit()

    def update_status(self, task_id: str, status: TaskStatus) -> None:
        """更新任务状态"""
        with self._lock:
            with self._get_connection() as conn:
                now = datetime.now().isoformat()
                if status == TaskStatus.RUNNING:
                    conn.execute(
                        "UPDATE tasks SET status = ?, started_at = ? WHERE id = ?",
                        (status.value, now, task_id)
                    )
                elif status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED):
                    conn.execute(
                        "UPDATE tasks SET status = ?, completed_at = ? WHERE id = ?",
                        (status.value, now, task_id)
                    )
                else:
                    conn.execute(
                        "UPDATE tasks SET status = ? WHERE id = ?",
                        (status.value, task_id)
                    )
                conn.commit()

    def _params_to_dict_any(self, params: GenerationParams | VideoGenerationParams) -> Dict[str, Any]:
        if isinstance(params, VideoGenerationParams):
            return {
                "prompt": params.prompt,
                "negative_prompt": params.negative_prompt,
                "model": params.model,
                "version": params.version,
                "width": params.width,
                "height": params.height,
                "num_frames": params.num_frames,
                "fps": params.fps,
                "steps": params.steps,
                "guide_scale": params.guide_scale,
                "shift": params.shift,
                "seed": params.seed,
                "image_path": params.image_path,
                "extra_params": params.extra_params or {},
            }
        return self._params_to_dict(params)

    def _params_to_dict(self, params: GenerationParams) -> Dict[str, Any]:
        """将参数转换为字典"""
        return {
            "prompt": params.prompt,
            "negative_prompt": params.negative_prompt,
            "model": params.model,
            "width": params.width,
            "height": params.height,
            "steps": params.steps,
            "guidance": params.guidance,
            "seed": params.seed,
            "lora": params.lora,
            "lora_scale": params.lora_scale,
            "img2img": params.img2img,
            "image_path": params.image_path,
            "strength": params.strength,
        }

    def _dict_to_params(self, data: Dict[str, Any]) -> GenerationParams:
        """将字典转换为参数"""
        return GenerationParams(
            prompt=data.get("prompt", ""),
            negative_prompt=data.get("negative_prompt", ""),
            model=data.get("model", "flux2-9b-distilled"),
            width=data.get("width", 1024),
            height=data.get("height", 1024),
            steps=data.get("steps", 4),
            guidance=data.get("guidance", 3.5),
            seed=data.get("seed"),
            lora=data.get("lora", ""),
            lora_scale=data.get("lora_scale", 0.8),
            img2img=data.get("img2img", False),
            image_path=data.get("image_path", ""),
            strength=data.get("strength", 0.4),
        )

    def _row_to_task(self, row: sqlite3.Row) -> GenerationTask:
        """将数据库行转换为任务对象"""
        raw = json.loads(row["params"])
        if "num_frames" in raw:
            vp = VideoGenerationParams(
                prompt=raw.get("prompt", ""),
                negative_prompt=raw.get("negative_prompt", ""),
                model=raw.get("model", ""),
                version=raw.get("version", ""),
                width=raw.get("width", 768),
                height=raw.get("height", 512),
                num_frames=raw.get("num_frames", 97),
                fps=raw.get("fps", 24),
                steps=raw.get("steps", 4),
                guide_scale=raw.get("guide_scale", 3.0),
                shift=raw.get("shift", 0.0),
                seed=raw.get("seed"),
                image_path=raw.get("image_path", ""),
                extra_params=raw.get("extra_params") or {},
            )
            return VideoGenerationTask(  # type: ignore[return-value]
                id=row["id"],
                params=vp,
                status=TaskStatus(row["status"]),
                progress=row["progress"] or 0.0,
                output_path=row["output_path"] or "",
                error_message=row["error_message"] or "",
                created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else datetime.now(),
                started_at=datetime.fromisoformat(row["started_at"]) if row["started_at"] else None,
                completed_at=datetime.fromisoformat(row["completed_at"]) if row["completed_at"] else None,
                logs=[],
            )
        params = self._dict_to_params(raw)
        return GenerationTask(
            id=row["id"],
            params=params,
            status=TaskStatus(row["status"]),
            progress=row["progress"] or 0.0,
            output_path=row["output_path"] or "",
            error_message=row["error_message"] or "",
            created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else datetime.now(),
            started_at=datetime.fromisoformat(row["started_at"]) if row["started_at"] else None,
            completed_at=datetime.fromisoformat(row["completed_at"]) if row["completed_at"] else None,
            logs=[],  # 日志按需从单独表加载
        )
