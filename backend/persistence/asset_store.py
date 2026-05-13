"""SQLite asset registration: uploads and generation results are unified as asset_id.

Optimization points:
1. Thread-local connection pool (thread-local) reuses connections to avoid creating new ones each time
2. WAL mode + synchronous NORMAL + mmap for improved concurrency and read performance
3. created_at index to accelerate gallery sorting/pagination by time
4. Reduced lock scope: file IO (copy/ffprobe/ffmpeg) outside lock, only DB writes inside lock
5. delete_batch uses explicit transactions to reduce lock holding time
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import subprocess
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from PIL import Image

from backend.core.asset_interfaces import IAssetStore


def _ffprobe_duration(path: Path) -> Optional[float]:
    try:
        r = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if r.returncode != 0 or not r.stdout.strip():
            return None
        return float(r.stdout.strip())
    except Exception:
        return None


def _ffmpeg_first_frame(video: Path, out_png: Path) -> bool:
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(video),
                "-frames:v",
                "1",
                str(out_png),
            ],
            capture_output=True,
            timeout=120,
            check=True,
        )
        return out_png.exists() and out_png.stat().st_size > 0
    except Exception:
        return False


class SQLiteAssetStore(IAssetStore):
    def __init__(self, db_path: Path, files_root: Path):
        self._db_path = db_path
        self._root = files_root
        self._root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._local = threading.local()
        self._init_db()

    @property
    def files_root(self) -> Path:
        """资产主文件所在根目录（与 ``create_from_file`` 落盘路径一致）。"""
        return self._root

    def _conn(self) -> sqlite3.Connection:
        """Thread-local connection pool: reuse the same connection per thread to reduce connection creation overhead."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            c = sqlite3.connect(str(self._db_path), check_same_thread=False)
            c.row_factory = sqlite3.Row
            # WAL mode + performance optimization (set once per connection)
            c.execute("PRAGMA journal_mode=WAL")
            c.execute("PRAGMA synchronous=NORMAL")
            c.execute("PRAGMA temp_store=MEMORY")
            c.execute("PRAGMA mmap_size=268435456")  # 256MB
            self._local.conn = c
        return self._local.conn

    def _init_db(self) -> None:
        with self._lock:
            with self._conn() as conn:
                conn.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS assets (
                        id TEXT PRIMARY KEY,
                        kind TEXT NOT NULL,
                        mime_type TEXT NOT NULL,
                        file_path TEXT NOT NULL,
                        thumbnail_path TEXT,
                        width INTEGER,
                        height INTEGER,
                        duration_seconds REAL,
                        source_task_id TEXT,
                        source_action TEXT,
                        metadata TEXT,
                        created_at TEXT NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS idx_assets_kind ON assets(kind);
                    CREATE INDEX IF NOT EXISTS idx_assets_task ON assets(source_task_id);
                    CREATE INDEX IF NOT EXISTS idx_assets_created ON assets(created_at);
                    """
                )
                conn.commit()

    def create_from_file(
        self,
        src_path: Path,
        *,
        kind: str,
        mime_type: str,
        source_task_id: str,
        metadata: Optional[dict[str, Any]] = None,
        source_action: Optional[str] = None,
    ) -> str:
        aid = "ast_" + uuid.uuid4().hex[:24]
        ext = src_path.suffix or ".bin"
        dest = self._root / f"{aid}{ext}"
        meta = dict(metadata or {})
        thumb_path: Optional[Path] = None
        w = h = None
        duration: Optional[float] = None

        # 1. File operations outside lock (copy / ffprobe / ffmpeg may take seconds)
        shutil.copy2(src_path, dest)
        if kind == "image" and mime_type.startswith("image/"):
            try:
                with Image.open(dest) as im:
                    w, h = im.size
                    meta.setdefault("width", w)
                    meta.setdefault("height", h)
            except Exception:
                pass
        elif kind == "video":
            duration = _ffprobe_duration(dest)
            if duration is None:
                try:
                    nf = float(meta.get("num_frames") or 0)
                    fps = float(meta.get("fps") or 0)
                    if nf > 0 and fps > 0:
                        duration = nf / fps
                except (TypeError, ValueError):
                    duration = None
            if duration is not None:
                meta.setdefault("duration_seconds", duration)
            poster = self._root / f"{aid}.poster.png"
            if _ffmpeg_first_frame(dest, poster):
                thumb_path = poster
                meta.setdefault("has_poster", True)
            mw, mh = meta.get("width"), meta.get("height")
            if isinstance(mw, int) and mw > 0:
                w = mw
            if isinstance(mh, int) and mh > 0:
                h = mh

        # 2. Database operations inside lock (single INSERT, millisecond level)
        with self._lock:
            conn = self._conn()
            conn.execute(
                """
                INSERT INTO assets (
                    id, kind, mime_type, file_path, thumbnail_path,
                    width, height, duration_seconds, source_task_id, source_action, metadata, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    aid,
                    kind,
                    mime_type,
                    str(dest),
                    str(thumb_path) if thumb_path else None,
                    w,
                    h,
                    duration,
                    source_task_id,
                    source_action,
                    json.dumps(meta, ensure_ascii=False),
                    datetime.now().isoformat(),
                ),
            )
            conn.commit()
        return aid

    @staticmethod
    def _write_image_thumb(im: Image.Image, out_path: Path, max_edge: int = 512) -> None:
        im = im.convert("RGBA") if im.mode not in ("RGB", "RGBA") else im
        w, h = im.size
        scale = min(1.0, float(max_edge) / max(w, h))
        if scale < 1.0:
            nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
            im = im.resize((nw, nh), Image.Resampling.LANCZOS)
        im.save(out_path, "WEBP", quality=82)

    def get_file_path(self, asset_id: str) -> Path:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT file_path FROM assets WHERE id = ?", (asset_id,)
            ).fetchone()
        if not row:
            raise FileNotFoundError(f"asset not found: {asset_id}")
        return Path(row["file_path"])

    def get_thumbnail_path(self, asset_id: str) -> Optional[Path]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT thumbnail_path FROM assets WHERE id = ?", (asset_id,)
            ).fetchone()
        if not row or not row["thumbnail_path"]:
            return None
        p = Path(row["thumbnail_path"])
        return p if p.exists() else None

    def read_bytes(self, asset_id: str) -> bytes:
        p = self.get_file_path(asset_id)
        return p.read_bytes()

    def delete(self, asset_id: str) -> bool:
        # 1. Query outside lock (read does not block)
        try:
            p = self.get_file_path(asset_id)
        except FileNotFoundError:
            return False
        tp = self.get_thumbnail_path(asset_id)

        # 2. File delete + DB delete inside lock (atomic operation)
        with self._lock:
            try:
                if p.exists():
                    p.unlink()
            except OSError:
                pass
            if tp:
                try:
                    if tp.exists():
                        tp.unlink()
                except OSError:
                    pass
            conn = self._conn()
            cur = conn.execute("DELETE FROM assets WHERE id = ?", (asset_id,))
            conn.commit()
            return cur.rowcount > 0

    def list_assets(
        self,
        *,
        kind: Optional[str] = None,
        source_task_id: Optional[str] = None,
        created_after: Optional[str] = None,
        created_before: Optional[str] = None,
        model: Optional[str] = None,
        search: Optional[str] = None,
        exclude_upload_refs: bool = False,
        sort_by: str = "created_at",
        sort_order: str = "desc",
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        where = ["1=1"]
        args: list[Any] = []
        if kind:
            where.append("kind = ?")
            args.append(kind)
        if source_task_id is not None and source_task_id != "":
            where.append("source_task_id = ?")
            args.append(source_task_id)
        if created_after:
            where.append("created_at >= ?")
            args.append(created_after)
        if created_before:
            where.append("created_at <= ?")
            args.append(created_before)
        if model:
            where.append("(metadata LIKE ?)")
            args.append(f'%"model": "{model}"%')
        if search:
            where.append("(metadata LIKE ?)")
            args.append(f"%{search}%")
        if exclude_upload_refs:
            where.append("NOT (COALESCE(source_task_id, '') = '' AND source_action = 'upload')")

        order_col = "created_at" if sort_by in ("created_at", "name", "width", "height") else "created_at"
        order_dir = "DESC" if sort_order.lower() == "desc" else "ASC"

        sql = (
            "SELECT id, kind, mime_type, file_path, thumbnail_path, "
            "width, height, duration_seconds, source_task_id, source_action, metadata, created_at "
            "FROM assets WHERE "
            + " AND ".join(where)
            + f" ORDER BY {order_col} {order_dir} LIMIT ? OFFSET ?"
        )
        args.extend([limit, offset])
        with self._conn() as conn:
            rows = conn.execute(sql, args).fetchall()
        out = []
        for r in rows:
            aid = r["id"]
            meta_raw = r["metadata"] or "{}"
            # Lazy parsing: load JSON only when needed
            meta = json.loads(meta_raw)
            if r["width"]:
                meta.setdefault("width", r["width"])
            if r["height"]:
                meta.setdefault("height", r["height"])
            if r["duration_seconds"] is not None:
                meta.setdefault("duration_seconds", r["duration_seconds"])
            out.append(
                {
                    "id": aid,
                    "kind": r["kind"],
                    "mime_type": r["mime_type"],
                    "path": r["file_path"],
                    "thumbnail_path": r["thumbnail_path"],
                    "thumbnail_url": f"/api/assets/{aid}/thumbnail",
                    "width": r["width"] or meta.get("width"),
                    "height": r["height"] or meta.get("height"),
                    "duration_seconds": r["duration_seconds"],
                    "source_task_id": r["source_task_id"] or "",
                    "source_action": r["source_action"],
                    "created_at": r["created_at"],
                    "metadata": meta,
                }
            )
        return out

    def delete_batch(self, asset_ids: list[str]) -> dict[str, Any]:
        """Batch delete assets, using explicit transaction to reduce lock holding time."""
        removed = []
        failed = []
        with self._lock:
            conn = self._conn()
            try:
                conn.execute("BEGIN")
                for aid in asset_ids:
                    try:
                        p = self.get_file_path(aid)
                        tp = self.get_thumbnail_path(aid)
                        cur = conn.execute("DELETE FROM assets WHERE id = ?", (aid,))
                        if cur.rowcount > 0:
                            try:
                                if p.exists():
                                    p.unlink()
                            except OSError:
                                pass
                            if tp:
                                try:
                                    if tp.exists():
                                        tp.unlink()
                                except OSError:
                                    pass
                            removed.append(aid)
                        else:
                            failed.append(aid)
                    except Exception:
                        failed.append(aid)
                conn.commit()
            except Exception:
                conn.rollback()
                raise
        return {"removed": removed, "failed": failed, "total": len(asset_ids)}

    def reconcile_disk_vs_db(self, *, dry_run: bool = True) -> dict[str, Any]:
        """Reconcile ``assets.file_path`` vs disk: count missing if main file does not exist; when ``dry_run=False``, delete DB rows (reuses ``delete``)."""
        with self._conn() as conn:
            rows = conn.execute("SELECT id, file_path FROM assets").fetchall()
        missing_ids: list[str] = []
        for r in rows:
            fp = Path(r["file_path"])
            if not fp.exists():
                missing_ids.append(r["id"])
        removed: list[str] = []
        if not dry_run:
            for aid in missing_ids:
                if self.delete(aid):
                    removed.append(aid)
        return {
            "dry_run": dry_run,
            "scanned_rows": len(rows),
            "missing_file_on_disk": len(missing_ids),
            "missing_asset_ids": missing_ids,
            "removed_from_database": removed if not dry_run else [],
        }
