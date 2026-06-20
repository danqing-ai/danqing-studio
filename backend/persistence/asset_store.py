"""SQLite asset registration: uploads and generation results are unified as asset_id.

Optimization points:
1. Thread-local connection pool (thread-local) reuses connections to avoid creating new ones each time
2. WAL mode + synchronous NORMAL + mmap for improved concurrency and read performance
3. created_at index to accelerate gallery sorting/pagination by time
4. Reduced lock scope: file IO (copy/ffprobe/ffmpeg) outside lock, only DB writes inside lock
5. delete_batch commits DB rows before unlinking files (crash-safe)
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


def _path_to_storage_key(path: Path, assets_root: Path) -> str:
    """Persist paths relative to ``outputs/assets`` when possible."""
    path = path.resolve()
    root = assets_root.resolve()
    try:
        return str(path.relative_to(root))
    except ValueError:
        return path.name


def _path_from_storage_key(stored: str, assets_root: Path) -> Path:
    """Map DB key → filesystem path (must be relative to ``outputs/assets``)."""
    raw = (stored or "").strip()
    if not raw:
        raise FileNotFoundError("empty asset path")
    p = Path(raw)
    if p.is_absolute():
        raise RuntimeError(
            f"asset path must be relative to outputs/assets, got absolute: {raw!r}; "
            "run: python scripts/repair_asset_paths.py"
        )
    return (assets_root.resolve() / p).resolve()


def _normalize_storage_key(stored: str) -> str | None:
    """Convert legacy absolute paths to a key relative to ``outputs/assets``."""
    raw = (stored or "").strip()
    if not raw:
        return None
    p = Path(raw)
    if not p.is_absolute():
        return raw.replace("\\", "/").lstrip("/")
    parts = p.parts
    for i in range(len(parts) - 1):
        if parts[i] == "outputs" and parts[i + 1] == "assets":
            tail = Path(*parts[i + 2 :])
            return str(tail).replace("\\", "/") if tail.parts else None
    return p.name or None


def repair_asset_paths_in_database(
    db_path: Path,
    assets_root: Path,
    *,
    former_workspace_roots: list[Path] | None = None,
) -> dict[str, Any]:
    """One-shot DB repair: rewrite rows to paths relative to ``outputs/assets``."""
    del former_workspace_roots  # prefix remap handled by normalizing to storage keys
    assets_root = assets_root.resolve()

    if not db_path.is_file():
        return {"ok": False, "error": f"database not found: {db_path}"}

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    updated_rows = 0
    missing_main: list[str] = []
    scanned = 0
    try:
        rows = conn.execute(
            "SELECT id, file_path, thumbnail_path FROM assets"
        ).fetchall()
        scanned = len(rows)
        for r in rows:
            patch: dict[str, str] = {}
            main_key: str | None = None
            for col in ("file_path", "thumbnail_path"):
                stored = r[col]
                if not stored:
                    continue
                key = _normalize_storage_key(str(stored))
                if not key:
                    continue
                if col == "file_path":
                    main_key = key
                if key != stored:
                    patch[col] = key
            if patch:
                sets = ", ".join(f"{col} = ?" for col in patch)
                conn.execute(
                    f"UPDATE assets SET {sets} WHERE id = ?",
                    (*patch.values(), r["id"]),
                )
                updated_rows += 1
            if main_key and not (assets_root / main_key).is_file():
                missing_main.append(r["id"])
        conn.commit()
    finally:
        conn.close()

    return {
        "ok": True,
        "db_path": str(db_path),
        "assets_root": str(assets_root),
        "scanned_rows": scanned,
        "updated_rows": updated_rows,
        "missing_main_file": missing_main,
    }


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


def _ffmpeg_preview_frame(video: Path, out_png: Path, *, seek_seconds: float | None = None) -> bool:
    try:
        cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
        ]
        if seek_seconds is not None and seek_seconds > 0:
            # Put -ss before input for faster seek when extracting one preview frame.
            cmd.extend(["-ss", f"{seek_seconds:.3f}"])
        cmd.extend(
            [
                "-i",
                str(video),
                "-frames:v",
                "1",
                str(out_png),
            ]
        )
        subprocess.run(
            cmd,
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

    def rebind(self, db_path: Path, files_root: Path) -> None:
        """Point store at a new workspace DB/assets root (e.g. after workspace migration)."""
        with self._lock:
            conn = getattr(self._local, "conn", None)
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass
                self._local.conn = None
            self._db_path = db_path.resolve()
            self._root = files_root.resolve()
            self._root.mkdir(parents=True, exist_ok=True)
        self._init_db()

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
                # Schema migration: add parent_asset_id and relation_type for lineage tracking
                cols = {r[1] for r in conn.execute("PRAGMA table_info(assets)").fetchall()}
                if "parent_asset_id" not in cols:
                    conn.execute("ALTER TABLE assets ADD COLUMN parent_asset_id TEXT")
                    conn.execute("ALTER TABLE assets ADD COLUMN relation_type TEXT")
                    conn.execute(
                        "CREATE INDEX IF NOT EXISTS idx_assets_parent ON assets(parent_asset_id)"
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
        parent_asset_id: Optional[str] = None,
        relation_type: Optional[str] = None,
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
                    thumb = self._root / f"{aid}.thumb.webp"
                    self._write_image_thumb(im, thumb)
                    thumb_path = thumb
            except Exception:
                pass
        elif kind in ("video", "audio"):
            duration = _ffprobe_duration(dest)
            if duration is None and kind == "video":
                try:
                    nf = float(meta.get("num_frames") or 0)
                    fps = float(meta.get("fps") or 0)
                    if nf > 0 and fps > 0:
                        duration = nf / fps
                except (TypeError, ValueError):
                    duration = None
            if duration is None and kind == "audio":
                try:
                    ds = meta.get("duration_seconds")
                    if ds is not None:
                        duration = float(ds)
                except (TypeError, ValueError):
                    duration = None
            if duration is not None:
                meta.setdefault("duration_seconds", duration)
            poster = self._root / f"{aid}.poster.png"
            seek_sec: float | None = None
            if kind == "video" and duration is not None and duration > 0:
                # Mid-frame preview better reflects overall video quality than first frame.
                seek_sec = max(0.0, float(duration) * 0.5)
            if _ffmpeg_preview_frame(dest, poster, seek_seconds=seek_sec):
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
                    width, height, duration_seconds, source_task_id, source_action, metadata, created_at,
                    parent_asset_id, relation_type
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    aid,
                    kind,
                    mime_type,
                    _path_to_storage_key(dest, self._root),
                    _path_to_storage_key(thumb_path, self._root) if thumb_path else None,
                    w,
                    h,
                    duration,
                    source_task_id,
                    source_action,
                    json.dumps(meta, ensure_ascii=False),
                    datetime.now().isoformat(),
                    parent_asset_id,
                    relation_type,
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

    def get_asset_record(self, asset_id: str) -> Optional[dict[str, Any]]:
        """Return a single asset row dict (same shape as list_assets items)."""
        with self._conn() as conn:
            row = conn.execute(
                """
                SELECT id, kind, mime_type, file_path, thumbnail_path,
                       width, height, duration_seconds, source_task_id, source_action,
                       metadata, created_at, parent_asset_id, relation_type
                FROM assets WHERE id = ?
                """,
                (asset_id,),
            ).fetchone()
        if not row:
            return None
        meta = {}
        try:
            meta = json.loads(row["metadata"] or "{}")
        except (json.JSONDecodeError, TypeError):
            pass
        aid = row["id"]
        if row["width"]:
            meta.setdefault("width", row["width"])
        if row["height"]:
            meta.setdefault("height", row["height"])
        if row["duration_seconds"] is not None:
            meta.setdefault("duration_seconds", row["duration_seconds"])
        return {
            "id": aid,
            "kind": row["kind"],
            "mime_type": row["mime_type"],
            "path": row["file_path"],
            "thumbnail_path": row["thumbnail_path"],
            "thumbnail_url": f"/api/assets/{aid}/thumbnail",
            "width": row["width"] or meta.get("width"),
            "height": row["height"] or meta.get("height"),
            "duration_seconds": row["duration_seconds"],
            "source_task_id": row["source_task_id"] or "",
            "source_action": row["source_action"],
            "created_at": row["created_at"],
            "metadata": meta,
            "parent_asset_id": row["parent_asset_id"],
            "relation_type": row["relation_type"],
        }

    def get_file_path(self, asset_id: str) -> Path:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT file_path FROM assets WHERE id = ?", (asset_id,)
            ).fetchone()
        if not row:
            raise FileNotFoundError(f"asset not found: {asset_id}")
        return _path_from_storage_key(row["file_path"], self._root)

    def get_thumbnail_path(self, asset_id: str) -> Optional[Path]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT thumbnail_path FROM assets WHERE id = ?", (asset_id,)
            ).fetchone()
        if not row or not row["thumbnail_path"]:
            return None
        p = _path_from_storage_key(row["thumbnail_path"], self._root)
        return p if p.exists() else None

    def ensure_image_thumbnail(self, asset_id: str, *, max_edge: int = 512) -> Optional[Path]:
        """Return thumbnail path, generating a WebP preview for image assets when missing."""
        tp = self.get_thumbnail_path(asset_id)
        if tp and tp.exists():
            return tp

        with self._conn() as conn:
            row = conn.execute(
                "SELECT kind, mime_type FROM assets WHERE id = ?", (asset_id,)
            ).fetchone()
        if not row or row["kind"] != "image":
            return None
        mime = str(row["mime_type"] or "")
        if mime and not mime.startswith("image/"):
            return None

        try:
            main = self.get_file_path(asset_id)
        except FileNotFoundError:
            return None
        if not main.exists():
            return None
        if main.suffix.lower() not in (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"):
            return None

        thumb = self._root / f"{asset_id}.thumb.webp"
        try:
            with Image.open(main) as im:
                self._write_image_thumb(im, thumb, max_edge=max_edge)
        except Exception:
            return None
        if not thumb.is_file():
            return None

        key = _path_to_storage_key(thumb, self._root)
        with self._lock:
            conn = self._conn()
            conn.execute(
                "UPDATE assets SET thumbnail_path = ? WHERE id = ?",
                (key, asset_id),
            )
            conn.commit()
        return thumb

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

    def purge_generation_step_previews(self, task_id: str) -> int:
        """Remove ephemeral denoise-step previews wrongly registered as assets (legacy)."""
        if not task_id:
            return 0
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT id FROM assets
                WHERE source_task_id = ?
                  AND (
                    source_action = 'preview'
                    OR metadata LIKE '%"preview": true%'
                    OR metadata LIKE '%"preview":true%'
                  )
                """,
                (task_id,),
            ).fetchall()
        n = 0
        for row in rows:
            if self.delete(row["id"]):
                n += 1
        return n

    def purge_all_generation_step_previews(self) -> int:
        """One-shot cleanup of legacy step previews registered as gallery assets."""
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT id FROM assets
                WHERE source_action = 'preview'
                   OR metadata LIKE '%"preview": true%'
                   OR metadata LIKE '%"preview":true%'
                """
            ).fetchall()
        n = 0
        for row in rows:
            if self.delete(row["id"]):
                n += 1
        return n

    def list_assets(
        self,
        *,
        kind: Optional[str] = None,
        source_task_id: Optional[str] = None,
        parent_asset_id: Optional[str] = None,
        created_after: Optional[str] = None,
        created_before: Optional[str] = None,
        model: Optional[str] = None,
        search: Optional[str] = None,
        exclude_upload_refs: bool = False,
        exclude_step_previews: bool = True,
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
        if parent_asset_id is not None and parent_asset_id != "":
            where.append("parent_asset_id = ?")
            args.append(parent_asset_id)
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
        if exclude_step_previews:
            where.append(
                "(COALESCE(source_action, '') != 'preview' AND "
                "metadata NOT LIKE '%\"preview\": true%' AND metadata NOT LIKE '%\"preview\":true%')"
            )

        order_col = "created_at" if sort_by in ("created_at", "name", "width", "height") else "created_at"
        order_dir = "DESC" if sort_order.lower() == "desc" else "ASC"

        sql = (
            "SELECT id, kind, mime_type, file_path, thumbnail_path, "
            "width, height, duration_seconds, source_task_id, source_action, metadata, created_at, "
            "parent_asset_id, relation_type "
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
                    "parent_asset_id": r["parent_asset_id"],
                    "relation_type": r["relation_type"],
                }
            )
        return out

    def get_lineage(self, asset_id: str) -> dict[str, Any]:
        """查询资产的谱系树：当前节点 + 祖先链 + 后代树。"""

        def _row_to_node(row) -> dict:
            meta = {}
            raw = row["metadata"] or "{}"
            try:
                meta = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                pass
            return {
                "id": row["id"],
                "kind": row["kind"],
                "file_path": row["file_path"],
                "thumbnail_path": row["thumbnail_path"],
                "width": row["width"],
                "height": row["height"],
                "created_at": row["created_at"],
                "metadata": meta,
                "relation_type": row["relation_type"],
                "parent": None,
                "children": [],
            }

        def _get_row(aid: str):
            with self._conn() as conn:
                return conn.execute(
                    """SELECT id, kind, file_path, thumbnail_path, width, height,
                       created_at, metadata, relation_type, parent_asset_id
                       FROM assets WHERE id = ?""",
                    (aid,),
                ).fetchone()

        def _get_children_rows(aid: str):
            with self._conn() as conn:
                return conn.execute(
                    """SELECT id, kind, file_path, thumbnail_path, width, height,
                       created_at, metadata, relation_type, parent_asset_id
                       FROM assets WHERE parent_asset_id = ?
                       ORDER BY created_at ASC""",
                    (aid,),
                ).fetchall()

        def _build_ancestor_chain(aid: str, visited: set) -> dict | None:
            if aid in visited:
                return None
            visited.add(aid)
            row = _get_row(aid)
            if not row:
                return None
            node = _row_to_node(row)
            if row["parent_asset_id"]:
                node["parent"] = _build_ancestor_chain(
                    row["parent_asset_id"], visited
                )
            return node

        root_row = _get_row(asset_id)
        if not root_row:
            raise FileNotFoundError(f"asset not found: {asset_id}")

        node = _row_to_node(root_row)

        # Ancestor chain
        if root_row["parent_asset_id"]:
            node["parent"] = _build_ancestor_chain(
                root_row["parent_asset_id"], {asset_id}
            )

        # Descendants tree — collect visited ancestors to prevent loops
        visited_desc = {asset_id}
        chain_aid = asset_id
        for _ in range(128):  # safety limit
            r = _get_row(chain_aid)
            if not r or not r["parent_asset_id"]:
                break
            visited_desc.add(r["parent_asset_id"])
            chain_aid = r["parent_asset_id"]

        def _build_descendants_tree(aid: str, visited: set) -> list[dict]:
            result: list[dict] = []
            for child_row in _get_children_rows(aid):
                cid = child_row["id"]
                if cid in visited:
                    continue
                visited.add(cid)
                child = _row_to_node(child_row)
                child["children"] = _build_descendants_tree(cid, visited)
                result.append(child)
            return result

        node["children"] = _build_descendants_tree(asset_id, visited_desc)
        return node

    def delete_batch(self, asset_ids: list[str]) -> dict[str, Any]:
        """Batch delete assets: commit DB rows first, then unlink files (crash-safe)."""
        removed: list[str] = []
        failed: list[str] = []
        files_to_unlink: list[tuple[Path, Optional[Path]]] = []
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
                            files_to_unlink.append((p, tp))
                            removed.append(aid)
                        else:
                            failed.append(aid)
                    except Exception:
                        failed.append(aid)
                conn.commit()
            except Exception:
                conn.rollback()
                raise
        for p, tp in files_to_unlink:
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
        return {"removed": removed, "failed": failed, "total": len(asset_ids)}

    def reconcile_disk_vs_db(self, *, dry_run: bool = True) -> dict[str, Any]:
        """Reconcile ``assets.file_path`` vs disk: count missing if main file does not exist; when ``dry_run=False``, delete DB rows (reuses ``delete``)."""
        with self._conn() as conn:
            rows = conn.execute("SELECT id, file_path FROM assets").fetchall()
        missing_ids: list[str] = []
        for r in rows:
            try:
                fp = _path_from_storage_key(r["file_path"], self._root)
            except (FileNotFoundError, RuntimeError):
                missing_ids.append(r["id"])
                continue
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
