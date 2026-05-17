"""Custom workspace root resolution and layout (models / outputs / db / config)."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

_WORKSPACE_SUBDIRS = (
    "config",
    "db",
    "models",
    "models/Lora",
    "outputs",
    "outputs/assets",
)

_LEGACY_DIR_KEYS = ("custom_models_dir", "custom_loras_dir", "custom_outputs_dir")

_WORKSPACE_TOP_LEVEL = ("config", "db", "models", "outputs")

_IGNORE_EMPTY_NAMES = frozenset({".DS_Store", "Thumbs.db", "desktop.ini"})


def read_bootstrap_config(bootstrap_root: Path) -> dict[str, Any]:
    path = bootstrap_root / "config" / ".app_config.json"
    if not path.is_file():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def write_bootstrap_workspace_pointer(bootstrap_root: Path, workspace_dir: str) -> None:
    cfg_dir = bootstrap_root / "config"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    path = cfg_dir / ".app_config.json"
    data: dict[str, Any] = {}
    if path.is_file():
        data = read_bootstrap_config(bootstrap_root)
    data["custom_workspace_dir"] = (workspace_dir or "").strip()
    for key in _LEGACY_DIR_KEYS:
        data.pop(key, None)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def is_workspace_configured(bootstrap_root: Path) -> bool:
    """True when the user has explicitly chosen a custom workspace directory."""
    raw = (read_bootstrap_config(bootstrap_root).get("custom_workspace_dir") or "").strip()
    return bool(raw)


def normalize_workspace_path(bootstrap_root: Path, raw: str) -> Path:
    text = (raw or "").strip()
    if not text:
        raise ValueError("workspace path is required")
    candidate = Path(text).expanduser()
    if not candidate.is_absolute():
        candidate = (bootstrap_root / candidate).resolve()
    else:
        candidate = candidate.resolve()
    return candidate


def is_empty_directory(path: Path) -> bool:
    if not path.exists():
        return True
    if not path.is_dir():
        raise RuntimeError(f"Not a directory: {path}")
    for entry in path.iterdir():
        if entry.name in _IGNORE_EMPTY_NAMES:
            continue
        return False
    return True


def _assert_workspace_paths_safe(old_root: Path, new_root: Path) -> None:
    old_r = old_root.resolve()
    new_r = new_root.resolve()
    if old_r == new_r:
        return
    try:
        if new_r.is_relative_to(old_r) or old_r.is_relative_to(new_r):
            raise RuntimeError(
                f"Workspace path must not be inside the current workspace (or vice versa): {new_r}"
            )
    except ValueError:
        pass


def migrate_workspace_data(old_root: Path, new_root: Path) -> None:
    """Move workspace data directories from old_root into an empty new_root."""
    old_r = old_root.resolve()
    new_r = new_root.resolve()
    if old_r == new_r:
        return
    _assert_workspace_paths_safe(old_r, new_r)
    if not is_empty_directory(new_r):
        raise RuntimeError(f"Target workspace directory is not empty: {new_r}")

    new_r.mkdir(parents=True, exist_ok=True)
    for name in _WORKSPACE_TOP_LEVEL:
        src = old_r / name
        dst = new_r / name
        if src.exists():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.move(str(src), str(dst))
        elif name == "models":
            (new_r / "models" / "Lora").mkdir(parents=True, exist_ok=True)
        else:
            (new_r / name).mkdir(parents=True, exist_ok=True)


def apply_workspace_relocation(
    *,
    bootstrap_root: Path,
    old_root: Path,
    new_path_raw: str,
) -> Path:
    """Validate empty target, migrate data, prepare layout; returns resolved new workspace root."""
    new_root = normalize_workspace_path(bootstrap_root, new_path_raw)
    if new_root.resolve() == old_root.resolve():
        return new_root
    if not is_empty_directory(new_root):
        raise RuntimeError(f"Target workspace directory is not empty: {new_root}")
    migrate_workspace_data(old_root, new_root)
    ensure_workspace_layout(new_root)
    seed_workspace_from_bootstrap(bootstrap_root, new_root)
    write_bootstrap_workspace_pointer(bootstrap_root, str(new_root))
    db_path = new_root / "db" / "studio.db"
    if db_path.is_file():
        from backend.persistence.asset_store import repair_asset_paths_in_database

        repair_asset_paths_in_database(
            db_path,
            new_root / "outputs" / "assets",
            former_workspace_roots=[old_root],
        )
    return new_root


def resolve_workspace_root(bootstrap_root: Path) -> Path:
    """Effective data root: custom workspace if configured, else install/dev root."""
    raw = (read_bootstrap_config(bootstrap_root).get("custom_workspace_dir") or "").strip()
    if not raw:
        return bootstrap_root.resolve()
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = (bootstrap_root / candidate).resolve()
    else:
        candidate = candidate.resolve()
    if not candidate.is_dir():
        raise RuntimeError(
            f"Configured custom_workspace_dir does not exist or is not a directory: {candidate}"
        )
    return candidate


def _tree_has_user_files(path: Path) -> bool:
    if not path.exists():
        return False
    for entry in path.rglob("*"):
        if entry.is_file() and entry.name not in _IGNORE_EMPTY_NAMES:
            return True
    return False


def prune_obsolete_bootstrap_data_dirs(bootstrap_root: Path) -> None:
    """Remove empty legacy ``models/`` / ``outputs/`` / ``db/`` under bootstrap after workspace migration."""
    bootstrap = bootstrap_root.resolve()
    if not is_workspace_configured(bootstrap):
        return
    workspace = resolve_workspace_root(bootstrap)
    if workspace == bootstrap:
        return
    for name in _WORKSPACE_TOP_LEVEL:
        if name == "config":
            continue
        path = bootstrap / name
        if path.exists() and not _tree_has_user_files(path):
            shutil.rmtree(path)


def prepare_data_directories(bootstrap_root: Path) -> Path:
    """Create data layout under the effective workspace; keep bootstrap free of models/outputs/db when relocated.

    Bootstrap always gets ``config/`` (workspace pointer). ``models/``, ``outputs/``, and ``db/`` are created
    only under the resolved workspace root.
    """
    bootstrap = bootstrap_root.resolve()
    (bootstrap / "config").mkdir(parents=True, exist_ok=True)
    root = resolve_workspace_root(bootstrap)
    ensure_workspace_layout(root)
    if root != bootstrap and is_workspace_configured(bootstrap):
        write_bootstrap_workspace_pointer(bootstrap, str(root))
    prune_obsolete_bootstrap_data_dirs(bootstrap)
    return root


def ensure_workspace_layout(workspace_root: Path) -> None:
    for rel in _WORKSPACE_SUBDIRS:
        (workspace_root / rel).mkdir(parents=True, exist_ok=True)


def seed_workspace_from_bootstrap(bootstrap_root: Path, workspace_root: Path) -> None:
    """Copy default config assets into a new workspace (no overwrite)."""
    src_cfg = bootstrap_root / "config"
    dst_cfg = workspace_root / "config"
    dst_cfg.mkdir(parents=True, exist_ok=True)
    if not src_cfg.is_dir():
        return
    for name in (".app_config.json", "models_registry.json", "presets.json"):
        src = src_cfg / name
        dst = dst_cfg / name
        if src.is_file() and not dst.exists():
            shutil.copy2(src, dst)


def workspace_layout_paths(workspace_root: Path) -> dict[str, str]:
    return {
        "workspace": str(workspace_root),
        "config": str(workspace_root / "config"),
        "db": str(workspace_root / "db"),
        "models": str(workspace_root / "models"),
        "outputs": str(workspace_root / "outputs"),
    }


def pick_directory_native(*, prompt: str) -> str:
    """macOS folder picker via AppleScript; fail loud on other platforms."""
    if sys.platform != "darwin":
        raise RuntimeError("Directory picker is only supported on macOS.")
    safe_prompt = prompt.replace("\\", "\\\\").replace('"', '\\"')
    script = f'POSIX path of (choose folder with prompt "{safe_prompt}")'
    proc = subprocess.run(
        ["osascript", "-e", script],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError("Directory picker was cancelled or failed.")
    path = (proc.stdout or "").strip()
    if not path:
        raise RuntimeError("Directory picker returned an empty path.")
    return path
