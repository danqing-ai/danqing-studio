"""Custom workspace root resolution and layout (models / outputs / db / config)."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from backend.utils.config_paths import (
    read_workspace_pointer,
    resolve_default_config_root,
    restore_workspace_config_from_defaults,
    seed_workspace_config_from_defaults,
    write_workspace_pointer,
)

_WORKSPACE_SUBDIRS = (
    "config",
    "db",
    "models",
    "models/Lora",
    "outputs",
    "outputs/assets",
)

_WORKSPACE_TOP_LEVEL = ("config", "db", "models", "outputs")

_IGNORE_EMPTY_NAMES = frozenset({".DS_Store", "Thumbs.db", "desktop.ini"})


def _resolve_default_config(
    bootstrap_root: Path,
    default_config_root: Path | None,
) -> Path:
    if default_config_root is not None:
        return default_config_root.resolve()
    return resolve_default_config_root(bootstrap_root=bootstrap_root.resolve(), bundle_root=None)


def sanitize_workspace_pointer(
    default_config_root: Path,
    *,
    bootstrap_root: Path,
) -> None:
    """Drop invalid pointers (missing dir, dev paths shipped in desktop bundles)."""
    raw = read_workspace_pointer(default_config_root)
    if not raw:
        return
    try:
        candidate = normalize_workspace_path(bootstrap_root, raw)
    except ValueError:
        write_workspace_pointer(default_config_root, "")
        return
    if not candidate.is_dir():
        write_workspace_pointer(default_config_root, "")


def is_workspace_configured(default_config_root: Path) -> bool:
    """True when ``default_config/workspace.pointer.json`` names a workspace directory."""
    return bool(read_workspace_pointer(default_config_root).strip())


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
    default_config_root: Path,
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
    seed_workspace_config_from_defaults(default_config_root, new_root)
    write_workspace_pointer(default_config_root, str(new_root))
    db_path = new_root / "db" / "studio.db"
    if db_path.is_file():
        from backend.persistence.asset_store import repair_asset_paths_in_database

        repair_asset_paths_in_database(
            db_path,
            new_root / "outputs" / "assets",
            former_workspace_roots=[old_root],
        )
    return new_root


def resolve_workspace_root(
    bootstrap_root: Path,
    *,
    default_config_root: Path,
) -> Path:
    """Effective data root: custom workspace if configured, else install/dev root."""
    raw = read_workspace_pointer(default_config_root)
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


def prune_obsolete_bootstrap_data_dirs(
    bootstrap_root: Path,
    *,
    default_config_root: Path,
) -> None:
    """Remove empty legacy data dirs under bootstrap after workspace migration."""
    bootstrap = bootstrap_root.resolve()
    if not is_workspace_configured(default_config_root):
        return
    workspace = resolve_workspace_root(bootstrap, default_config_root=default_config_root)
    if workspace == bootstrap:
        return
    for name in _WORKSPACE_TOP_LEVEL:
        if name == "config":
            continue
        path = bootstrap / name
        if path.exists() and not _tree_has_user_files(path):
            shutil.rmtree(path)
    legacy_cfg = bootstrap / "config"
    if legacy_cfg.is_dir() and not _tree_has_user_files(legacy_cfg):
        shutil.rmtree(legacy_cfg)


def prepare_data_directories(
    bootstrap_root: Path,
    *,
    default_config_root: Path | None = None,
) -> Path:
    """Create data layout under the effective workspace root."""
    bootstrap = bootstrap_root.resolve()
    default_cfg = _resolve_default_config(bootstrap, default_config_root)
    sanitize_workspace_pointer(default_cfg, bootstrap_root=bootstrap)
    root = resolve_workspace_root(bootstrap, default_config_root=default_cfg)
    ensure_workspace_layout(root)
    seed_workspace_config_from_defaults(default_cfg, root)
    prune_obsolete_bootstrap_data_dirs(bootstrap, default_config_root=default_cfg)
    return root


def ensure_workspace_layout(workspace_root: Path) -> None:
    for rel in _WORKSPACE_SUBDIRS:
        (workspace_root / rel).mkdir(parents=True, exist_ok=True)


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
