"""Shipped defaults (``default_config/``) vs bootstrap / workspace ``config/``.

``workspace.pointer.json`` lives **only** under bootstrap ``config/`` (install / server-data),
never under ``default_config/`` and never under the user workspace directory.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

DEFAULT_CONFIG_DIRNAME = "default_config"
BOOTSTRAP_POINTER_FILE = "workspace.pointer.json"
WORKSPACE_SETTINGS_FILE = ".app_config.json"
RESTORABLE_CONFIG_FILES = ("models_registry.json", "presets.json")


def resolve_default_config_root(*, bootstrap_root: Path, bundle_root: Path | None) -> Path:
    """Read-only factory defaults: locales, models_registry, presets."""
    if bundle_root is not None:
        bundled = bundle_root / DEFAULT_CONFIG_DIRNAME
        if bundled.is_dir():
            return bundled.resolve()
    candidate = bootstrap_root / DEFAULT_CONFIG_DIRNAME
    if candidate.is_dir():
        return candidate.resolve()
    raise RuntimeError(
        f"Missing {DEFAULT_CONFIG_DIRNAME}/ (expected under install root or PyInstaller bundle)"
    )


def read_bootstrap_workspace_pointer(bootstrap_root: Path) -> str:
    """Workspace path from bootstrap ``config/workspace.pointer.json`` only."""
    path = bootstrap_root.resolve() / "config" / BOOTSTRAP_POINTER_FILE
    if not path.is_file():
        return ""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return (data.get("custom_workspace_dir") or "").strip()
    except Exception:
        return ""


def write_bootstrap_workspace_pointer(bootstrap_root: Path, workspace_dir: str) -> None:
    """Bootstrap stores only the workspace path (not app settings or registry)."""
    cfg_dir = bootstrap_root / "config"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    path = cfg_dir / BOOTSTRAP_POINTER_FILE
    payload = {"custom_workspace_dir": (workspace_dir or "").strip()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def seed_workspace_config_from_defaults(default_root: Path, workspace_root: Path) -> None:
    """Copy factory registry/presets into workspace ``config/`` if missing."""
    dst_cfg = workspace_root / "config"
    dst_cfg.mkdir(parents=True, exist_ok=True)
    for name in RESTORABLE_CONFIG_FILES:
        src = default_root / name
        dst = dst_cfg / name
        if src.is_file() and not dst.exists():
            shutil.copy2(src, dst)


def restore_workspace_config_from_defaults(
    workspace_root: Path,
    default_root: Path,
    *,
    names: tuple[str, ...] = RESTORABLE_CONFIG_FILES,
) -> list[str]:
    """Overwrite workspace config files from factory defaults."""
    restored: list[str] = []
    dst_cfg = workspace_root / "config"
    dst_cfg.mkdir(parents=True, exist_ok=True)
    for name in names:
        src = default_root / name
        if not src.is_file():
            continue
        shutil.copy2(src, dst_cfg / name)
        restored.append(name)
    return restored
