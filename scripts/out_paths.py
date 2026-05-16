"""
Unified build output paths — all packaging artifacts live under ``out/``.

Layout::

    out/
      frontend/dist/          # Vite production build
      sidecar/danqing-api/  # PyInstaller onedir (Tauri bundles this tree)
      pyinstaller/work/     # PyInstaller cache
      pyinstaller/spec/     # Generated .spec files
      desktop/cargo/        # Cargo target-dir (Tauri / Rust)
        release/bundle/     # .app / .dmg
"""

from __future__ import annotations

import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = PROJECT_ROOT / "out"

FRONTEND_DIST = OUT_ROOT / "frontend" / "dist"
SIDECAR_ROOT = OUT_ROOT / "sidecar"
SIDECAR_DIR = SIDECAR_ROOT / "danqing-api"
PYINSTALLER_WORK = OUT_ROOT / "pyinstaller" / "work"
PYINSTALLER_SPEC = OUT_ROOT / "pyinstaller" / "spec"
DESKTOP_CARGO_TARGET = OUT_ROOT / "desktop" / "cargo"
DESKTOP_BUNDLE_DIR = DESKTOP_CARGO_TARGET / "release" / "bundle"

# Legacy paths (pre-unification); removed by ``clean_build_artifacts``.
LEGACY_PATHS: tuple[Path, ...] = (
    PROJECT_ROOT / "dist",
    PROJECT_ROOT / "build",
    PROJECT_ROOT / "frontend" / "dist",
    PROJECT_ROOT / "desktop" / "src-tauri" / "target",
    PROJECT_ROOT / "danqing-api.spec",
    PROJECT_ROOT / "DanQingStudio.spec",
    PROJECT_ROOT / "scripts" / "Info.plist",
)


def ensure_out_layout() -> None:
    for path in (
        OUT_ROOT,
        FRONTEND_DIST.parent,
        SIDECAR_ROOT,
        PYINSTALLER_WORK,
        PYINSTALLER_SPEC,
        DESKTOP_CARGO_TARGET,
    ):
        path.mkdir(parents=True, exist_ok=True)


def clean_build_artifacts(*, include_frontend: bool = True) -> list[Path]:
    """Delete ``out/`` and legacy build dirs. Returns paths that were removed."""
    removed: list[Path] = []
    targets: list[Path] = list(LEGACY_PATHS)
    if include_frontend:
        targets.insert(0, OUT_ROOT)
    else:
        # Keep Vite output; drop sidecar + desktop + pyinstaller cache only.
        targets.insert(
            0,
            OUT_ROOT / "sidecar",
        )
        targets.insert(1, OUT_ROOT / "pyinstaller")
        targets.insert(2, OUT_ROOT / "desktop")

    seen: set[Path] = set()
    for path in targets:
        resolved = path.resolve()
        if resolved in seen or not path.exists():
            continue
        seen.add(resolved)
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        removed.append(path)
    return removed
