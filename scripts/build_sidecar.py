#!/usr/bin/env python3
"""
Build a console PyInstaller onedir bundle for the FastAPI server (Tauri sidecar).

Output: dist/danqing-api/  (entire directory is bundled into the Tauri app via tauri.conf bundle.resources)

Usage:
    python scripts/build_sidecar.py

Packaging notes:
  - If PyInstaller still warns about ``tensorboard``, install it in the venv
    (``pip install tensorboard``) or upgrade PyInstaller / pyinstaller-hooks-contrib.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
import pyinstaller_common as pc  # noqa: E402

PROJECT_ROOT = Path(__file__).parent.parent.resolve()


def build(*, clean: bool = True) -> None:
    try:
        import PyInstaller.__main__
    except ImportError as e:
        print("Error: PyInstaller not installed. pip install pyinstaller")
        raise SystemExit(1) from e

    entry_point = PROJECT_ROOT / "backend" / "main.py"
    if not entry_point.exists():
        print(f"Error: missing entry {entry_point}")
        raise SystemExit(1)

    name = "danqing-api"
    cmd: list[str] = [
        str(entry_point),
        "--name",
        name,
        "--console",
        "--onedir",
    ]
    if clean:
        cmd.append("--clean")
    cmd.append("--noconfirm")

    if sys.platform == "darwin":
        import platform as _platform

        if _platform.machine().lower() == "arm64":
            cmd.extend(["--target-architecture", "arm64"])

    for imp in pc.get_hidden_imports():
        cmd.extend(["--hidden-import", imp])
    for data in pc.get_data_files(PROJECT_ROOT):
        cmd.extend(["--add-data", data])
    for binary in pc.get_binary_files(PROJECT_ROOT):
        cmd.extend(["--add-binary", binary])
    for hook in pc.get_runtime_hooks(PROJECT_ROOT):
        cmd.extend(["--runtime-hook", hook])

    hooks_dir = pc.pyinstaller_hooks_dir(PROJECT_ROOT)
    if hooks_dir.is_dir():
        cmd.extend(["--additional-hooks-dir", str(hooks_dir)])

    pc.apply_pyinstaller_packaging_filters()
    for mod in pc.get_exclude_modules():
        cmd.extend(["--exclude-module", mod])

    cmd.extend(["--distpath", str(PROJECT_ROOT / "dist")])
    cmd.extend(["--workpath", str(PROJECT_ROOT / "build")])
    cmd.extend(["--specpath", str(PROJECT_ROOT)])

    print("Building sidecar:", name)
    PyInstaller.__main__.run(cmd)
    out = PROJECT_ROOT / "dist" / name
    if not out.exists():
        print("Warning: expected output missing:", out)
    else:
        print("Sidecar bundle:", out)


def main() -> None:
    parser = argparse.ArgumentParser(description="PyInstaller sidecar for Tauri desktop")
    parser.add_argument("--no-clean", action="store_true", help="Skip PyInstaller --clean")
    args = parser.parse_args()
    build(clean=not args.no_clean)


if __name__ == "__main__":
    main()
