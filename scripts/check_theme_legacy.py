#!/usr/bin/env python3
"""Forbid Element Plus CSS / layout remnants in Studio frontend and dq-ui."""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STUDIO_STYLES = ROOT / "frontend" / "src" / "styles"
STUDIO_SRC = ROOT / "frontend" / "src"
DQ_UI_PACKAGES = (ROOT / ".." / "dq-ui" / "packages").resolve()

FORBIDDEN_CLASS = re.compile(
    r"""class\s*=\s*['"][^'"]*\bel-(?:button|input|select|menu|tabs|table|form|card|dialog|drawer|row|col)\b"""
)
FORBIDDEN_EP_PREFIX = re.compile(r"\b(?:studio-ep|settings-ep|gallery-ep)[-_]")
FORBIDDEN_STYLE_EL = re.compile(r"\.el-[a-z0-9_-]+")
FORBIDDEN_EL_TOKEN = re.compile(r"--el-")
FORBIDDEN_BROKEN_MODEL_CARD = re.compile(r"\.modsurface\s+card")


def _scan_css(path: Path, label: str, failures: list[str]) -> None:
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if FORBIDDEN_STYLE_EL.search(line):
            failures.append(f"{label}/{path.name}:{i}: remove `.el-*` selector")
        if FORBIDDEN_EL_TOKEN.search(line):
            failures.append(f"{label}/{path.name}:{i}: remove `--el-*` token")
        if FORBIDDEN_BROKEN_MODEL_CARD.search(line):
            failures.append(f"{label}/{path.name}:{i}: fix corrupted `.model-card` selector")
        if FORBIDDEN_EP_PREFIX.search(line):
            failures.append(f"{label}/{path.name}:{i}: remove *-ep-* layout class prefix")


def _scan_vue(path: Path, label: str, failures: list[str]) -> None:
    text = path.read_text(encoding="utf-8")
    rel = path.as_posix().split("/src/", 1)[-1] if "/src/" in path.as_posix() else path.name
    for i, line in enumerate(text.splitlines(), 1):
        if FORBIDDEN_CLASS.search(line):
            failures.append(f"{label}/{rel}:{i}: EP component class in template")
        if FORBIDDEN_EP_PREFIX.search(line):
            failures.append(f"{label}/{rel}:{i}: remove *-ep-* layout class prefix")
        if "--el-" in line:
            failures.append(f"{label}/{rel}:{i}: use `--dq-*` instead of `--el-*`")


def main() -> int:
    failures: list[str] = []

    if STUDIO_STYLES.is_dir():
        for path in STUDIO_STYLES.glob("*.css"):
            _scan_css(path, "frontend/styles", failures)

    if STUDIO_SRC.is_dir():
        for path in STUDIO_SRC.rglob("*.vue"):
            _scan_vue(path, "frontend", failures)

    if DQ_UI_PACKAGES.is_dir():
        for path in DQ_UI_PACKAGES.rglob("*.css"):
            if "node_modules" in path.parts:
                continue
            rel = path.relative_to(DQ_UI_PACKAGES)
            _scan_css(path, f"dq-ui/{rel.parent}", failures)
        for path in DQ_UI_PACKAGES.rglob("*.vue"):
            if "node_modules" in path.parts:
                continue
            _scan_vue(path, "dq-ui", failures)

    if failures:
        print(f"Theme legacy check failed ({len(failures)}):")
        for f in failures[:50]:
            print(f"  - {f}")
        if len(failures) > 50:
            print(f"  … and {len(failures) - 50} more")
        return 1
    print("Theme legacy check OK (Studio + dq-ui)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
