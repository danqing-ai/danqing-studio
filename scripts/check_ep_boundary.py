#!/usr/bin/env python3
"""Guardrails: no Element Plus in Studio frontend or dq-ui packages."""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FRONTEND_SRC = ROOT / "frontend" / "src"
DQ_UI_ROOT = (ROOT / ".." / "dq-ui" / "packages").resolve()

SCAN_SUFFIXES = {".vue", ".ts", ".css"}

FORBIDDEN_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"""from\s+['"]element-plus['"]"""), "do not import element-plus"),
    (re.compile(r"""@element-plus/icons-vue"""), "use Lucide / registerDqIcons"),
    (re.compile(r"\bElMessageBox\b"), "use confirm from @/utils/feedback"),
    (re.compile(r"\bElMessage\b"), "use toast from @/utils/feedback"),
    (re.compile(r"\bElNotification\b"), "use toast.notify from @/utils/feedback"),
    (re.compile(r"\bv-loading\b"), "use v-dq-loading"),
    (re.compile(r"<el-[a-z]"), "use Dq* components in templates"),
    (re.compile(r"\belement-plus\b", re.I), "remove element-plus reference"),
    (re.compile(r"Element\s+Plus"), "remove Element Plus reference"),
    (re.compile(r"\bstudio-ep[-_]"), "rename studio-ep-* layout classes"),
    (re.compile(r"\bsettings-ep[-_]"), "rename settings-ep-* layout classes"),
    (re.compile(r"\bgallery-ep[-_]"), "rename gallery-ep-* layout classes"),
]


def _scan_tree(root: Path, label: str, failures: list[str]) -> None:
    if not root.is_dir():
        failures.append(f"{label}: directory missing ({root})")
        return
    for path in root.rglob("*"):
        if path.suffix not in SCAN_SUFFIXES:
            continue
        if "node_modules" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        rel = path.relative_to(root).as_posix()
        for pattern, hint in FORBIDDEN_PATTERNS:
            if pattern.search(text):
                failures.append(f"{label}/{rel}: {hint}")


def main() -> int:
    failures: list[str] = []
    _scan_tree(FRONTEND_SRC, "frontend/src", failures)
    for pkg in ("ui", "shell", "tokens"):
        _scan_tree(DQ_UI_ROOT / pkg / "src", f"dq-ui/{pkg}/src", failures)

    if failures:
        print(f"EP boundary check failed ({len(failures)}):")
        for f in failures[:60]:
            print(f"  - {f}")
        if len(failures) > 60:
            print(f"  … and {len(failures) - 60} more")
        return 1
    print("EP boundary check OK (Studio + dq-ui)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
