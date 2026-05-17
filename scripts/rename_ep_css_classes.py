#!/usr/bin/env python3
"""Rename legacy studio-ep-* / settings-ep-* layout classes (Element Plus migration era)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGETS = [
    ROOT / "frontend" / "src",
]

REPLACEMENTS = [
    ("settings-ep-", "settings-"),
    ("studio-ep-", "studio-"),
]


def main() -> int:
    n = 0
    for base in TARGETS:
        for path in base.rglob("*"):
            if path.suffix not in {".vue", ".css", ".ts"}:
                continue
            text = path.read_text(encoding="utf-8")
            updated = text
            for old, new in REPLACEMENTS:
                updated = updated.replace(old, new)
            if updated != text:
                path.write_text(updated)
                n += 1
                print(path.relative_to(ROOT))
    print(f"renamed in {n} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
