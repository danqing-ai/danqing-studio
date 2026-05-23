#!/usr/bin/env python3
"""
Guard known family-local AdaLN/modulation split regressions.

Rule:
- Under ``backend/engine/families/``, specific legacy split patterns are forbidden.
- Families should reuse shared helpers in ``backend/engine/common/norm.py``
  (for example ``unpack_modulation_2way`` / ``unpack_modulation_4way`` /
  ``unpack_modulation_6way`` / ``unpack_modulation_6table``).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = ROOT / "backend" / "engine" / "families"

FORBIDDEN_PATTERNS = (
    re.compile(r"modulation\.chunk\(\s*6\s*,\s*dim\s*=\s*1\s*\)"),
    re.compile(r"modulation\.shape\[-1\]\s*//\s*4"),
    re.compile(r"gate_msa\s*=\s*v\[:,\s*:D\]\s*\[:,\s*None,\s*:\]"),
    re.compile(r"scale\s*=\s*v\[\.\.\.,\s*:D\]"),
)


def main() -> int:
    violations: list[str] = []
    if not FAMILIES.is_dir():
        return 0
    for path in sorted(FAMILIES.rglob("*.py")):
        rel = str(path.relative_to(ROOT)).replace("\\", "/")
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        lines = text.splitlines()
        for pat in FORBIDDEN_PATTERNS:
            for m in pat.finditer(text):
                line_no = text.count("\n", 0, m.start()) + 1
                snippet = lines[line_no - 1].strip() if 0 < line_no <= len(lines) else "<unknown>"
                violations.append(f"{rel}:{line_no}: {snippet}")
    if violations:
        print("Forbidden family-local modulation split patterns found:\n", file=sys.stderr)
        for item in violations:
            print(item, file=sys.stderr)
        print(
            "\nUse shared modulation helpers from backend/engine/common/norm.py instead.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
