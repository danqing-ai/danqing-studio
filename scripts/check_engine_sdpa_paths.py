#!/usr/bin/env python3
"""
Guard family-local SDPA direct-call regressions.

Rule:
- Under ``backend/engine/families/``, direct MLX SDPA calls are forbidden:
  - ``mx.fast.scaled_dot_product_attention(...)``
  - ``from mlx.core.fast import scaled_dot_product_attention``
  - bare ``scaled_dot_product_attention(...)`` calls
- Under ``backend/engine/families/``, direct torch SDPA calls are forbidden:
  - ``F.scaled_dot_product_attention(...)``
  - ``torch.nn.functional.scaled_dot_product_attention(...)``
- Families should route through shared helpers in ``backend/engine/common/attention.py``.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = ROOT / "backend" / "engine" / "families"

FORBIDDEN_PATTERNS = (
    re.compile(r"\bmx\.fast\.scaled_dot_product_attention\s*\("),
    re.compile(r"^\s*from\s+mlx\.core\.fast\s+import\s+scaled_dot_product_attention\b", re.M),
    re.compile(r"\bF\.scaled_dot_product_attention\s*\("),
    re.compile(r"\btorch\.nn\.functional\.scaled_dot_product_attention\s*\("),
    re.compile(r"(?<![A-Za-z0-9_\.])scaled_dot_product_attention\s*\("),
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
        print("Forbidden family-local SDPA paths found:\n", file=sys.stderr)
        for item in violations:
            print(item, file=sys.stderr)
        print(
            "\nUse shared helpers from backend/engine/common/attention.py instead.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
