#!/usr/bin/env python3
"""
Guard known family-local RoPE helper regressions.

Rule:
- Under ``backend/engine/families/``, specific legacy helper names are forbidden.
- Families should reuse shared RoPE helpers in ``backend/engine/common/embeddings.py``
  (for example ``apply_complex_rope_bshd`` / ``apply_complex_rope_from_cis_bshd``).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = ROOT / "backend" / "engine" / "families"

FORBIDDEN_PATTERNS = (
    re.compile(r"^\s*def\s+_apply_rope_bshd\s*\(", re.M),
    re.compile(r"^\s*def\s+_apply_rope_qwen\s*\(", re.M),
    re.compile(r"^\s*def\s+_apply_rotary\s*\(\s*self\s*,\s*x\s*,\s*freqs_cis\s*\)", re.M),
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
        print("Forbidden family-local RoPE helpers found:\n", file=sys.stderr)
        for item in violations:
            print(item, file=sys.stderr)
        print(
            "\nUse shared helpers from backend/engine/common/embeddings.py instead.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
