#!/usr/bin/env python3
"""
Guard family-local duplicate primitive classes (SelfAttention / RMSNorm style).

Rule:
- Under ``backend/engine/families/``, direct generic primitive names are forbidden
  by default: ``SelfAttention``, ``RMSNorm``, ``_RMSNorm``.
- Family-specific names (for example ``WanVAERMSNorm``) are allowed; generic names
  are reserved for ``backend/engine/common`` primitives.
- Existing historical paths can be temporarily exempted via
  ``scripts/engine_family_primitives_allowlist.txt``.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = ROOT / "backend" / "engine" / "families"
ALLOWLIST = ROOT / "scripts" / "engine_family_primitives_allowlist.txt"

FORBIDDEN_CLASS_PATTERNS = (
    re.compile(r"^\s*class\s+SelfAttention\s*[\(:]", re.M),
    re.compile(r"^\s*class\s+RMSNorm\s*[\(:]", re.M),
    re.compile(r"^\s*class\s+_RMSNorm\s*[\(:]", re.M),
)


def _load_allowlist() -> list[str]:
    if not ALLOWLIST.is_file():
        return []
    out: list[str] = []
    for line in ALLOWLIST.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            out.append(s.rstrip("/"))
    return out


def _is_allowed(rel: str, allowlist: list[str]) -> bool:
    return any(rel == prefix or rel.startswith(prefix + "/") for prefix in allowlist)


def _collect_violations(allowlist: list[str]) -> list[str]:
    violations: list[str] = []
    if not FAMILIES.is_dir():
        return violations
    for path in sorted(FAMILIES.rglob("*.py")):
        rel = str(path.relative_to(ROOT)).replace("\\", "/")
        if _is_allowed(rel, allowlist):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        lines = text.splitlines()
        for pat in FORBIDDEN_CLASS_PATTERNS:
            for m in pat.finditer(text):
                line_no = text.count("\n", 0, m.start()) + 1
                snippet = lines[line_no - 1].strip() if 0 < line_no <= len(lines) else "<unknown>"
                violations.append(f"{rel}:{line_no}: {snippet}")
    return violations


def _write_allowlist() -> int:
    found: list[str] = []
    if not FAMILIES.is_dir():
        ALLOWLIST.write_text("", encoding="utf-8")
        print(f"Wrote 0 paths to {ALLOWLIST}")
        return 0
    for path in sorted(FAMILIES.rglob("*.py")):
        rel = str(path.relative_to(ROOT)).replace("\\", "/")
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        if any(p.search(text) for p in FORBIDDEN_CLASS_PATTERNS):
            found.append(rel)
    ALLOWLIST.write_text("\n".join(found) + ("\n" if found else ""), encoding="utf-8")
    print(f"Wrote {len(found)} paths to {ALLOWLIST}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--write-allowlist",
        action="store_true",
        help="Regenerate allowlist from current tree (migration utility).",
    )
    args = ap.parse_args()

    if args.write_allowlist:
        return _write_allowlist()

    allowlist = _load_allowlist()
    violations = _collect_violations(allowlist)
    if violations:
        print("Forbidden family primitive class patterns:\n", file=sys.stderr)
        for item in violations:
            print(item, file=sys.stderr)
        print(
            "\nReuse backend/engine/common primitives or add temporary allowlist entries "
            f"with review: {ALLOWLIST}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
