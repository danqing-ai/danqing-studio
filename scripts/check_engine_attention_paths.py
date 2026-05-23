#!/usr/bin/env python3
"""
Guard attention call paths in engine families.

Rule:
- Under ``backend/engine/families/``, direct ``ctx.attention(...)`` calls are forbidden.
- Family implementations must use shared helpers from ``backend/engine/common/attention.py``
  (for example ``attention_blhd`` / ``attention_bhsd``) to keep attention paths reusable.

Temporary exemptions can be listed in
``scripts/engine_attention_paths_allowlist.txt`` (one repo-relative path per line).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = ROOT / "backend" / "engine" / "families"
ALLOWLIST = ROOT / "scripts" / "engine_attention_paths_allowlist.txt"

FORBIDDEN_RE = re.compile(r"\bctx\.attention\s*\(")


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
        for i, line in enumerate(text.splitlines(), 1):
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if FORBIDDEN_RE.search(s):
                violations.append(f"{rel}:{i}: {s[:120]}")
    return violations


def _write_allowlist() -> int:
    found: list[str] = []
    for path in sorted(FAMILIES.rglob("*.py")):
        rel = str(path.relative_to(ROOT)).replace("\\", "/")
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        if FORBIDDEN_RE.search(text):
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
        print("Forbidden direct attention calls in families:\n", file=sys.stderr)
        for item in violations:
            print(item, file=sys.stderr)
        print(
            "\nFix by routing through backend/engine/common/attention.py helpers, or temporarily "
            f"allowlist with review: {ALLOWLIST}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
