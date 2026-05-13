#!/usr/bin/env python3
"""
Enforce: top-level ``import mlx`` / ``import torch`` (and ``from mlx`` / ``from torch``)
only under ``backend/engine/runtime/`` or in ``*_mlx.py`` / ``*_cuda.py``.

Paths listed in ``scripts/engine_backend_import_allowlist.txt`` (one repo-relative path per line)
are skipped until migrated; shrink the allowlist over time — do not grow it without review.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "backend" / "engine"
ALLOWLIST = ROOT / "scripts" / "engine_backend_import_allowlist.txt"

FORBIDDEN_PREFIXES = (
    "import mlx",
    "from mlx",
    "import torch",
    "from torch",
)


def _allowed_path(rel: str) -> bool:
    if rel.startswith("backend/engine/runtime/"):
        return True
    name = Path(rel).name
    if name.endswith("_mlx.py") or name.endswith("_cuda.py"):
        return True
    return False


def _load_allowlist() -> set[str]:
    if not ALLOWLIST.is_file():
        return set()
    out: set[str] = set()
    for line in ALLOWLIST.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.add(line)
    return out


def _violations_in_file(path: Path, allowlist: set[str]) -> list[str]:
    rel = str(path.relative_to(ROOT)).replace("\\", "/")
    if _allowed_path(rel) or rel in allowlist:
        return []
    bad: list[str] = []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return bad
    for i, line in enumerate(text.splitlines(), 1):
        s = line.strip()
        if s.startswith("#") or not s:
            continue
        if s.startswith(FORBIDDEN_PREFIXES):
            bad.append(f"{rel}:{i}: {s[:80]}")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-allowlist", action="store_true", help="Regenerate allowlist from current tree (dev only)")
    args = ap.parse_args()
    allowlist = _load_allowlist()

    if args.write_allowlist:
        found: list[str] = []
        for path in sorted(ENGINE.rglob("*.py")):
            rel = str(path.relative_to(ROOT)).replace("\\", "/")
            if _allowed_path(rel):
                continue
            if _violations_in_file(path, set()):
                found.append(rel)
        ALLOWLIST.write_text("\n".join(found) + "\n", encoding="utf-8")
        print(f"Wrote {len(found)} paths to {ALLOWLIST}")
        return 0

    all_bad: list[str] = []
    for path in sorted(ENGINE.rglob("*.py")):
        rel = str(path.relative_to(ROOT)).replace("\\", "/")
        if _allowed_path(rel):
            continue
        for msg in _violations_in_file(path, allowlist):
            all_bad.append(msg)

    if all_bad:
        print("Forbidden top-level mlx/torch imports:\n", file=sys.stderr)
        for m in all_bad:
            print(m, file=sys.stderr)
        print(
            "\nFix by moving imports to *_mlx.py / *_cuda.py / runtime/, "
            f"or remove from allowlist after refactor: {ALLOWLIST}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
