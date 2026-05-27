#!/usr/bin/env python3
"""
Guard family directory layout against vendor-like parallel subtrees.

Rule:
- Under ``backend/engine/families/<family>/``, directories named
  ``mlx``, ``torch``, ``runtime``, ``common`` are forbidden by default.
- Existing historical paths may be temporarily exempted via
  ``scripts/engine_family_layout_allowlist.txt``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = ROOT / "backend" / "engine" / "families"
ALLOWLIST = ROOT / "scripts" / "engine_family_layout_allowlist.txt"

FORBIDDEN_DIR_NAMES = {"mlx", "torch", "runtime", "common"}

# Parallel codec wrapper trees under backend/engine/ (use vae_codec_registry.py instead).
FORBIDDEN_ENGINE_CODEC_DIRS = (
    "backend/engine/vae_codecs",
    "backend/engine/video_codecs",
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

    for family_dir in sorted(p for p in FAMILIES.iterdir() if p.is_dir()):
        for path in sorted(family_dir.rglob("*")):
            if not path.is_dir():
                continue
            if path.name not in FORBIDDEN_DIR_NAMES:
                continue
            rel = str(path.relative_to(ROOT)).replace("\\", "/")
            if _is_allowed(rel, allowlist):
                continue
            violations.append(
                f"{rel}: forbidden family subtree directory '{path.name}' "
                "(use common/ or *_mlx.py/*_cuda.py hooks instead)"
            )
    return violations


def _collect_engine_codec_violations() -> list[str]:
    violations: list[str] = []
    for rel in FORBIDDEN_ENGINE_CODEC_DIRS:
        path = ROOT / rel
        if path.is_dir():
            violations.append(
                f"{rel}/: forbidden engine codec wrapper directory "
                "(register handlers in vae_codec_registry.py / video_codec_registry.py)"
            )
    return violations


def _write_allowlist() -> int:
    found: list[str] = []
    for family_dir in sorted(p for p in FAMILIES.iterdir() if p.is_dir()):
        for path in sorted(family_dir.rglob("*")):
            if path.is_dir() and path.name in FORBIDDEN_DIR_NAMES:
                rel = str(path.relative_to(ROOT)).replace("\\", "/")
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
    violations.extend(_collect_engine_codec_violations())
    if violations:
        print("Forbidden family layout patterns:\n", file=sys.stderr)
        for item in violations:
            print(item, file=sys.stderr)
        print(
            "\nFix by flattening into family modules or common/, or temporarily allowlist "
            f"with review: {ALLOWLIST}",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
