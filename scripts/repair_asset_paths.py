#!/usr/bin/env python3
"""Rewrite ``assets.file_path`` / ``thumbnail_path`` to keys relative to ``outputs/assets``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from backend.persistence.asset_store import repair_asset_paths_in_database
from backend.utils.path_utils import PathResolver


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace",
        type=str,
        default="",
        help="Workspace root (default: effective root from bootstrap config)",
    )
    parser.add_argument(
        "--former-workspace",
        action="append",
        default=[],
        help="Previous workspace root(s) to remap absolute paths from (repeatable)",
    )
    args = parser.parse_args()

    if args.workspace.strip():
        workspace = Path(args.workspace).expanduser().resolve()
    else:
        workspace = PathResolver(_REPO).get_project_root()

    db_path = workspace / "db" / "studio.db"
    assets_root = workspace / "outputs" / "assets"
    if not db_path.is_file():
        print(f"no database at {db_path}", file=sys.stderr)
        return 1

    former = [Path(p).expanduser().resolve() for p in args.former_workspace if p.strip()]
    report = repair_asset_paths_in_database(
        db_path,
        assets_root,
        former_workspace_roots=former,
    )
    print(report)
    return 0 if report.get("ok", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
