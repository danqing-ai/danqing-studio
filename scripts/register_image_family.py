#!/usr/bin/env python3
"""Print (and optionally apply) registry wiring for a new Shape-A image family."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _snake(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", s.strip())
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)
    return s.strip("_").lower()


def _pascal(s: str) -> str:
    return "".join(part.capitalize() for part in _snake(s).split("_") if part)


def _bootstrap_snippet(family: str, fn_name: str) -> str:
    return f"""    from backend.engine.families.{family}.plugin import {fn_name}
    {fn_name}()"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--family", required=True, help="Registry family id (e.g. my_family)")
    ap.add_argument("--class", dest="class_name", default="", help="Transformer class name")
    ap.add_argument("--scaffold", action="store_true", help="Run scaffold_image_family.py first")
    args = ap.parse_args()

    family = _snake(args.family)
    cls_name = args.class_name or _pascal(family) + "Transformer"
    register_fn = f"register_{family}_plugin"
    build_fn = f"build_{family}_plugin"

    if args.scaffold:
        import subprocess

        rc = subprocess.call(
            [sys.executable, str(ROOT / "scripts/scaffold_image_family.py"), "--family", family],
            cwd=ROOT,
        )
        if rc != 0:
            return rc

    print("=== model_configs.py ===")
    print(f"  Add {cls_name}Config dataclass + FAMILY_CONFIG_MAP[{family!r}]")
    print()
    print("=== _transformer_registry.py ===")
    print(f'  _TRANSFORMER["{family}"] = (')
    print(f'      "backend.engine.families.{family}.transformer",')
    print(f'      "{cls_name}",')
    print("  )")
    print()
    print("=== registry/bootstrap.py ===")
    print(_bootstrap_snippet(family, register_fn))
    print()
    print("=== models_registry.json ===")
    print(f"  Add families.{family} row (paradigm, media, backends, hooks, …)")
    print(f"  Add models.<model_id> with runtime.family={family!r}")
    print()
    print("=== Verify ===")
    print("  make sync-models-registry")
    print(f"  python -m py_compile backend/engine/families/{family}/*.py")
    print("  make verify-engine-stack")
    print("  bin/danqing-generate --model <id> --prompt test")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
