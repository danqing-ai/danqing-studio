#!/usr/bin/env python3
"""Copy factory models_registry.json into the effective workspace config/."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.utils.config_paths import resolve_default_config_root
from backend.utils.workspace import resolve_workspace_root

DEFAULT_REGISTRY = ROOT / "default_config" / "models_registry.json"


def main() -> int:
    if not DEFAULT_REGISTRY.is_file():
        print(f"Missing factory registry: {DEFAULT_REGISTRY}", file=sys.stderr)
        return 1
    default_cfg = resolve_default_config_root(bootstrap_root=ROOT, bundle_root=None)
    workspace = resolve_workspace_root(ROOT, default_config_root=default_cfg)
    dst_dir = workspace / "config"
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / "models_registry.json"
    shutil.copy2(DEFAULT_REGISTRY, dst)
    print(f"Synced {DEFAULT_REGISTRY.name} -> {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
