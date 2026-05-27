#!/usr/bin/env python3
"""Verify remap output keys match transformer ``_param_map`` for an image family."""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--family", required=True, help="Registry family id (e.g. qwen_image, flux2)")
    args = ap.parse_args()
    family = args.family

    from backend.engine._transformer_registry import get_transformer_class, get_weight_remap
    from backend.engine.config.model_configs import get_config_class

    config_cls = get_config_class(family)
    config = config_cls()
    trans_cls = get_transformer_class(family)

    class _NullCtx:
        backend = "mlx"

        def bfloat16(self):
            return "bf16"

        def float32(self):
            return "f32"

    model = trans_cls(config, _NullCtx())
    param_keys = set(model.parameters().keys()) if hasattr(model, "parameters") else set()
    if not param_keys and hasattr(model, "_param_map"):
        param_keys = set(getattr(model, "_param_map", {}).keys())

    remap_fn = get_weight_remap(family)
    if remap_fn is None:
        print(f"No weight remap registered for family={family!r}", file=sys.stderr)
        return 1

    # Dry-run remap identity on param keys (validates remap accepts model key space shape).
    sample = {k: None for k in sorted(param_keys)}
    if not sample:
        print(f"WARN: empty param map for {family}; load a checkpoint to validate remap keys")
        return 0

    try:
        remapped = remap_fn(sample)
    except Exception as exc:
        print(f"FAIL: remap raised: {exc}", file=sys.stderr)
        return 1

    remap_keys = set(remapped.keys())
    missing_in_remap = param_keys - remap_keys
    extra_in_remap = remap_keys - param_keys
    if missing_in_remap or extra_in_remap:
        print(f"FAIL: key parity mismatch for family={family}", file=sys.stderr)
        if missing_in_remap:
            print(f"  missing in remap output ({len(missing_in_remap)}):", file=sys.stderr)
            for k in sorted(missing_in_remap)[:20]:
                print(f"    - {k}", file=sys.stderr)
        if extra_in_remap:
            print(f"  extra in remap output ({len(extra_in_remap)}):", file=sys.stderr)
            for k in sorted(extra_in_remap)[:20]:
                print(f"    + {k}", file=sys.stderr)
        return 1

    print(f"OK: {family} remap key parity ({len(param_keys)} keys)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
