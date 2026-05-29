#!/usr/bin/env python3
"""Scaffold a Shape-A (DiT + ImagePipeline) image family directory."""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = ROOT / "backend" / "engine" / "families"


def _snake(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", s.strip())
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)
    return s.strip("_").lower()


def _pascal(s: str) -> str:
    return "".join(part.capitalize() for part in _snake(s).split("_") if part)


def _write(path: Path, content: str, *, force: bool) -> None:
    if path.exists() and not force:
        print(f"skip existing {path.relative_to(ROOT)}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    print(f"wrote {path.relative_to(ROOT)}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--family", required=True, help="Registry family id (e.g. my_family)")
    ap.add_argument("--class", dest="class_name", default="", help="Transformer class name")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    family = _snake(args.family)
    cls_name = args.class_name or _pascal(family) + "Transformer"
    family_dir = FAMILIES / family
    if family_dir.exists() and not args.force:
        print(f"FAIL: {family_dir} already exists (use --force)", file=sys.stderr)
        return 1

    _write(
        family_dir / "transformer.py",
        f'''"""{cls_name} — public stem; MLX in ``transformer_mlx``."""
from __future__ import annotations

from typing import Any

from backend.engine.common.dit_stem import DelegatingDiTStem


class {cls_name}(DelegatingDiTStem):
    def __init__(self, config: Any, ctx: Any):
        from .transformer_mlx import {cls_name} as _MLX

        super().__init__(config, ctx, mlx_cls=_MLX, unavailable_product="{cls_name}")
''',
        force=args.force,
    )
    _write(
        family_dir / "transformer_mlx.py",
        f'''"""{cls_name} MLX implementation."""
from __future__ import annotations

from typing import Any

from backend.engine.common._base import TransformerBase


class {cls_name}(TransformerBase):
    def __init__(self, config: Any, ctx: Any):
        super().__init__(config, ctx)
        self._param_map: dict[str, Any] = {{}}

    def forward(self, latents, t, *, txt_embeds=None, **kwargs):
        raise NotImplementedError("{cls_name}.forward is not implemented yet")

    def parameters(self):
        return self._param_map
''',
        force=args.force,
    )
    _write(
        family_dir / "weights.py",
        f'''"""Weight remap for {family}."""
from __future__ import annotations


def remap_{family}_weights(weights: dict) -> dict:
    remapped = {{}}
    for key, tensor in weights.items():
        remapped[key] = tensor
    return remapped
''',
        force=args.force,
    )

    print("\nRegistry snippets (manual):")
    print(f'  _TRANSFORMER["{family}"] = ("backend.engine.families.{family}.transformer", "{cls_name}")')
    print(f'  _WEIGHT_REMAP["{family}"] = ("backend.engine.families.{family}.weights", "remap_{family}_weights")')
    print(f'  bundle_manifest.FAMILY_BUNDLE_CONTRACTS: add "{family}" required components')
    print("\nNext: model_configs dataclass + FAMILY_CONFIG_MAP + models_registry.json entry")
    print("See docs/engine_new_model_checklist.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
