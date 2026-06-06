#!/usr/bin/env python3
"""Prune PyInstaller ``full`` (CUDA) sidecar bloat — no MLX, slim optional deps."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import out_paths as op  # noqa: E402
from prune_sidecar import _rm, _TRANSFORMERS_MODEL_KEEP  # noqa: E402

# Trees never needed for CUDA image inference desktop/server bundles.
_REMOVE_TREE_NAMES = (
    "mlx",
    "mlx_lm",
    "hf_xet",
    "cv2",
    "opencv_python",
    "pyarrow",
    "pandas",
    "matplotlib",
    "scipy",
    "sklearn",
    "accelerate",
    "bitsandbytes",
    "tensorboard",
    "tensorboard_data_server",
    "torchaudio",
    "torchvision",
    "functorch",
    "triton",
)

_REMOVE_DIST_INFO_PREFIXES = (
    "mlx",
    "opencv",
    "scipy",
    "pandas",
    "matplotlib",
    "pyarrow",
    "cv2",
    "torchaudio",
    "torchvision",
    "tensorboard",
    "pip-",
    "setuptools-",
    "wheel-",
)

# Drop under _internal when present (PyInstaller collected but unused on CUDA path).
_REMOVE_INTERNAL_GLOBS = (
    "backend/engine/runtime/mlx",
    "backend/engine/families/seedvr2",
    "backend/engine/families/ace_step",
    "backend/engine/pipelines/music_pipeline",
)


def prune_cuda_sidecar(sidecar_dir: Path, *, slim_transformers: bool = True) -> list[str]:
    internal = sidecar_dir / "_internal"
    if not internal.is_dir():
        raise SystemExit(f"Not a PyInstaller onedir: {sidecar_dir}")

    removed: list[str] = []

    for name in _REMOVE_TREE_NAMES:
        target = internal / name
        if _rm(target):
            removed.append(name)

    for rel in _REMOVE_INTERNAL_GLOBS:
        target = internal / rel
        if _rm(target):
            removed.append(rel.replace("/", "."))

    for mlx_py in sorted(internal.rglob("*_mlx.py")):
        rel = mlx_py.relative_to(internal)
        if _rm(mlx_py):
            removed.append(str(rel).replace("\\", "/"))

    for entry in internal.iterdir():
        if not entry.is_dir() or not entry.name.endswith(".dist-info"):
            continue
        if any(entry.name.startswith(p) for p in _REMOVE_DIST_INFO_PREFIXES):
            if _rm(entry):
                removed.append(entry.name)

    if slim_transformers:
        models_dir = internal / "transformers" / "models"
        if models_dir.is_dir():
            for child in models_dir.iterdir():
                if child.is_dir() and child.name not in _TRANSFORMERS_MODEL_KEEP:
                    if _rm(child):
                        removed.append(f"transformers.models.{child.name}")

    return removed


def finalize_cuda_sidecar(sidecar_dir: Path, *, slim_transformers: bool = True) -> list[str]:
    return prune_cuda_sidecar(sidecar_dir, slim_transformers=slim_transformers)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prune CUDA desktop/server sidecar bloat")
    parser.add_argument(
        "--sidecar-dir",
        type=Path,
        default=op.SIDECAR_DIR,
        help=f"PyInstaller onedir (default: {op.SIDECAR_DIR})",
    )
    parser.add_argument(
        "--no-slim-transformers",
        action="store_true",
        help="Keep full transformers.models (larger bundle)",
    )
    args = parser.parse_args()

    before = sum(f.stat().st_size for f in args.sidecar_dir.rglob("*") if f.is_file())
    removed = finalize_cuda_sidecar(args.sidecar_dir, slim_transformers=not args.no_slim_transformers)
    after = sum(f.stat().st_size for f in args.sidecar_dir.rglob("*") if f.is_file())

    if not removed:
        print("Nothing pruned.")
        return
    print(f"Pruned {len(removed)} entries from {args.sidecar_dir}")
    print(f"  size: {before / 1e6:.1f} MB -> {after / 1e6:.1f} MB  (saved {(before - after) / 1e6:.1f} MB)")
    for name in removed[:24]:
        print(f"  - {name}")
    if len(removed) > 24:
        print(f"  ... and {len(removed) - 24} more")


if __name__ == "__main__":
    main()
