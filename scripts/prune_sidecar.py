#!/usr/bin/env python3
"""Remove PyInstaller bloat from MLX desktop sidecar (post-build)."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import out_paths as op  # noqa: E402

# Tokenizer / text-encoder architectures used by DanQing MLX paths (fail loud if a new model needs another).
# ``auto`` + ``encoder_decoder`` are required for ``from transformers import AutoTokenizer``.
_TRANSFORMERS_MODEL_KEEP = frozenset(
    {
        "auto",
        "encoder_decoder",
        "t5",
        "clip",
        "qwen2",
        "qwen2_vl",
        "qwen2_5_vl",
        "mistral",
        "llama",
        "gemma",
        "gemma2",
        "gpt2",
        "bart",
        "roberta",
        "xlm_roberta",
        "phi",
        "qwen3",
    }
)

_REMOVE_TREE_NAMES = (
    "hf_xet",
    "torch",
    "torchvision",
    "torchaudio",
    "cv2",
    "pyarrow",
    "pandas",
    "matplotlib",
    "scipy",
    "sklearn",
    "accelerate",
    "bitsandbytes",
    "tensorboard",
)

# MLX loads default.metallib from the executable directory; DYLD_LIBRARY_PATH in a Python hook is too late.
_MLX_RUNTIME_ROOT_FILES = ("mlx.metallib", "libmlx.dylib", "libjaccl.dylib")


def layout_mlx_runtime(sidecar_dir: Path) -> list[str]:
    mlx_lib = sidecar_dir / "_internal" / "mlx" / "lib"
    if not mlx_lib.is_dir():
        raise SystemExit(f"Missing MLX lib dir: {mlx_lib}")

    placed: list[str] = []
    for name in _MLX_RUNTIME_ROOT_FILES:
        src = mlx_lib / name
        if not src.is_file():
            raise SystemExit(f"Missing MLX bundle file: {src}")
        dst = sidecar_dir / name
        shutil.copy2(src, dst)
        if name.endswith(".dylib"):
            dst.chmod(0o755)
        placed.append(name)
    return placed


_REMOVE_DIST_INFO_PREFIXES = (
    "torch",
    "torchvision",
    "torchaudio",
    "opencv",
    "scipy",
    "pandas",
    "matplotlib",
    "pyarrow",
    "cv2",
    "pip-",
    "setuptools-",
    "wheel-",
)


def _rm(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()
    return 1


def prune_sidecar(sidecar_dir: Path, *, slim_transformers: bool = True) -> list[str]:
    internal = sidecar_dir / "_internal"
    if not internal.is_dir():
        raise SystemExit(f"Not a PyInstaller onedir: {sidecar_dir}")

    removed: list[str] = []

    for name in _REMOVE_TREE_NAMES:
        target = internal / name
        if _rm(target):
            removed.append(name)

    for entry in internal.iterdir():
        if not entry.is_dir() or not entry.name.endswith(".dist-info"):
            continue
        base = entry.name.split("-")[0]
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


def finalize_mlx_sidecar(sidecar_dir: Path, *, slim_transformers: bool = True) -> tuple[list[str], list[str]]:
    removed = prune_sidecar(sidecar_dir, slim_transformers=slim_transformers)
    placed = layout_mlx_runtime(sidecar_dir)
    return removed, placed


def main() -> None:
    parser = argparse.ArgumentParser(description="Prune MLX desktop sidecar bloat")
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
    removed, placed = finalize_mlx_sidecar(
        args.sidecar_dir, slim_transformers=not args.no_slim_transformers
    )
    after = sum(f.stat().st_size for f in args.sidecar_dir.rglob("*") if f.is_file())

    if placed:
        print(f"MLX runtime files at sidecar root: {', '.join(placed)}")
    if not removed:
        print("Nothing pruned.")
        if not placed:
            return
    print(f"Pruned {len(removed)} entries from {args.sidecar_dir}")
    print(f"  size: {before / 1e6:.1f} MB -> {after / 1e6:.1f} MB  (saved {(before - after) / 1e6:.1f} MB)")
    for name in removed[:20]:
        print(f"  - {name}")
    if len(removed) > 20:
        print(f"  ... and {len(removed) - 20} more")


if __name__ == "__main__":
    main()
