"""Post-download hook: convert PyTorch HeartMuLa weights to MLX once (heartlib-mlx layout)."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from backend.engine.families.heartmula.bundle import (
    mlx_weights_path,
    mlx_weights_ready,
    resolve_heartmula_bundle,
)
from backend.engine.families.heartmula.weights_mlx import convert_pytorch_to_mlx

logger = logging.getLogger(__name__)

HOOK_TYPE = "heartmula_mlx_weights"


def prune_pytorch_weights(component_dir: Path) -> list[str]:
    """Drop upstream PyTorch weight files after ``mlx/model.safetensors`` exists.

    Keeps ``config.json``, tokenizer copies, and the MLX cache. Only removes
    top-level weight blobs in the component directory (not under ``mlx/``).
    """
    component_dir = Path(component_dir)
    if not mlx_weights_path(component_dir).is_file():
        return []

    removed: list[str] = []
    for entry in component_dir.iterdir():
        if not entry.is_file():
            continue
        name = entry.name
        if name.startswith("pytorch_model") and name.endswith(".bin"):
            entry.unlink()
            removed.append(name)
            continue
        if name.endswith(".safetensors"):
            entry.unlink()
            removed.append(name)
            continue
        if name == "model.safetensors.index.json":
            entry.unlink()
            removed.append(name)
            continue
        if name.endswith(".pt") or name.endswith(".pth"):
            entry.unlink()
            removed.append(name)
    return removed


def _convert_component(src_dir: Path, *, model_type: str, dtype: str) -> None:
    out_dir = src_dir / "mlx"
    out_path = mlx_weights_path(src_dir)
    if out_path.is_file():
        logger.info("HeartMuLa MLX weights already present: %s", out_path)
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    convert_pytorch_to_mlx(
        src_path=src_dir,
        dst_path=out_dir,
        model_type=model_type,
        dtype=dtype,
    )
    if not out_path.is_file():
        raise RuntimeError(
            f"HeartMuLa {model_type} MLX conversion did not write {out_path}"
        )


def run_heartmula_mlx_weights(
    *,
    bundle_root: Path,
    model_name: str,
    version_key: str | None,
    hook_spec: dict[str, Any],
) -> None:
    """Install hook entrypoint (registered as ``heartmula_mlx_weights``)."""
    del model_name, version_key
    root = Path(bundle_root)
    paths = resolve_heartmula_bundle(root)

    if not mlx_weights_ready(root):
        dtype = str(hook_spec.get("dtype") or "bfloat16").strip().lower()
        if dtype not in ("float32", "float16", "bfloat16"):
            raise ValueError(
                f"heartmula_mlx_weights: unsupported dtype {dtype!r} "
                "(use float32, float16, or bfloat16)"
            )
        logger.info("HeartMuLa MLX hook: converting LM at %s", paths.mula_torch)
        _convert_component(paths.mula_torch, model_type="heartmula", dtype=dtype)
        logger.info("HeartMuLa MLX hook: converting Codec at %s", paths.codec_torch)
        _convert_component(paths.codec_torch, model_type="heartcodec", dtype=dtype)
        if not mlx_weights_ready(root):
            raise RuntimeError(f"HeartMuLa MLX hook finished but cache incomplete: {root}")
    else:
        logger.info("HeartMuLa MLX hook: conversion skipped (cache present): %s", root)

    for label, component in (
        ("LM", paths.mula_torch),
        ("Codec", paths.codec_torch),
    ):
        pruned = prune_pytorch_weights(component)
        if pruned:
            logger.info(
                "HeartMuLa MLX hook: pruned PyTorch weights from %s (%s): %s",
                label,
                component,
                ", ".join(pruned),
            )

    logger.info("HeartMuLa MLX hook complete for %s", root)
