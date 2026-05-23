"""Apply Qwen VAE weights from a local bundle using repo-local mapping (no external deps)."""
from __future__ import annotations

import importlib
from pathlib import Path

import mlx.core as mx


def apply_qwen_vae_weights_from_bundle(
    vae,
    bundle_root: Path,
    *,
    project_root: Path,
    eval_fn=None,
) -> None:
    del project_root
    vae_dir = bundle_root / "vae"
    if not vae_dir.is_dir():
        raise RuntimeError(f"Qwen Image VAE: missing directory {vae_dir}")

    from backend.engine.families.qwen.weights import apply_qwen_vae_weights

    raw: dict = {}
    for sf in sorted(vae_dir.glob("*.safetensors")):
        raw.update(dict(mx.load(str(sf))))
    nested = apply_qwen_vae_weights(raw)
    vae.update(nested)
    if eval_fn is not None:
        eval_fn(vae)
    else:
        importlib.import_module("mlx.core").eval(vae)
