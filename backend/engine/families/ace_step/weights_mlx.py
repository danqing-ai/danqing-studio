"""ACE-Step safetensors → MLX weight loading (mlx imports allowed here only)."""
from __future__ import annotations

from typing import Any, List, Tuple

from backend.engine.common.mlx_runtime_fallback import (
    iter_safetensors_float32_numpy as _iter_safetensors_float32_numpy,
)
from backend.engine.families.ace_step.weights import _convert_decoder_tensor_for_mlx


def load_prefix_weights_for_mlx(
    safetensors_path: str,
    prefix: str,
    *,
    strip_prefix: bool = True,
    array_fn: Any | None = None,
) -> List[Tuple[str, Any]]:
    import mlx.core as mx

    if array_fn is None:
        array_fn = mx.array
    weights: List[Tuple[str, Any]] = []
    for key, arr in _iter_safetensors_float32_numpy(safetensors_path):
        if not key.startswith(prefix):
            continue
        mlx_key = key[len(prefix) :] if strip_prefix else key
        weights.append((mlx_key, array_fn(arr)))
    if not weights:
        raise RuntimeError(f"No weights with prefix {prefix!r} in {safetensors_path}")
    return weights


def load_decoder_safetensors_for_mlx(
    safetensors_path: str, *, array_fn: Any | None = None
) -> List[Tuple[str, Any]]:
    import mlx.core as mx

    if array_fn is None:
        array_fn = mx.array
    weights: List[Tuple[str, Any]] = []
    for key, tensor in _iter_safetensors_float32_numpy(safetensors_path):
        if not key.startswith("decoder.") or "rotary_emb" in key:
            continue
        mlx_key, np_val = _convert_decoder_tensor_for_mlx(key, tensor)
        weights.append((mlx_key, array_fn(np_val)))
    return weights
