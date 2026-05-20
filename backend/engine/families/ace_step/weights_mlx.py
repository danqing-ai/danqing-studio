"""ACE-Step safetensors → MLX weight loading (mlx imports allowed here only)."""
from __future__ import annotations

from typing import Any, List, Tuple

from backend.engine.families.ace_step.weights import _convert_decoder_tensor_for_mlx


def load_prefix_weights_for_mlx(
    safetensors_path: str,
    prefix: str,
    *,
    strip_prefix: bool = True,
) -> List[Tuple[str, Any]]:
    import mlx.core as mx
    from safetensors import safe_open

    weights: List[Tuple[str, Any]] = []
    with safe_open(safetensors_path, framework="pt", device="cpu") as handle:
        for key in handle.keys():
            if not key.startswith(prefix):
                continue
            arr = handle.get_tensor(key).detach().cpu().float().numpy()
            mlx_key = key[len(prefix) :] if strip_prefix else key
            weights.append((mlx_key, mx.array(arr)))
    if not weights:
        raise RuntimeError(f"No weights with prefix {prefix!r} in {safetensors_path}")
    return weights


def load_decoder_safetensors_for_mlx(safetensors_path: str) -> List[Tuple[str, Any]]:
    import mlx.core as mx
    from safetensors import safe_open

    weights: List[Tuple[str, Any]] = []
    with safe_open(safetensors_path, framework="pt", device="cpu") as handle:
        for key in handle.keys():
            if not key.startswith("decoder.") or "rotary_emb" in key:
                continue
            tensor = handle.get_tensor(key).detach().cpu().float().numpy()
            mlx_key, np_val = _convert_decoder_tensor_for_mlx(key, tensor)
            weights.append((mlx_key, mx.array(np_val)))
    return weights
