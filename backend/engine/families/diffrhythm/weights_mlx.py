"""DiffRhythm 2 safetensors → MLX CFM weight loading."""
from __future__ import annotations

from typing import Any, List, Tuple

from backend.engine.families.diffrhythm.weights import remap_diffrhythm_weights


def load_diffrhythm_safetensors_for_mlx(
    safetensors_path: str, *, array_fn: Any | None = None
) -> List[Tuple[str, Any]]:
    """Load DiffRhythm 2 CFM weights from safetensors into MLX arrays.

    Args:
        safetensors_path: Path to model.safetensors
        array_fn: Optional array factory (defaults to mx.array)

    Returns:
        List of (key, mlx_array) tuples ready for ``load_weights``.
    """
    import mlx.core as mx
    from safetensors import safe_open

    if array_fn is None:
        array_fn = mx.array

    raw_weights: List[Tuple[str, Any]] = []
    with safe_open(safetensors_path, framework="pt", device="cpu") as handle:
        for key in handle.keys():
            tensor = handle.get_tensor(key).detach().cpu().float().numpy()
            raw_weights.append((key, array_fn(tensor)))

    return remap_diffrhythm_weights(raw_weights)
