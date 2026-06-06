"""
DiffRhythm 2 weight remap — CFM checkpoint keys for MLX ``load_cfm_weights``.

Checkpoint keys are typically prefixed with ``transformer.`` (inner DiT inside CFM).
"""
from __future__ import annotations

from typing import Any, List, Tuple


def remap_diffrhythm_weights(
    raw_weights: List[Tuple[str, Any]],
) -> List[Tuple[str, Any]]:
    """Pass-through with optional prefix normalization for CFM safetensors."""
    remapped: List[Tuple[str, Any]] = []
    for key, tensor in raw_weights:
        new_key = key
        if new_key.startswith("module."):
            new_key = new_key[len("module.") :]
        remapped.append((new_key, tensor))
    return remapped
