"""CogView4 weight remapping — diffusers → flat MLX keys."""
from __future__ import annotations

from typing import Any


def remap_cogview4_weights(weights: dict[str, Any]) -> dict[str, Any]:
    """Strip diffusers prefixes; keys align with ``CogView4DiTMLX`` param map."""
    remapped: dict[str, Any] = {}
    for key, tensor in weights.items():
        new_key = key
        for prefix in ("transformer.", "model."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix) :]
        remapped[new_key] = tensor
    return remapped
