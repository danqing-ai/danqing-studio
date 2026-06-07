"""ERNIE-Image weight remapping — diffusers → DanQing flat keys."""
from __future__ import annotations

from typing import Any


def remap_ernie_image_weights(weights: dict[str, Any]) -> dict[str, Any]:
    """Map diffusers ERNIE-Image transformer keys to ``ErnieImageTransformer`` param map."""
    remapped: dict[str, Any] = {}
    for key, tensor in weights.items():
        new_key = key
        for prefix in ("transformer.", "model."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix) :]
        new_key = new_key.replace(".to_out.0.", ".to_out_0.")
        new_key = new_key.replace("adaLN_modulation.1.", "adaLN_modulation.linear.")
        if new_key.startswith("time_proj."):
            continue
        if new_key == "x_embedder.proj.weight" and hasattr(tensor, "ndim") and int(tensor.ndim) == 4:
            tensor = tensor.squeeze(-1).squeeze(-1)
        remapped[new_key] = tensor
    return remapped
