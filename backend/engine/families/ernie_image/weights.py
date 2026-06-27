"""ERNIE-Image weight remapping — diffusers → DanQing flat keys."""
from __future__ import annotations

from typing import Any


def _squeeze_x_embedder_proj_weight(tensor: Any) -> Any:
    """MLX-community ERNIE bundles use (out, 1, 1, in) conv weights for patch embed."""
    if not hasattr(tensor, "ndim") or int(tensor.ndim) != 4:
        return tensor
    if int(tensor.shape[1]) == 1 and int(tensor.shape[2]) == 1:
        return tensor.squeeze(axis=(1, 2))
    return tensor.reshape(int(tensor.shape[0]), -1)


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
        if new_key.startswith("adaln_modulation."):
            new_key = "adaLN_modulation.linear." + new_key[len("adaln_modulation.") :]
        if new_key.startswith("time_proj."):
            continue
        if new_key == "x_embedder.proj.weight" and hasattr(tensor, "ndim") and int(tensor.ndim) == 4:
            tensor = _squeeze_x_embedder_proj_weight(tensor)
        remapped[new_key] = tensor
    return remapped
