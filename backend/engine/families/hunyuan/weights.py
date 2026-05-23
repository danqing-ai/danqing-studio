"""HunyuanVideo-1.5 DiT weight remap — diffusers → flat ``HunyuanVideoTransformer`` keys."""
from __future__ import annotations


def remap_hunyuan_weights(weights: dict) -> dict:
    """Strip common checkpoint prefixes; remap Conv3d patch weights to MLX NHWC layout."""
    import importlib

    mx = importlib.import_module("mlx.core")

    def _conv3d_torch_to_mlx(w):
        if not isinstance(w, mx.array) or w.ndim != 5:
            return w
        # PyTorch Conv3d [O, I, T, H, W] → MLX [O, T, H, W, I]
        return mx.transpose(w, (0, 2, 3, 4, 1))

    remapped: dict = {}
    for key, tensor in weights.items():
        new_key = key
        for prefix in ("transformer.", "model.transformer.", "module."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
                break
        new_key = new_key.replace(".default.", ".")
        if ".lora_" in new_key or new_key.startswith("lora_"):
            continue
        if new_key == "x_embedder.proj.weight":
            tensor = _conv3d_torch_to_mlx(tensor)
        remapped[new_key] = tensor
    return remapped
