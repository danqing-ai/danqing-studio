"""Z-Image 权重映射。"""
from __future__ import annotations
def remap_zimage_weights(weights: dict, patch_size: int = 2) -> dict:
    """将 diffusers 格式 Z-Image 权重键映射为 DanQing 引擎键名。

    源格式 (mflux/diffusers):
        all_x_embedder.2-1.weight
        all_final_layer.2-1.adaLN_modulation.1.bias
        t_embedder.mlp.0.weight
        layers.0.adaLN_modulation.0.weight
        layers.0.attention.to_out.0.weight
        cap_embedder.0.weight  (RMSNorm)
        cap_embedder.1.bias    (Linear)
        context_refiner.0.attention_norm1.weight

    目标格式 (DanQing):
        x_embedder.weight
        final_layer.adaLN_modulation.bias
        t_embedder.linear1.weight
        layers.0.adaLN_modulation.weight
        layers.0.attention.to_out.weight
        cap_norm.weight
        cap_embedder.bias
        context_refiner.0.attn_norm1.weight
    """
    prefix_key = f"{patch_size}-1"
    remapped = {}

    for key, tensor in weights.items():
        new_key = key

        # all_x_embedder.{patch}-1 → x_embedder
        new_key = new_key.replace(f"all_x_embedder.{prefix_key}", "x_embedder")
        # all_final_layer.{patch}-1 → final_layer
        new_key = new_key.replace(f"all_final_layer.{prefix_key}", "final_layer")

        # t_embedder.mlp.0 → t_embedder.linear1
        new_key = new_key.replace("t_embedder.mlp.0.", "t_embedder.linear1.")
        # t_embedder.mlp.2 → t_embedder.linear2
        new_key = new_key.replace("t_embedder.mlp.2.", "t_embedder.linear2.")

        # attention_norm1 → attn_norm1
        new_key = new_key.replace("attention_norm1.", "attn_norm1.")
        # attention_norm2 → attn_norm2
        new_key = new_key.replace("attention_norm2.", "attn_norm2.")

        # .to_out.0 → .to_out (DanQing uses direct Linear, not Sequential/list)
        new_key = new_key.replace(".to_out.0.", ".to_out.")

        # adaLN_modulation: mflux uses list [nn.Linear(...)], DanQing also uses list now
        # Keep .0. index for transformer blocks and noise_refiner/context_refiner
        # For final_layer: diffusers uses .1., mflux maps to .0., keep .0.
        new_key = new_key.replace(".adaLN_modulation.1.", ".adaLN_modulation.0.")

        # cap_embedder.0 → cap_norm (RMSNorm)
        new_key = new_key.replace("cap_embedder.0.", "cap_norm.")
        # cap_embedder.1 → cap_embedder (Linear)
        new_key = new_key.replace("cap_embedder.1.", "cap_embedder.")

        remapped[new_key] = tensor

    return remapped

