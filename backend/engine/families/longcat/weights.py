"""LongCat-Image 权重映射。"""
from __future__ import annotations


def remap_longcat_weights(weights: dict) -> dict:
    """Map diffusers LongCat-Image keys to DanQing flat ``_param_map`` keys."""
    remapped = {}
    for key, tensor in weights.items():
        new_key = key
        new_key = new_key.replace("time_embed.timestep_embedder.", "time_embed.")
        new_key = new_key.replace(".to_out.0.", ".to_out.")
        new_key = new_key.replace(".to_out.0_weight", ".to_out_weight")
        new_key = new_key.replace(".to_out.0_bias", ".to_out_bias")
        new_key = new_key.replace(".ff.net.0.proj.", ".ff.net_0_proj.")
        new_key = new_key.replace(".ff.net.2.", ".ff.net_2.")
        new_key = new_key.replace(".ff_context.net.0.proj.", ".ff_context.net_0_proj.")
        new_key = new_key.replace(".ff_context.net.2.", ".ff_context.net_2.")
        remapped[new_key] = tensor
    return remapped
