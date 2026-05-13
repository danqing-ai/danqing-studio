"""Flux.1 权重键映射 — diffusers ``FluxTransformer2DModel`` → DanQing 模块命名。"""
from __future__ import annotations


def remap_flux1_weights(weights: dict) -> dict:
    """Normalize checkpoint keys for ``Flux1Transformer``.

    Covers ``ff.net.*``, ``to_out.0``, and common ``time_text_embed`` / pooled projection prefixes.
    """
    remapped = {}
    for key, tensor in weights.items():
        new_key = key
        new_key = new_key.replace(".to_out.0.", ".to_out.")
        new_key = new_key.replace(".ff.net.0.proj.", ".ff.net_0_proj.")
        new_key = new_key.replace(".ff.net.2.", ".ff.net_2.")
        new_key = new_key.replace(".ff_context.net.0.proj.", ".ff_context.net_0_proj.")
        new_key = new_key.replace(".ff_context.net.2.", ".ff_context.net_2.")
        # Combined timestep + pooled (diffusers ``time_text_embed``)
        new_key = new_key.replace("time_text_embed.timestep_embedder.linear_1", "time_in.mlp.0")
        new_key = new_key.replace("time_text_embed.timestep_embedder.linear_2", "time_in.mlp.2")
        new_key = new_key.replace("time_text_embed.text_proj.", "vector_in.")
        remapped[new_key] = tensor
    return remapped
