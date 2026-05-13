"""Wan 视频模型权重映射 — diffusers → DanQing 键名转换。"""
from __future__ import annotations


def remap_wan_weights(weights: dict) -> dict:
    """将 diffusers Wan 权重键映射为 DanQing 引擎键名。

    DanQing WanTransformer 使用与 diffusers 兼容的键名结构，仅需处理：
    - patch_embedding → patch_embed.proj
    - time_embedding.linear_* → time_embed.mlp.layers.*
    - ffn.net.N.proj → mlp.layers.N (移除 Sequential 包装)
    - norm → final_norm
    - head → proj_out
    """
    remapped = {}
    for key, tensor in weights.items():
        new_key = key

        # patch_embedding.weight/bias → patch_embed.proj.weight/bias
        new_key = new_key.replace("patch_embedding.", "patch_embed.proj.")

        # time_embedding.linear_1.* → time_embed.mlp.layers.0.*
        new_key = new_key.replace("time_embedding.linear_1.", "time_embed.mlp.layers.0.")
        # time_embedding.linear_2.* → time_embed.mlp.layers.2.*
        new_key = new_key.replace("time_embedding.linear_2.", "time_embed.mlp.layers.2.")

        # ffn.net.0.proj → mlp.layers.0
        new_key = new_key.replace(".ffn.net.0.proj.", ".mlp.layers.0.")
        # ffn.net.2 → mlp.layers.2
        new_key = new_key.replace(".ffn.net.2.", ".mlp.layers.2.")

        # norm.weight → final_norm.weight
        new_key = new_key.replace("norm.weight", "final_norm.weight")

        # head.weight/bias → proj_out.weight/bias
        new_key = new_key.replace("head.", "proj_out.")

        remapped[new_key] = tensor
    return remapped
