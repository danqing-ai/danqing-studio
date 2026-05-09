"""LongCat-Image 权重映射。"""
from __future__ import annotations
def remap_longcat_weights(weights: dict) -> dict:
    """将 diffusers LongCat-Image 权重键映射为 DanQing 引擎键名。
    
    DanQing 实现使用与 diffusers 一致的键名，仅需映射：
    - to_out.0 → to_out (DanQing 使用独立 Linear，非 Sequential)
    - ff.net.0.proj → ff.net_0_proj (避免 MLX Sequential 命名差异)
    - ff.net.2 → ff.net_2
    - ff_context.net.0.proj → ff_context.net_0_proj
    - ff_context.net.2 → ff_context.net_2
    - time_embed.timestep_embedder → time_embed (去除 diffusers 包装层)
    
    注意：norm1/norm1_context 的 AdaLN 权重键名与 diffusers 100% 一致，
    无需变换（diffusers: norm1.linear.weight → DanQing: norm1.linear.weight）。
    """
    remapped = {}
    for key, tensor in weights.items():
        new_key = key
        # 时间嵌入: timestep_embedder 包装层移除
        new_key = new_key.replace("time_embed.timestep_embedder.", "time_embed.")
        # 注意力输出: to_out.0 → to_out
        new_key = new_key.replace(".to_out.0.", ".to_out.")
        new_key = new_key.replace(".to_out.0_weight", ".to_out_weight")
        new_key = new_key.replace(".to_out.0_bias", ".to_out_bias")
        # FFN: net.0.proj → net_0_proj, net.2 → net_2
        new_key = new_key.replace(".ff.net.0.proj.", ".ff.net_0_proj.")
        new_key = new_key.replace(".ff.net.2.", ".ff.net_2.")
        new_key = new_key.replace(".ff_context.net.0.proj.", ".ff_context.net_0_proj.")
        new_key = new_key.replace(".ff_context.net.2.", ".ff_context.net_2.")
        remapped[new_key] = tensor
    return remapped
