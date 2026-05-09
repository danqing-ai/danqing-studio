"""VAE 权重映射。"""
from __future__ import annotations
def remap_vae_weights(weights: dict) -> dict:
    """将 diffusers 格式 VAE 权重键映射为 DanQing VAEDecoder 键名。

    diffusers 格式:
        decoder.conv_in.weight (shape: [O, kH, kW, I])
        decoder.mid_block.resnets.0.norm1.weight
        decoder.up_blocks.3.resnets.2.conv2.weight

    DanQing 格式:
        conv_in.weight (shape: [O, I, kH, kW] — MLX Conv2d)
        mid_resnet1.norm1.weight
        up4_resnets.2.conv2.weight
    """
    remapped = {}
    up_block_map = {0: "up1", 1: "up2", 2: "up3", 3: "up4"}

    for key, tensor in weights.items():
        new_key = key

        # 跳过 encoder 部分
        if "encoder." in new_key or "quant_conv" in new_key or "post_quant" in new_key:
            continue

        # Strip decoder. 前缀
        new_key = new_key.replace("decoder.", "")

        # mid_block.resnets.0 → mid_resnet1, .1 → mid_resnet2
        new_key = new_key.replace("mid_block.resnets.0", "mid_resnet1")
        new_key = new_key.replace("mid_block.resnets.1", "mid_resnet2")

        # up_blocks.{i}.resnets → up{i+1}_resnets
        for i, name in up_block_map.items():
            new_key = new_key.replace(f"up_blocks.{i}.resnets", f"{name}_resnets")
            new_key = new_key.replace(f"up_blocks.{i}.upsamplers.0", f"{name}_up")

        # conv_norm_out → norm_out
        new_key = new_key.replace("conv_norm_out", "norm_out")

        # mid_block.attentions.0 → mid_attn
        new_key = new_key.replace("mid_block.attentions.0.group_norm", "mid_attn.norm")
        new_key = new_key.replace("mid_block.attentions.0.to_q", "mid_attn.to_q")
        new_key = new_key.replace("mid_block.attentions.0.to_k", "mid_attn.to_k")
        new_key = new_key.replace("mid_block.attentions.0.to_v", "mid_attn.to_v")
        new_key = new_key.replace("mid_block.attentions.0.to_out.0", "mid_attn.to_out")

        # Conv2d weight: diffusers (O, I, kH, kW) → MLX (O, kH, kW, I)
        if ".weight" in new_key and tensor.ndim == 4:
            tensor = tensor.transpose(0, 2, 3, 1)

        remapped[new_key] = tensor

    return remapped

