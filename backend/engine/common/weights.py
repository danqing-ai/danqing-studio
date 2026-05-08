"""
权重工具 — safetensors 加载 / LoRA 注入 / 量化。

所有后端共用。
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable, Optional


def parse_size_gb(size_str: str) -> float:
    """解析 size 字段，如 '23.8GB' → 23.8。"""
    match = re.match(r"([\d.]+)\s*(GB|MB)", str(size_str).strip(), re.IGNORECASE)
    if not match:
        return 0.0
    val = float(match.group(1))
    unit = match.group(2).upper()
    return val if unit == "GB" else val / 1024


def load_safetensors(path: str, ctx: Any = None) -> dict:
    """加载 safetensors 权重文件。

    MLX: 使用 mx.load()
    CUDA: 使用 safetensors.torch.load_file()
    """
    if ctx is not None:
        return ctx.load_weights(path)
    # 默认行为：尝试 MLX
    try:
        import mlx.core as mx
        return dict(mx.load(path))
    except ImportError:
        pass
    try:
        import safetensors.torch
        return safetensors.torch.load_file(path, device="cpu")
    except ImportError:
        raise RuntimeError("No safetensors loader available (install mlx or safetensors)")


def save_safetensors(weights: dict, path: str, ctx: Any = None) -> None:
    """保存 safetensors 权重。"""
    if ctx is not None:
        ctx.save_weights(weights, path)
        return
    try:
        import mlx.core as mx
        mx.save_safetensors(path, weights)
        return
    except ImportError:
        pass
    try:
        import safetensors.torch
        safetensors.torch.save_file(weights, path)
    except ImportError:
        raise RuntimeError("No safetensors saver available")


class LoRAConfig:
    """单个 LoRA 配置。"""

    def __init__(self, path: str, strength: float = 1.0,
                 target_modules: Optional[list[str]] = None):
        self.path = path
        self.strength = strength
        self.target_modules = target_modules


def _lora_key_to_module(key: str) -> str:
    """从 LoRA 键中提取目标模块名。

    lora_A / lora_down → 目标层，例如:
    "transformer.single_blocks.0.attn.to_q.lora_A" → "transformer.single_blocks.0.attn.to_q"
    """
    for suffix in (".lora_A", ".lora_B", ".lora_down", ".lora_up",
                    "lora_A", "lora_B", "lora_down", "lora_up"):
        if suffix in key:
            return key.replace(suffix, "").rstrip(".")
    return key


def load_lora_weights(lora_path: str) -> dict[str, tuple[Any, Any, float]]:
    """加载 LoRA 权重，返回 {module_name: (lora_A, lora_B, alpha)}。

    LoRA 文件通常包含 lora_down / lora_up 或 lora_A / lora_B 键对。
    """
    weights = load_safetensors(lora_path)
    # 尝试读取 alpha / rank 元数据
    lora_config = {}
    config_path = Path(lora_path).parent / "lora_config.json"
    if not config_path.exists():
        config_path = Path(lora_path).with_suffix(".json")
    if config_path.exists():
        with open(config_path) as f:
            lora_config = json.load(f)

    default_alpha = lora_config.get("lora_alpha", lora_config.get("alpha", 8))
    groups: dict[str, dict[str, Any]] = {}

    for key, tensor in weights.items():
        module = _lora_key_to_module(key)
        if module not in groups:
            groups[module] = {}
        if "lora_down" in key or "lora_A" in key or "lora_A." in key:
            groups[module]["down"] = tensor
        elif "lora_up" in key or "lora_B" in key or "lora_B." in key:
            groups[module]["up"] = tensor
        elif "alpha" in key.lower():
            groups[module]["alpha"] = float(tensor.item() if hasattr(tensor, "item") else tensor)

    result = {}
    for module, parts in groups.items():
        if "down" in parts and "up" in parts:
            alpha = parts.get("alpha", default_alpha)
            result[module] = (parts["down"], parts["up"], alpha)
    return result


def inject_lora(weights: dict, lora_configs: list[LoRAConfig]) -> dict:
    """将多个 LoRA 注入权重 dict (weight merge)。

    返回修改后的权重副本。
    """
    result = dict(weights)
    for cfg in lora_configs:
        loras = load_lora_weights(cfg.path)
        for module_name, (down, up, alpha) in loras.items():
            # 查找对应的线性层权重键
            weight_key = f"{module_name}.weight"
            if weight_key in result:
                orig_weight = result[weight_key]
                # W' = W + (alpha / rank) * strength * (up @ down)
                rank = down.shape[0]
                scale = (alpha / rank) * cfg.strength
                delta = up @ down  # matrix multiply
                result[weight_key] = orig_weight + scale * delta
    return result


def quantize_weights(weights: dict, bits: int = 4,
                     group_size: int = 64) -> dict:
    """简易量化权重：int4/int8 分组量化。

    返回量化后的权重 dict（含 scale + zero_point）。
    """
    if bits not in (4, 8):
        raise ValueError(f"Unsupported bits: {bits}")

    quantized = {}
    for key, tensor in weights.items():
        if tensor.ndim < 2:
            quantized[key] = tensor
            continue
        # 只量化 2D 权重（线性层）
        orig_shape = tensor.shape
        groups = (tensor.shape[-1] + group_size - 1) // group_size
        padded_dim = groups * group_size

        import mlx.core as mx
        if isinstance(tensor, mx.array):
            flat = tensor.reshape(-1, tensor.shape[-1])
        else:
            import numpy as np
            flat = np.array(tensor).reshape(-1, tensor.shape[-1])

        flat_float = flat.astype(float)
        flat_min = flat_float.min(axis=-1, keepdims=True)
        flat_max = flat_float.max(axis=-1, keepdims=True)
        scale_val = (flat_max - flat_min) / ((1 << bits) - 1)
        scale_val = scale_val + 1e-6
        zero_point = (-flat_min / scale_val).round()

        q = ((flat_float / scale_val) + zero_point).round().clip(0, (1 << bits) - 1).astype(int)

        quantized[key] = q
        quantized[f"{key}.scale"] = scale_val.reshape(-1)
        quantized[f"{key}.zero_point"] = zero_point.reshape(-1)

    return quantized


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

        # .to_out.0 → .to_out
        new_key = new_key.replace(".to_out.0.", ".to_out.")

        # .adaLN_modulation.0. → .adaLN_modulation.
        new_key = new_key.replace(".adaLN_modulation.0.", ".adaLN_modulation.")
        # .adaLN_modulation.1. → .adaLN_modulation.  (final_layer case)
        new_key = new_key.replace(".adaLN_modulation.1.", ".adaLN_modulation.")

        # cap_embedder.0 → cap_norm (RMSNorm)
        new_key = new_key.replace("cap_embedder.0.", "cap_norm.")
        # cap_embedder.1 → cap_embedder (Linear)
        new_key = new_key.replace("cap_embedder.1.", "cap_embedder.")

        remapped[new_key] = tensor

    return remapped


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


def remap_flux2_weights(weights: dict) -> dict:
    """将 diffusers Flux.2 Klein 权重键映射为 DanQing 引擎键名。"""
    remapped = {}
    for key, tensor in weights.items():
        new_key = key
        new_key = new_key.replace("single_transformer_blocks", "single_blocks")
        new_key = new_key.replace("transformer_blocks", "joint_blocks")
        new_key = new_key.replace("double_stream_modulation_img", "double_mod_img")
        new_key = new_key.replace("double_stream_modulation_txt", "double_mod_txt")
        new_key = new_key.replace("single_stream_modulation", "single_mod")
        new_key = new_key.replace(".to_qkv_mlp_proj", ".attn_proj")
        new_key = new_key.replace(".to_add_out", ".to_add_out")
        new_key = new_key.replace(".to_out.0", ".to_out")
        new_key = new_key.replace("t_embedder.0.", "t_embedder.layers.0.")
        new_key = new_key.replace("t_embedder.2.", "t_embedder.layers.2.")
        if "norm_added_k" in new_key or "norm_added_q" in new_key:
            continue
        remapped[new_key] = tensor
    return remapped

def remap_longcat_weights(weights: dict) -> dict:
    """将 diffusers LongCat-Image 权重键映射为 DanQing 引擎键名。"""
    remapped = {}
    for key, tensor in weights.items():
        new_key = key
        # x_embedder → x_embedder (same)
        # context_embedder → txt_in
        new_key = new_key.replace("context_embedder", "txt_in")
        # time_embed → time_in
        new_key = new_key.replace("time_embed", "time_in")
        # single_transformer_blocks → single_blocks
        new_key = new_key.replace("single_transformer_blocks", "single_blocks")
        # transformer_blocks → joint_blocks
        new_key = new_key.replace("transformer_blocks", "joint_blocks")
        # norm_out → final_norm
        new_key = new_key.replace("norm_out.linear", "final_norm")
        # Skip norm_out.linear (our final_norm is RMSNorm, diff architecture)
        if "final_norm.weight" in new_key or "final_norm.bias" in new_key:
            # These are the norm_out.linear params, skip for now
            continue
        # proj_mlp + proj_out in single block: keep as-is
        # norm.linear → norm.layers.1 (Sequential)
        new_key = new_key.replace(".norm.linear", ".norm.layers.1")
        # to_out.0 → to_out (LLM style, joint block)
        new_key = new_key.replace(".to_out.0", ".to_out")
        # norm_k/norm_q → stay (SelfAttention handles these)
        # norm_added_k/norm_added_q → skip (not in our impl)
        if "norm_added_k" in new_key or "norm_added_q" in new_key:
            continue
        remapped[new_key] = tensor
    return remapped
