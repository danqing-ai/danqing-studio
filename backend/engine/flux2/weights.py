"""Flux2 权重映射 — remap_flux2_weights + remap_flux2_lora_keys。"""
from __future__ import annotations
import re
from typing import Any
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
def remap_flux2_lora_keys(lora_weights: dict) -> dict[str, tuple[Any, Any, float]]:
    """将 Flux1 格式 LoRA 键名映射为 Flux2 模型参数键名。

    处理 BFL/diffusers 格式 (diffusion_model.double_blocks.*) → DanQing flux2 格式。
    """
    import re
    # 先分组 (A/B 对)
    groups: dict[str, dict] = {}
    default_alpha = 8.0

    for key, tensor in lora_weights.items():
        if "alpha" in key.lower():
            continue  # alpha 单独处理
        module = _lora_key_to_module(key)
        if module not in groups:
            groups[module] = {}
        if "lora_down" in key or "lora_A." in key or "lora_A_weight" in key:
            groups[module]["down"] = tensor
        elif "lora_up" in key or "lora_B." in key or "lora_B_weight" in key:
            groups[module]["up"] = tensor

    remapped: dict[str, tuple[Any, Any, float]] = {}
    for module, parts in groups.items():
        if "up" not in parts or "down" not in parts:
            continue
        up, down = parts["up"], parts["down"]
        alpha = parts.get("alpha", default_alpha)
        rank = down.shape[0]

        new_module = module

        # Strip common prefixes
        new_module = new_module.replace("diffusion_model.", "")

        # Double blocks: double_blocks.{i}.img_attn.qkv → transformer_blocks.{i}.attn.to_{q,k,v}
        m = re.match(r"double_blocks\.(\d+)\.img_attn\.qkv", new_module)
        if m:
            block = m.group(1)
            # split up tensor into 3 equal parts along dim 0
            for idx, suffix in enumerate(["to_q", "to_k", "to_v"]):
                tk = f"transformer_blocks.{block}.attn.{suffix}"
                chunk_size = up.shape[0] // 3
                down_chunk_size = down.shape[0] // 3
                u = up[idx * chunk_size:(idx + 1) * chunk_size] if up.shape[0] % 3 == 0 else up
                d = down[idx * down_chunk_size:(idx + 1) * down_chunk_size] if down.shape[0] % 3 == 0 else down
                remapped[tk] = (d, u, alpha)
            continue

        # Double blocks: img_attn.proj → transformer_blocks.{i}.attn.to_out
        m = re.match(r"double_blocks\.(\d+)\.img_attn\.proj", new_module)
        if m:
            new_module = f"transformer_blocks.{m.group(1)}.attn.to_out"

        # Double blocks: txt_attn.qkv → transformer_blocks.{i}.attn.add_{q,k,v}_proj
        m = re.match(r"double_blocks\.(\d+)\.txt_attn\.qkv", new_module)
        if m:
            block = m.group(1)
            for idx, suffix in enumerate(["add_q_proj", "add_k_proj", "add_v_proj"]):
                tk = f"transformer_blocks.{block}.attn.{suffix}"
                chunk_size = up.shape[0] // 3
                down_chunk_size = down.shape[0] // 3
                u = up[idx * chunk_size:(idx + 1) * chunk_size] if up.shape[0] % 3 == 0 else up
                d = down[idx * down_chunk_size:(idx + 1) * down_chunk_size] if down.shape[0] % 3 == 0 else down
                remapped[tk] = (d, u, alpha)
            continue

        # Double blocks: txt_attn.proj → transformer_blocks.{i}.attn.to_add_out
        m = re.match(r"double_blocks\.(\d+)\.txt_attn\.proj", new_module)
        if m:
            new_module = f"transformer_blocks.{m.group(1)}.attn.to_add_out"

        # Double blocks: img_mlp.0 → transformer_blocks.{i}.ff.linear_in
        m = re.match(r"double_blocks\.(\d+)\.img_mlp\.0", new_module)
        if m:
            new_module = f"transformer_blocks.{m.group(1)}.ff.linear_in"

        # Double blocks: img_mlp.2 → transformer_blocks.{i}.ff.linear_out
        m = re.match(r"double_blocks\.(\d+)\.img_mlp\.2", new_module)
        if m:
            new_module = f"transformer_blocks.{m.group(1)}.ff.linear_out"

        # Double blocks: txt_mlp.0 → transformer_blocks.{i}.ff_context.linear_in
        m = re.match(r"double_blocks\.(\d+)\.txt_mlp\.0", new_module)
        if m:
            new_module = f"transformer_blocks.{m.group(1)}.ff_context.linear_in"

        # Double blocks: txt_mlp.2 → transformer_blocks.{i}.ff_context.linear_out
        m = re.match(r"double_blocks\.(\d+)\.txt_mlp\.2", new_module)
        if m:
            new_module = f"transformer_blocks.{m.group(1)}.ff_context.linear_out"

        # Single blocks: single_blocks.{i}.linear1 → single_transformer_blocks.{i}.attn.to_qkv_mlp_proj
        m = re.match(r"single_blocks\.(\d+)\.linear1", new_module)
        if m:
            new_module = f"single_transformer_blocks.{m.group(1)}.attn.to_qkv_mlp_proj"

        # modulation: single_blocks.{i}.modulation.1 → single_stream_modulation.linear
        if "modulation" in new_module and "single" in new_module:
            new_module = "single_stream_modulation.linear"
        elif "img_mod" in new_module:
            new_module = "double_stream_modulation_img.linear"
        elif "txt_mod" in new_module:
            new_module = "double_stream_modulation_txt.linear"

        # Skip norm layers (no weight to merge into)
        if "norm" in new_module and "modulation" not in new_module:
            continue

        remapped[new_module] = (down, up, alpha)

    return remapped
def remap_flux2_weights(weights: dict) -> dict:
    """将 diffusers Flux.2 Klein 权重键映射为 DanQing 引擎键名。

    DanQing 现在与 mflux 结构一致，仅需少量映射：
    - .to_out.0 → .to_out (移除 Sequential wrapper)
    - .to_add_out.0 → .to_add_out (同上)
    - time_guidance_embed.timestep_embedder. → time_guidance_embed. (移除包装层)
    - 跳过 norm_added_q/norm_added_k (DanQing 使用 norm_q/norm_k 复用)
    """
    remapped = {}
    for key, tensor in weights.items():
        new_key = key
        # Attention output: remove .0 from Sequential wrapper
        new_key = new_key.replace(".to_out.0.", ".to_out.")
        new_key = new_key.replace(".to_add_out.0.", ".to_add_out.")
        # Timestep embedder: diffusers wraps in `timestep_embedder`, remove that layer
        new_key = new_key.replace("time_guidance_embed.timestep_embedder.", "time_guidance_embed.")
        # norm_added_q/norm_added_k are now used in DanQing Flux2Attention
        # (previously skipped because DanQing reused norm_q/norm_k for both)
        remapped[new_key] = tensor
    return remapped

