"""HunyuanVideo-1.5 DiT weight remap — diffusers / ModelScope → flat ``HunyuanVideoTransformer`` keys."""
from __future__ import annotations

import re


def _conv3d_torch_to_mlx(w):
    import importlib

    mx = importlib.import_module("mlx.core")
    if not isinstance(w, mx.array) or w.ndim != 5:
        return w
    # PyTorch Conv3d [O, I, T, H, W] → MLX [O, T, H, W, I]
    if int(w.shape[-1]) >= int(w.shape[1]):
        return w
    return mx.transpose(w, (0, 2, 3, 4, 1))


def _split_qkv_linear(weight, bias):
    import importlib

    mx = importlib.import_module("mlx.core")
    wq, wk, wv = mx.split(weight, 3, axis=0)
    if bias is None:
        return (wq, wk, wv), None
    bq, bk, bv = mx.split(bias, 3, axis=0)
    return (wq, wk, wv), (bq, bk, bv)


def _is_hunyuan_modelscope_native(key: str) -> bool:
    return key.startswith(
        (
            "double_blocks.",
            "img_in.",
            "txt_in.",
            "byt5_in.",
            "vision_in.",
            "time_in.",
            "cond_type_embedding.",
            "final_layer.",
        )
    )


def _remap_txt_refiner_block_key(key: str) -> str | None:
    prefix = "txt_in.individual_token_refiner.blocks."
    if not key.startswith(prefix):
        return None
    rest = key[len(prefix):]
    dot = rest.find(".")
    if dot <= 0:
        return None
    idx, tail = rest[:dot], rest[dot + 1 :]
    base = f"context_embedder.token_refiner.refiner_blocks.{idx}"
    mapping = {
        "self_attn_proj.": f"{base}.attn.to_out.0.",
        "norm1.": f"{base}.norm1.",
        "norm2.": f"{base}.norm2.",
        "mlp.fc1.": f"{base}.ff.net.0.",
        "mlp.fc2.": f"{base}.ff.net.2.",
        "adaLN_modulation.1.": f"{base}.norm_out.linear.",
    }
    for src, dst in mapping.items():
        if tail.startswith(src):
            return dst + tail[len(src):]
    return None


def _remap_hunyuan_modelscope_key(key: str) -> str | None:
    """Map one Tencent ModelScope DiT key to ``HunyuanVideoDiTMLX`` flat param name."""
    if key == "img_in.proj.weight":
        return "x_embedder.proj.weight"
    if key == "img_in.proj.bias":
        return "x_embedder.proj.bias"
    if key.startswith("time_in.mlp.0."):
        return key.replace("time_in.mlp.0.", "time_embed.timestep_embedder.linear_1.")
    if key.startswith("time_in.mlp.2."):
        return key.replace("time_in.mlp.2.", "time_embed.timestep_embedder.linear_2.")
    if key == "cond_type_embedding.weight":
        return "cond_type_embed.weight"

    if key.startswith("byt5_in.layernorm."):
        return key.replace("byt5_in.layernorm.", "context_embedder_2.norm.")
    for src, dst in (
        ("byt5_in.fc1.", "context_embedder_2.linear_1."),
        ("byt5_in.fc2.", "context_embedder_2.linear_2."),
        ("byt5_in.fc3.", "context_embedder_2.linear_3."),
    ):
        if key.startswith(src):
            return key.replace(src, dst)

    for src, dst in (
        ("vision_in.proj.0.", "image_embedder.norm_in."),
        ("vision_in.proj.1.", "image_embedder.linear_1."),
        ("vision_in.proj.3.", "image_embedder.linear_2."),
        ("vision_in.proj.4.", "image_embedder.norm_out."),
    ):
        if key.startswith(src):
            return key.replace(src, dst)

    if key.startswith("txt_in.input_embedder."):
        return key.replace("txt_in.input_embedder.", "context_embedder.proj_in.")
    if key.startswith("txt_in.t_embedder.mlp.0."):
        return key.replace(
            "txt_in.t_embedder.mlp.0.",
            "context_embedder.time_text_embed.timestep_embedder.linear_1.",
        )
    if key.startswith("txt_in.t_embedder.mlp.2."):
        return key.replace(
            "txt_in.t_embedder.mlp.2.",
            "context_embedder.time_text_embed.timestep_embedder.linear_2.",
        )
    if key.startswith("txt_in.c_embedder.linear_1."):
        return key.replace(
            "txt_in.c_embedder.linear_1.",
            "context_embedder.time_text_embed.text_embedder.linear_1.",
        )
    if key.startswith("txt_in.c_embedder.linear_2."):
        return key.replace(
            "txt_in.c_embedder.linear_2.",
            "context_embedder.time_text_embed.text_embedder.linear_2.",
        )

    refiner_key = _remap_txt_refiner_block_key(key)
    if refiner_key is not None:
        return refiner_key

    m = re.match(r"double_blocks\.(\d+)\.img_attn_q\.(weight|bias)", key)
    if m:
        return f"transformer_blocks.{m.group(1)}.attn.to_q.{m.group(2)}"
    m = re.match(r"double_blocks\.(\d+)\.img_attn_k\.(weight|bias)", key)
    if m:
        return f"transformer_blocks.{m.group(1)}.attn.to_k.{m.group(2)}"
    m = re.match(r"double_blocks\.(\d+)\.img_attn_v\.(weight|bias)", key)
    if m:
        return f"transformer_blocks.{m.group(1)}.attn.to_v.{m.group(2)}"
    m = re.match(r"double_blocks\.(\d+)\.img_attn_q_norm\.weight", key)
    if m:
        return f"transformer_blocks.{m.group(1)}.attn.norm_q.weight"
    m = re.match(r"double_blocks\.(\d+)\.img_attn_k_norm\.weight", key)
    if m:
        return f"transformer_blocks.{m.group(1)}.attn.norm_k.weight"
    m = re.match(r"double_blocks\.(\d+)\.img_attn_proj\.(weight|bias)", key)
    if m:
        return f"transformer_blocks.{m.group(1)}.attn.to_out.0.{m.group(2)}"

    m = re.match(r"double_blocks\.(\d+)\.txt_attn_q\.(weight|bias)", key)
    if m:
        return f"transformer_blocks.{m.group(1)}.attn.add_q_proj.{m.group(2)}"
    m = re.match(r"double_blocks\.(\d+)\.txt_attn_k\.(weight|bias)", key)
    if m:
        return f"transformer_blocks.{m.group(1)}.attn.add_k_proj.{m.group(2)}"
    m = re.match(r"double_blocks\.(\d+)\.txt_attn_v\.(weight|bias)", key)
    if m:
        return f"transformer_blocks.{m.group(1)}.attn.add_v_proj.{m.group(2)}"
    m = re.match(r"double_blocks\.(\d+)\.txt_attn_q_norm\.weight", key)
    if m:
        return f"transformer_blocks.{m.group(1)}.attn.norm_added_q.weight"
    m = re.match(r"double_blocks\.(\d+)\.txt_attn_k_norm\.weight", key)
    if m:
        return f"transformer_blocks.{m.group(1)}.attn.norm_added_k.weight"
    m = re.match(r"double_blocks\.(\d+)\.txt_attn_proj\.(weight|bias)", key)
    if m:
        return f"transformer_blocks.{m.group(1)}.attn.to_add_out.{m.group(2)}"

    for attn_side, ff_side in (
        ("img_mlp.fc1.", "ff.net.0."),
        ("img_mlp.fc2.", "ff.net.2."),
        ("txt_mlp.fc1.", "ff_context.net.0."),
        ("txt_mlp.fc2.", "ff_context.net.2."),
    ):
        m = re.match(rf"double_blocks\.(\d+)\.{re.escape(attn_side)}(weight|bias)", key)
        if m:
            return f"transformer_blocks.{m.group(1)}.{ff_side}{m.group(2)}"

    m = re.match(r"double_blocks\.(\d+)\.img_mod\.linear\.(weight|bias)", key)
    if m:
        return f"transformer_blocks.{m.group(1)}.norm1.linear.{m.group(2)}"
    m = re.match(r"double_blocks\.(\d+)\.txt_mod\.linear\.(weight|bias)", key)
    if m:
        return f"transformer_blocks.{m.group(1)}.norm1_context.linear.{m.group(2)}"

    if key.startswith("final_layer.adaLN_modulation.1."):
        return key.replace("final_layer.adaLN_modulation.1.", "norm_out.linear.")
    if key.startswith("final_layer.linear."):
        return key.replace("final_layer.linear.", "proj_out.")
    return None


def _remap_hunyuan_modelscope_weights(weights: dict) -> dict:
    remapped: dict = {}
    pending_qkv: dict[str, tuple] = {}

    for key, tensor in weights.items():
        m = re.match(
            r"txt_in\.individual_token_refiner\.blocks\.(\d+)\.self_attn_qkv\.(weight|bias)",
            key,
        )
        if m:
            pending_qkv.setdefault(m.group(1), {})[m.group(2)] = tensor
            continue

        new_key = _remap_hunyuan_modelscope_key(key)
        if new_key is None:
            continue
        if new_key == "x_embedder.proj.weight":
            tensor = _conv3d_torch_to_mlx(tensor)
        remapped[new_key] = tensor

    for block_idx, parts in pending_qkv.items():
        weight = parts.get("weight")
        bias = parts.get("bias")
        if weight is None:
            continue
        (wq, wk, wv), bias_parts = _split_qkv_linear(weight, bias)
        prefix = f"context_embedder.token_refiner.refiner_blocks.{block_idx}.attn"
        remapped[f"{prefix}.to_q.weight"] = wq
        remapped[f"{prefix}.to_k.weight"] = wk
        remapped[f"{prefix}.to_v.weight"] = wv
        if bias_parts is not None:
            bq, bk, bv = bias_parts
            remapped[f"{prefix}.to_q.bias"] = bq
            remapped[f"{prefix}.to_k.bias"] = bk
            remapped[f"{prefix}.to_v.bias"] = bv

    return remapped


def remap_hunyuan_weights(weights: dict) -> dict:
    """Strip common checkpoint prefixes; remap Conv3d patch weights to MLX NHWC layout."""
    sample_key = next(iter(weights.keys()), "")
    if _is_hunyuan_modelscope_native(sample_key) or any(
        _is_hunyuan_modelscope_native(k) for k in list(weights.keys())[:32]
    ):
        return _remap_hunyuan_modelscope_weights(weights)

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
