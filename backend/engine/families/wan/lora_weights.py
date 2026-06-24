"""Wan video DiT LoRA key grouping (LightX2V / Lightning rank-64)."""
from __future__ import annotations

import re
from typing import Any


def _lora_key_to_module(key: str) -> str:
    for suffix in (
        ".lora_A.default.weight",
        ".lora_B.default.weight",
        ".lora_A.weight",
        ".lora_B.weight",
        ".lora_down.weight",
        ".lora_up.weight",
        ".lora_A",
        ".lora_B",
        ".lora_down",
        ".lora_up",
        "lora_A",
        "lora_B",
        "lora_down",
        "lora_up",
    ):
        if suffix in key:
            return key.replace(suffix, "").rstrip(".")
    return key


def _is_lora_down_key(key: str) -> bool:
    k = key.lower()
    return "lora_down" in k or ".lora_a." in k or k.endswith(".lora_a") or ".lora_a." in k


def _is_lora_up_key(key: str) -> bool:
    k = key.lower()
    return "lora_up" in k or ".lora_b." in k or k.endswith(".lora_b") or ".lora_b." in k


def remap_wan_lora_keys(lora_weights: dict[str, Any]) -> dict[str, tuple[Any, Any, float]]:
    """Group flat LoRA safetensors into ``module → (down, up, alpha)`` for Wan DiT."""
    groups: dict[str, dict[str, Any]] = {}
    alphas: dict[str, float] = {}

    for key, tensor in lora_weights.items():
        kl = key.lower()
        if kl.endswith(".alpha") or kl.endswith("_alpha") or key.endswith(".lora_alpha"):
            module = _lora_key_to_module(key.replace(".alpha", "").replace(".lora_alpha", ""))
            try:
                alphas[module] = float(tensor) if not hasattr(tensor, "item") else float(tensor.item())
            except (TypeError, ValueError):
                pass
            continue
        if "lora" not in kl and not kl.endswith(".diff"):
            continue
        if kl.endswith(".diff_b") or kl.endswith(".diff_m"):
            continue
        module = _lora_key_to_module(key)
        if module not in groups:
            groups[module] = {}
        if _is_lora_down_key(key):
            groups[module]["down"] = tensor
        elif _is_lora_up_key(key):
            groups[module]["up"] = tensor
        elif kl.endswith(".diff") or kl.endswith(".diff.weight"):
            groups[module]["diff"] = tensor

    remapped: dict[str, tuple[Any, Any, float]] = {}
    for module, parts in groups.items():
        if "diff" in parts and "up" not in parts:
            remapped[f"{module}.delta"] = (parts["diff"], parts["diff"], 1.0)
            continue
        if "up" not in parts or "down" not in parts:
            continue
        alpha = alphas.get(module, float(parts["down"].shape[0]))
        remapped[module] = (parts["down"], parts["up"], alpha)
    return remapped


def wan_lora_param_key(module_name: str) -> str:
    """Map LoRA module stem to ``WanModelMLX._param_map`` weight key."""
    name = module_name
    if name.startswith("transformer."):
        name = name[len("transformer.") :]
    if name.startswith("diffusion_model."):
        name = name[len("diffusion_model.") :]
    name = re.sub(r"\.attn1\.to_q\.", ".self_attn.q.", name)
    name = re.sub(r"\.attn1\.to_k\.", ".self_attn.k.", name)
    name = re.sub(r"\.attn1\.to_v\.", ".self_attn.v.", name)
    name = re.sub(r"\.attn1\.to_out\.0\.", ".self_attn.o.", name)
    name = re.sub(r"\.attn2\.to_q\.", ".cross_attn.q.", name)
    name = re.sub(r"\.attn2\.to_k\.", ".cross_attn.k.", name)
    name = re.sub(r"\.attn2\.to_v\.", ".cross_attn.v.", name)
    name = re.sub(r"\.attn2\.to_out\.0\.", ".cross_attn.o.", name)
    name = name.replace(".ffn.net.0.proj.", ".ffn.layer_0.")
    name = name.replace(".ffn.net.2.", ".ffn.layer_2.")
    name = re.sub(r"\.ffn\.0(?=\.|$)", ".ffn.layer_0", name)
    name = re.sub(r"\.ffn\.2(?=\.|$)", ".ffn.layer_2", name)
    if name.endswith(".delta"):
        return name.replace(".delta", ".weight")
    if not name.endswith(".weight"):
        name = f"{name}.weight"
    return name
