"""权重工具 — safetensors 加载 / LoRA 注入 / 量化 / VAE 映射。模型专属映射在 models/image/_*_weights.py。"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional

from backend.engine.common.weights._vae import remap_vae_weights


def parse_size_gb(size_str: str) -> float:
    match = re.match(r"([\d.]+)\s*(GB|MB)", str(size_str).strip(), re.IGNORECASE)
    if not match:
        return 0.0
    val = float(match.group(1))
    unit = match.group(2).upper()
    return val if unit == "GB" else val / 1024


def load_safetensors(path: str, ctx: Any = None) -> dict:
    if ctx is not None:
        return ctx.load_weights(path)
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
    def __init__(self, path: str, strength: float = 1.0,
                 target_modules: Optional[list[str]] = None):
        self.path = path
        self.strength = strength
        self.target_modules = target_modules


def _lora_key_to_module(key: str) -> str:
    for suffix in (".lora_A", ".lora_B", ".lora_down", ".lora_up",
                    "lora_A", "lora_B", "lora_down", "lora_up"):
        if suffix in key:
            return key.replace(suffix, "").rstrip(".")
    return key


def load_lora_weights(lora_path: str) -> dict[str, tuple[Any, Any, float]]:
    weights = load_safetensors(lora_path)
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
    result = dict(weights)
    for cfg in lora_configs:
        loras = load_lora_weights(cfg.path)
        for module_name, (down, up, alpha) in loras.items():
            weight_key = f"{module_name}.weight"
            if weight_key in result:
                orig_weight = result[weight_key]
                rank = down.shape[0]
                scale = (alpha / rank) * cfg.strength
                delta = up @ down
                result[weight_key] = orig_weight + scale * delta
    return result


def quantize_weights(weights: dict, bits: int = 4,
                     group_size: int = 64) -> dict:
    if bits not in (4, 8):
        raise ValueError(f"Unsupported bits: {bits}")
    quantized = {}
    for key, tensor in weights.items():
        if tensor.ndim < 2:
            quantized[key] = tensor
            continue
        groups = (tensor.shape[-1] + group_size - 1) // group_size
        import mlx.core as mx
        if isinstance(tensor, mx.array):
            flat = tensor.reshape(-1, tensor.shape[-1])
        else:
            import numpy as np
            flat = np.array(tensor).reshape(-1, tensor.shape[-1])
        flat_float = flat.astype(float)
        flat_min = flat_float.min(axis=-1, keepdims=True)
        flat_max = flat_float.max(axis=-1, keepdims=True)
        scale_val = (flat_max - flat_min) / ((1 << bits) - 1) + 1e-6
        zero_point = (-flat_min / scale_val).round()
        q = ((flat_float / scale_val) + zero_point).round().clip(0, (1 << bits) - 1).astype(int)
        quantized[key] = q
        quantized[f"{key}.scale"] = scale_val.reshape(-1)
        quantized[f"{key}.zero_point"] = zero_point.reshape(-1)
    return quantized
