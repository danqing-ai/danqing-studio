"""Qwen-Image 权重 remap — 对外入口（MLX 映射在 ``weights_mlx``）。"""
from __future__ import annotations

from typing import Any


def remap_qwen_transformer_weights(weights: dict) -> dict:
    from backend.engine.families.qwen.weights_mlx import remap_qwen_transformer_weights as _remap

    return _remap(weights)


def remap_qwen_lora_module_prefix(hf_stem: str) -> str:
    from backend.engine.families.qwen.weights_mlx import remap_qwen_lora_module_prefix as _fn

    return _fn(hf_stem)


def remap_qwen_lora_keys(lora_weights: dict) -> dict[str, tuple[Any, Any, float]]:
    from backend.engine.families.qwen.weights_mlx import remap_qwen_lora_keys as _fn

    return _fn(lora_weights)
