"""SeedVR2 权重 bundle 元数据（仅 3B / 7B 超分路径使用）。"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

import mlx.core as mx


class ModelConfig:
    """与上游扁平 safetensors 布局对齐的最小配置。"""

    precision: Any = mx.bfloat16

    def __init__(
        self,
        priority: int,
        aliases: list[str],
        model_name: str,
        base_model: str | None,
        controlnet_model: str | None,
        custom_transformer_model: str | None,
        num_train_steps: int | None,
        max_sequence_length: int | None,
        supports_guidance: bool | None,
        requires_sigma_shift: bool | None,
        transformer_overrides: dict | None = None,
        text_encoder_overrides: dict | None = None,
        sigma_base_shift: float = 0.5,
        sigma_max_shift: float = 1.15,
        sigma_base_seq_len: int = 256,
        sigma_max_seq_len: int = 4096,
        sigma_shift_terminal: float | None = None,
    ):
        self.aliases = aliases
        self.model_name = model_name
        self.base_model = base_model
        self.controlnet_model = controlnet_model
        self.custom_transformer_model = custom_transformer_model
        self.num_train_steps = num_train_steps
        self.max_sequence_length = max_sequence_length
        self.supports_guidance = supports_guidance
        self.requires_sigma_shift = requires_sigma_shift
        self.priority = priority
        self.transformer_overrides = transformer_overrides or {}
        self.text_encoder_overrides = text_encoder_overrides or {}
        self.sigma_base_shift = sigma_base_shift
        self.sigma_max_shift = sigma_max_shift
        self.sigma_base_seq_len = sigma_base_seq_len
        self.sigma_max_seq_len = sigma_max_seq_len
        self.sigma_shift_terminal = sigma_shift_terminal

    @staticmethod
    @lru_cache
    def seedvr2_3b() -> "ModelConfig":
        return AVAILABLE_MODELS["seedvr2-3b"]

    @staticmethod
    @lru_cache
    def seedvr2_7b() -> "ModelConfig":
        return AVAILABLE_MODELS["seedvr2-7b"]


AVAILABLE_MODELS = {
    "seedvr2-3b": ModelConfig(
        priority=22,
        aliases=["seedvr2-3b", "seedvr2"],
        model_name="numz/SeedVR2_comfyUI",
        base_model=None,
        controlnet_model=None,
        custom_transformer_model=None,
        num_train_steps=None,
        max_sequence_length=None,
        supports_guidance=True,
        requires_sigma_shift=None,
    ),
    "seedvr2-7b": ModelConfig(
        priority=23,
        aliases=["seedvr2-7b", "seedvr2-7B"],
        model_name="numz/SeedVR2_comfyUI",
        base_model=None,
        controlnet_model=None,
        custom_transformer_model=None,
        num_train_steps=None,
        max_sequence_length=None,
        supports_guidance=True,
        requires_sigma_shift=None,
        transformer_overrides={
            "vid_dim": 3072,
            "heads": 24,
            "num_layers": 36,
            "mm_layers": 36,
            "rope_dim": 64,
            "rope_on_text": False,
            "rope_freqs_for": "pixel",
            "mlp_type": "normal",
            "use_output_ada": False,
            "last_layer_vid_only": False,
        },
    ),
}
