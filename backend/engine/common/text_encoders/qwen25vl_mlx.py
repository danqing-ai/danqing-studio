"""Qwen2.5-VL 文本编码 — MLX 张量封装；HF / PyTorch 前向在 ``qwen25vl_cuda``。"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import mlx.core as mx

from backend.engine.common.text_encoders.qwen25vl_cuda import (
    encode_prompt_embeds_qwen25vl_numpy,
    load_qwen25_vl_torch_model,
)


def encode_prompt_embeds_qwen25vl(
    tokenizer,
    model,
    prompts: list[str],
    *,
    tokenizer_max_length: int = 512,
    device: str = "cpu",
) -> mx.array:
    """``[B, tokenizer_max_length, hidden]`` MLX float32 — matches upstream slice layout."""
    hidden_np = encode_prompt_embeds_qwen25vl_numpy(
        tokenizer,
        model,
        prompts,
        tokenizer_max_length=tokenizer_max_length,
        device=device,
    )
    return mx.array(hidden_np, dtype=mx.float32)


class Qwen25VLEncoder:
    """Qwen2.5-VL 文本编码器。

    加载 transformers 模型，在 CPU 上运行，返回 MLX array。
    """

    def __init__(self, model_path: str | Path, device: str = "cpu"):
        from transformers import Qwen2Tokenizer

        self.model_path = Path(model_path)
        self.device = device

        text_encoder_path = self.model_path / "text_encoder"
        text_processor_path = self.model_path / "text_processor"

        self.tokenizer = Qwen2Tokenizer.from_pretrained(str(text_processor_path))

        self.model = load_qwen25_vl_torch_model(text_encoder_path, device=self.device)

    def encode(self, prompts: list[str], max_length: int = 512) -> mx.array:
        """Encodes prompts with the LongCatImage-compatible chat template + padding."""
        return encode_prompt_embeds_qwen25vl(
            self.tokenizer,
            self.model,
            prompts,
            tokenizer_max_length=max_length,
            device=self.device,
        )

    def __call__(self, prompts: list[str]) -> mx.array:
        return self.encode(prompts)
