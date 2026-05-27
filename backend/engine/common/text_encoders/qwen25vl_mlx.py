"""Qwen2.5-VL 文本编码 — MLX 张量封装；HF / PyTorch CPU 前向在 ``qwen25vl_cuda``（非桌面 MLX 包）。

Deprecated for runtime generation: Qwen-Image and HunyuanVideo use pure MLX via
``families/qwen/text_encoder_mlx.load_qwen25vl_mlx_encoder``. Kept for CUDA packaging
hidden imports only.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import mlx.core as mx


def encode_prompt_embeds_qwen25vl(
    tokenizer,
    model,
    prompts: list[str],
    *,
    tokenizer_max_length: int = 512,
    device: str = "cpu",
    array_fn: Any | None = None,
) -> mx.array:
    """``[B, tokenizer_max_length, hidden]`` MLX float32 — matches upstream slice layout."""
    try:
        from backend.engine.common.text_encoders.qwen25vl_cuda import (
            encode_prompt_embeds_qwen25vl_numpy,
        )
    except ImportError as e:
        raise RuntimeError(
            "Qwen2.5-VL encoding requires PyTorch (qwen25vl_cuda); "
            "not included in MLX-only desktop bundles."
        ) from e

    hidden_np = encode_prompt_embeds_qwen25vl_numpy(
        tokenizer,
        model,
        prompts,
        tokenizer_max_length=tokenizer_max_length,
        device=device,
    )
    if array_fn is None:
        array_fn = mx.array
    return array_fn(hidden_np, dtype=mx.float32)


class Qwen25VLEncoder:
    """Qwen2.5-VL 文本编码器。

    加载 transformers 模型，在 CPU 上运行，返回 MLX array。
    """

    def __init__(self, model_path: str | Path, device: str = "cpu", *, array_fn: Any | None = None):
        from transformers import Qwen2Tokenizer

        self.model_path = Path(model_path)
        self.device = device
        self._array_fn = array_fn or mx.array

        text_encoder_path = self.model_path / "text_encoder"
        text_processor_path = self.model_path / "text_processor"

        self.tokenizer = Qwen2Tokenizer.from_pretrained(str(text_processor_path))

        try:
            from backend.engine.common.text_encoders.qwen25vl_cuda import (
                load_qwen25_vl_torch_model,
            )
        except ImportError as e:
            raise RuntimeError(
                "Qwen2.5-VL encoder requires PyTorch (qwen25vl_cuda); "
                "not included in MLX-only desktop bundles."
            ) from e

        self.model = load_qwen25_vl_torch_model(text_encoder_path, device=self.device)

    def encode(self, prompts: list[str], max_length: int = 512) -> mx.array:
        """Encodes prompts with the LongCatImage-compatible chat template + padding."""
        return encode_prompt_embeds_qwen25vl(
            self.tokenizer,
            self.model,
            prompts,
            tokenizer_max_length=max_length,
            device=self.device,
            array_fn=self._array_fn,
        )

    def __call__(self, prompts: list[str]) -> mx.array:
        return self.encode(prompts)
