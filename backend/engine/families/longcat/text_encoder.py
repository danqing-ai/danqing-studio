"""LongCat-Image text encoder — Qwen2.5-VL (HF on CPU → MLX embeds)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import mlx.core as mx


class LongCatTextEncoder:
    def __init__(self, ctx: Any, model_path: str | Path, **kwargs):
        from transformers import AutoModel, Qwen2Tokenizer
        import torch

        del ctx
        self.model_path = Path(model_path)
        tokenizer_path = kwargs.get("tokenizer_path")

        if tokenizer_path and Path(tokenizer_path).exists():
            tok_dir = Path(tokenizer_path)
        elif (self.model_path / "text_processor").exists():
            tok_dir = self.model_path / "text_processor"
        elif (self.model_path / "tokenizer").exists():
            tok_dir = self.model_path / "tokenizer"
        else:
            tok_dir = self.model_path

        if self.model_path.is_dir() and (self.model_path / "config.json").exists():
            text_encoder_path = self.model_path
        elif (self.model_path / "text_encoder").exists():
            text_encoder_path = self.model_path / "text_encoder"
        else:
            text_encoder_path = self.model_path

        self.tokenizer = Qwen2Tokenizer.from_pretrained(str(tok_dir))
        self.model = AutoModel.from_pretrained(
            str(text_encoder_path),
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to("cpu")
        self.model.eval()

    def encode(self, prompts: list[str], max_length: int = 512) -> mx.array:
        import torch

        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]

        hidden_np = hidden_states.cpu().float().numpy()
        return mx.array(hidden_np, dtype=mx.float32)

    def __call__(self, prompts: list[str]) -> mx.array:
        return self.encode(prompts)
