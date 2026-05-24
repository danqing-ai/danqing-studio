"""Flux2 Klein text encoder built on shared Qwen3 MLX stack."""
from __future__ import annotations

from typing import Any

import mlx.core as mx

from backend.engine.common.text_encoders.qwen3_mlx import (
    Qwen3EncoderModel,
    build_qwen3_mlx_encoder,
)


class Flux2TextEncoder:
    """Flux2 Qwen3 prompt encoder matching mflux behavior."""

    def __init__(self, ctx: Any, model_path: str, tokenizer_path: str = "", max_seq_len: int = 512, **_kw: Any):
        self.ctx = ctx
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path or model_path
        self.max_seq_len = max_seq_len
        self._tokenizer = None
        self._model: Qwen3EncoderModel | None = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path, trust_remote_code=True)
        return self._tokenizer

    def _ensure_model(self) -> Qwen3EncoderModel:
        if self._model is None:
            self._model = build_qwen3_mlx_encoder(self.model_path, self.ctx)
        return self._model

    def encode(self, texts: list[str]) -> Any:
        tokenizer = self.tokenizer
        chat_texts = []
        for text in texts:
            chat_texts.append(
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": text}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            )
        tokens = tokenizer(
            chat_texts,
            padding="max_length",
            max_length=self.max_seq_len,
            truncation=True,
            return_tensors="np",
        )
        input_ids = self.ctx.array(tokens["input_ids"], dtype=mx.int32)
        attention_mask = self.ctx.array(tokens["attention_mask"], dtype=mx.int32)
        model = self._ensure_model()
        out = model.get_prompt_embeds(
            input_ids=input_ids,
            attention_mask=attention_mask,
            hidden_state_layers=(9, 18, 27),
        )
        return out.astype(mx.bfloat16)
