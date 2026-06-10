"""T5-XXL 文本编码器 — MLX / CUDA dispatch（CUDA 见 ``t5_cuda``）。"""
from __future__ import annotations

from typing import Any

from backend.engine.common.codecs.text_encoders.t5_mlx import T5EncoderMlx


class T5Encoder(T5EncoderMlx):
    """Public T5 encoder — CUDA forward stays out of ``*_mlx.py``."""

    def encode(self, texts: list[str]) -> Any:
        ctx = self.ctx
        if getattr(ctx, "backend", None) != "mlx":
            from backend.engine.common.codecs.text_encoders.t5_cuda import (
                t5_forward_torch,
                t5_prepare_torch_tensors,
            )

            tokenizer = self.tokenizer
            tokens = tokenizer(
                texts,
                padding="max_length",
                max_length=self.max_seq_len,
                truncation=True,
                return_tensors="np",
            )
            tid, tam = t5_prepare_torch_tensors(ctx, tokens["input_ids"], tokens["attention_mask"])
            return t5_forward_torch(self, tid, tam)
        return super().encode(texts)

    def encode_with_mask(self, texts: list[str]) -> tuple[Any, Any]:
        ctx = self.ctx
        if getattr(ctx, "backend", None) != "mlx":
            from backend.engine.common.codecs.text_encoders.t5_cuda import (
                t5_forward_torch,
                t5_prepare_torch_tensors,
            )

            tokenizer = self.tokenizer
            tokens = tokenizer(
                texts,
                padding="max_length",
                max_length=self.max_seq_len,
                truncation=True,
                return_tensors="np",
            )
            input_ids = tokens["input_ids"]
            attention_mask = tokens["attention_mask"]
            tid, tam = t5_prepare_torch_tensors(ctx, input_ids, attention_mask)
            hidden = t5_forward_torch(self, tid, tam)
            mask = ctx.array(attention_mask.astype(bool))
            return hidden, mask
        return super().encode_with_mask(texts)

    def encode_tokenized_np(
        self,
        input_ids: Any,
        attention_mask: Any,
    ) -> tuple[Any, Any]:
        ctx = self.ctx
        if getattr(ctx, "backend", None) != "mlx":
            import numpy as np
            from backend.engine.common.codecs.text_encoders.t5_cuda import (
                t5_forward_torch,
                t5_prepare_torch_tensors,
            )

            tid, tam = t5_prepare_torch_tensors(ctx, input_ids, attention_mask)
            hidden = t5_forward_torch(self, tid, tam)
            mask = ctx.array(np.asarray(attention_mask).astype(bool))
            return hidden, mask
        return super().encode_tokenized_np(input_ids, attention_mask)
