"""CLIP 文本编码器 — MLX / CUDA dispatch（CUDA 见 ``clip_cuda``）。"""
from __future__ import annotations

from typing import Any

from backend.engine.common.codecs.text_encoders.clip_mlx import CLIPEncoderMlx


class CLIPEncoder(CLIPEncoderMlx):
    """Public CLIP encoder — CUDA forward stays out of ``*_mlx.py``."""

    def encode(self, texts: list[str]) -> tuple[Any, Any]:
        if getattr(self.ctx, "backend", None) != "mlx":
            from backend.engine.common.codecs.text_encoders.clip_cuda import clip_encoder_encode_from_numpy

            tokenizer = self.tokenizer
            tokens = tokenizer(
                texts,
                padding="max_length",
                max_length=self.max_seq_len,
                truncation=True,
                return_tensors="np",
            )
            return clip_encoder_encode_from_numpy(self, tokens["input_ids"])
        return super().encode(texts)
