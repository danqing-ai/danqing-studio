"""SeedVR2 超分用的固定正提示嵌入（无运行时文本编码器）。"""
from __future__ import annotations

from pathlib import Path

import mlx.core as mx


def _pos_emb_path() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "pos_emb.safetensors"


class SeedVR2PositiveEmbeddings:
    """与权重 bundle 配套的 ``pos_emb.safetensors`` 常量嵌入。"""

    @staticmethod
    def load(batch_size: int = 1) -> mx.array:
        emb = mx.load(str(_pos_emb_path()))["embedding"]
        if emb.ndim == 2:
            emb = emb[None, ...]
        if batch_size > 1:
            emb = mx.repeat(emb, batch_size, axis=0)
        return emb
