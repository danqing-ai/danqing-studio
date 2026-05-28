"""文本编码器 — T5 / CLIP / Qwen2.5-VL / Qwen3 MLX（PEP 562 懒加载，避免 import 即拉起重依赖）。"""
from __future__ import annotations

__all__ = (
    "T5Encoder",
    "CLIPEncoder",
    "Qwen25VLEncoder",
    "build_zimage_mlx_encoder",
    "Flux2TextEncoder",
    "MlxRMSNorm",
)


def __getattr__(name: str):
    if name == "T5Encoder":
        from backend.engine.common.text_encoders.t5_mlx import T5Encoder

        return T5Encoder
    if name == "CLIPEncoder":
        from backend.engine.common.text_encoders.clip_mlx import CLIPEncoder

        return CLIPEncoder
    if name == "Qwen25VLEncoder":
        from backend.engine.common.text_encoders.qwen25vl_mlx import Qwen25VLEncoder

        return Qwen25VLEncoder
    if name == "build_zimage_mlx_encoder":
        from backend.engine.common.text_encoders.qwen3_mlx import build_zimage_mlx_encoder

        return build_zimage_mlx_encoder
    if name == "MlxRMSNorm":
        from backend.engine.common.text_encoders.qwen3_mlx import MlxRMSNorm

        return MlxRMSNorm
    if name == "Flux2TextEncoder":
        from backend.engine.common.text_encoders.qwen3_mlx import Flux2TextEncoder

        return Flux2TextEncoder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return list(__all__)
