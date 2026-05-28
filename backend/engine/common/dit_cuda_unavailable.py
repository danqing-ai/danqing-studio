"""DiT CUDA 未实现 — 多族共用 fail-loud（common 仅保留 ≥2 调用方）。"""
from __future__ import annotations


def raise_cuda_dit_unavailable(product: str) -> None:
    raise RuntimeError(
        f"{product} CUDA DiT 尚未实现；请在 models_registry 的 backends 中使用 mlx，"
        f"或在 Apple Silicon 上使用 MLX 路径。"
    )
