"""CLIP 文本编码器 — PyTorch / CUDA 前向（形态 B：与 ``clip_mlx`` 分离）。"""
from __future__ import annotations

from typing import Any


def clip_encoder_encode_from_numpy(encoder: Any, input_ids_np) -> tuple[Any, Any]:
    """将 ``encode`` 中 ``return_tensors='np'`` 的 ``input_ids`` 送 HF CLIP，返回池化向量与 last_hidden_state。"""
    pooled_np, hidden_np = clip_cpu_torch_bridge_numpy(encoder, input_ids_np)
    import torch

    dev = getattr(encoder.ctx, "_device", None) or getattr(encoder.ctx, "device", None) or "cpu"
    if dev != "cpu" and str(dev).startswith("cuda"):
        return (
            torch.tensor(pooled_np, device=dev),
            torch.tensor(hidden_np, device=dev),
        )
    return pooled_np, hidden_np


def clip_cpu_torch_bridge_numpy(encoder: Any, input_ids_np) -> tuple[Any, Any]:
    """HF CLIPTextModel on CPU — for MLX hosts without ``mlx_lm.models.clip``."""
    import numpy as np
    import torch
    from transformers import CLIPTextModel

    input_ids = torch.tensor(input_ids_np, dtype=torch.long, device="cpu")
    if encoder._torch_bridge_model is None:
        encoder._torch_bridge_model = CLIPTextModel.from_pretrained(
            encoder.model_path, dtype=torch.float32
        )
        encoder._torch_bridge_model.eval()
    with torch.no_grad():
        outputs = encoder._torch_bridge_model(input_ids=input_ids)
    pooled = outputs.pooler_output.detach().cpu().numpy()
    hidden = outputs.last_hidden_state.detach().cpu().numpy()
    return np.asarray(pooled, dtype=np.float32), np.asarray(hidden, dtype=np.float32)
