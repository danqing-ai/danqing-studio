"""CLIP 文本编码器 — PyTorch / CUDA 前向（形态 B：与 ``clip_mlx`` 分离）。"""
from __future__ import annotations

from typing import Any


def clip_encoder_encode_from_numpy(encoder: Any, input_ids_np) -> tuple[Any, Any]:
    """将 ``encode`` 中 ``return_tensors='np'`` 的 ``input_ids`` 送 HF CLIP，返回池化向量与 last_hidden_state。"""
    import torch

    dev = getattr(encoder.ctx, "_device", None) or getattr(encoder.ctx, "device", None) or "cpu"
    input_ids = torch.tensor(input_ids_np, dtype=torch.int32, device=dev)
    from transformers import CLIPTextModel

    if encoder._model is None:
        encoder._model = CLIPTextModel.from_pretrained(
            encoder.model_path, dtype=torch.float32
        ).to(dev)
        encoder._model.eval()
    with torch.no_grad():
        outputs = encoder._model(input_ids=input_ids)
    return outputs.pooler_output, outputs.last_hidden_state
