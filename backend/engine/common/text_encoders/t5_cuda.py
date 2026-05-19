"""T5 文本编码器 — PyTorch 前向与 CPU 桥接（形态 B：与 ``t5_mlx`` 分离）。"""
from __future__ import annotations

from typing import Any

import numpy as np


def t5_prepare_torch_tensors(ctx: Any, input_ids_np, attention_mask_np):
    import torch

    dev = getattr(ctx, "_device", None) or getattr(ctx, "device", None) or "cpu"
    return (
        torch.tensor(input_ids_np, dtype=torch.int32, device=dev),
        torch.tensor(attention_mask_np, dtype=torch.float32, device=dev),
    )


def t5_forward_torch(encoder: Any, input_ids, attention_mask) -> Any:
    import torch
    from transformers import T5EncoderModel

    if encoder._model is None:
        dev = getattr(encoder.ctx, "_device", None) or getattr(encoder.ctx, "device", None) or "cpu"
        encoder._model = T5EncoderModel.from_pretrained(
            encoder.model_path, dtype=torch.float32
        ).to(dev)
        encoder._model.eval()
    enc_kwargs: dict[str, Any] = {"input_ids": input_ids}
    if attention_mask is not None:
        enc_kwargs["attention_mask"] = attention_mask
    with torch.no_grad():
        outputs = encoder._model(**enc_kwargs)
    return outputs.last_hidden_state


def t5_cpu_torch_bridge_hidden_numpy(encoder: Any, input_ids, attention_mask) -> np.ndarray:
    """HF T5Encoder on CPU → ``last_hidden_state`` numpy（mlx-lm 缺 T5 时由 MLX 路径调用）。"""
    import torch
    from transformers import T5EncoderModel

    if encoder._torch_bridge_model is None:
        encoder._torch_bridge_model = T5EncoderModel.from_pretrained(
            encoder.model_path, dtype=torch.float32
        ).to("cpu").eval()
    ii = np.asarray(input_ids)
    am = np.asarray(attention_mask)
    enc_kwargs: dict[str, Any] = {
        "input_ids": torch.tensor(ii, dtype=torch.long, device="cpu"),
    }
    if attention_mask is not None:
        enc_kwargs["attention_mask"] = torch.tensor(
            np.asarray(attention_mask), dtype=torch.float32, device="cpu"
        )
    with torch.no_grad():
        outputs = encoder._torch_bridge_model(**enc_kwargs)
    return outputs.last_hidden_state.numpy()
