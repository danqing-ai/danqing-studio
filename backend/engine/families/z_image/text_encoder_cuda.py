"""Z-Image 文本编码器 — PyTorch / CUDA 前向（形态 B：与 ``text_encoder_mlx`` 分离）。"""
from __future__ import annotations

from typing import Any


def zimage_prepare_torch_ids(ctx: Any, input_ids_np, attention_mask_np):
    import torch

    input_ids = torch.tensor(input_ids_np, dtype=torch.long, device=ctx._device)
    attention_mask = torch.tensor(attention_mask_np, dtype=torch.long, device=ctx._device)
    return input_ids, attention_mask


def zimage_text_encoder_forward_torch(encoder: Any, input_ids, attention_mask) -> Any:
    import torch
    from transformers import AutoModel

    if encoder._model is None:
        encoder._model = AutoModel.from_pretrained(
            encoder.model_path,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        ).to(encoder.ctx._device)
        encoder._model.eval()

    with torch.no_grad():
        outputs = encoder._model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
    hs = outputs.hidden_states
    if hs is None:
        raise RuntimeError(
            "ZImageTextEncoder (torch): expected hidden_states; "
            "ensure transformers supports output_hidden_states for this checkpoint."
        )

    if encoder.hidden_state_layers is not None:
        layer_outputs = [hs[i] for i in encoder.hidden_state_layers]
        stacked = torch.stack(layer_outputs, dim=1)
        B, L, S, D = stacked.shape
        result = stacked.transpose(1, 2).reshape(B, S, L * D)
    else:
        result = hs[-2]

    num_valid = int(attention_mask.sum().item())
    result = result[:, :num_valid, :]
    return result.to(dtype=torch.bfloat16)
