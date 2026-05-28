"""Qwen-Image 文本编码 — PyTorch / CUDA（HF Qwen2.5-VL trunk + Image 提示模板）。"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

_QWEN_IMAGE_PROMPT_TEMPLATE = (
    "<|im_start|>system\n"
    "Describe the image by detailing the color, shape, size, texture, quantity, text, "
    "spatial relationships of the objects and background:<|im_end|>\n"
    "<|im_start|>user\n{}<|im_end|>\n"
    "<|im_start|>assistant\n"
)


def _load_qwen_image_torch_encoder(bundle_root: Path, device: torch.device) -> tuple[Any, Any]:
    from transformers import AutoModel, AutoTokenizer

    te_path = bundle_root / "text_encoder"
    tok_path = bundle_root / "tokenizer"
    if not te_path.is_dir():
        raise RuntimeError(f"Qwen Image CUDA: missing text_encoder under {bundle_root}")
    if not tok_path.is_dir():
        tok_path = te_path

    tokenizer = AutoTokenizer.from_pretrained(str(tok_path), trust_remote_code=True)
    text_model = AutoModel.from_pretrained(
        str(te_path),
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)
    text_model.eval()
    return tokenizer, text_model


def encode_qwen_image_prompt_cuda(
    encoder: Any,
    *,
    prompt: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode one prompt → ``(hidden_states, attention_mask)`` bfloat16 on device."""
    tokenizer = encoder._tokenizer
    filled = _QWEN_IMAGE_PROMPT_TEMPLATE.format(prompt)
    batch = tokenizer(
        filled,
        return_tensors="pt",
        padding="longest",
        max_length=1058,
        truncation=True,
    )
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    text_model = encoder._text_model

    with torch.no_grad():
        out = text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            return_dict=True,
        )
    hidden = out.last_hidden_state.to(dtype=torch.bfloat16)
    return hidden, attention_mask


class QwenImageTextEncoderCuda:
    """CUDA 文本编码器句柄（由 ``qwen_image_mlx.QwenImageTextEncoder`` 在 cuda 后端选用）。"""

    def __init__(self, ctx: Any, bundle_root: str | Path, **_kw: Any):
        self.ctx = ctx
        self.bundle_root = Path(bundle_root)
        self._device = torch.device(getattr(ctx, "_device", "cuda"))
        self._tokenizer, self._text_model = _load_qwen_image_torch_encoder(
            self.bundle_root, self._device
        )

    def encode(self, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        prompt = texts[0] if texts else ""
        if not prompt:
            raise RuntimeError("QwenImageTextEncoder.encode requires non-empty texts")
        wrapper = type(
            "_Enc",
            (),
            {"_tokenizer": self._tokenizer, "_text_model": self._text_model},
        )()
        return encode_qwen_image_prompt_cuda(wrapper, prompt=prompt, device=self._device)
