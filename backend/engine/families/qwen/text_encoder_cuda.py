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

    def release_weights(self) -> None:
        self._text_model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


_QWEN_EDIT_PROMPT_TEMPLATE = (
    "<|im_start|>system\n"
    "Describe the key features of the input image (color, shape, size, texture, objects, background), "
    "then explain how the user's text instruction should alter or modify the image. "
    "Generate a new image that meets the user's requirements while maintaining consistency "
    "with the original input where appropriate.<|im_end|>\n"
    "<|im_start|>user\n"
    "<|vision_start|><|image_pad|><|vision_end|>{}<|im_end|>\n"
    "<|im_start|>assistant\n"
)
_QWEN_EDIT_PLUS_USER_TEMPLATE = (
    "<|im_start|>system\n"
    "Describe the key features of the input image (color, shape, size, texture, objects, background), "
    "then explain how the user's text instruction should alter or modify the image. "
    "Generate a new image that meets the user's requirements while maintaining consistency "
    "with the original input where appropriate.<|im_end|>\n"
    "<|im_start|>user\n"
    "{}<|im_end|>\n"
    "<|im_start|>assistant\n"
)
_QWEN_EDIT_DROP_IDX = 64


def _format_qwen_edit_cuda_prompt(text: str, *, use_picture_prefix: bool, num_images: int) -> str:
    if use_picture_prefix:
        img_part = "".join(
            f"Picture {i + 1}: <|vision_start|><|image_pad|><|vision_end|>"
            for i in range(max(num_images, 1))
        )
        return _QWEN_EDIT_PLUS_USER_TEMPLATE.format(img_part + text)
    return _QWEN_EDIT_PROMPT_TEMPLATE.format(text)


def _load_qwen_edit_torch_encoder(bundle_root: Path, device: torch.device) -> tuple[Any, Any, Any]:
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    te_path = bundle_root / "text_encoder"
    tok_path = bundle_root / "tokenizer"
    if not te_path.is_dir():
        raise RuntimeError(f"Qwen Image Edit CUDA: missing text_encoder under {bundle_root}")
    processor = AutoProcessor.from_pretrained(str(tok_path if tok_path.is_dir() else te_path))
    text_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(te_path),
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)
    text_model.eval()
    return processor, text_model, device


def encode_qwen_edit_prompts_cuda(
    *,
    bundle_root: Path,
    device: torch.device,
    prompt: str,
    negative_prompt: str,
    sources: list[Any],
    use_picture_prefix: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    from PIL import Image

    if not sources:
        raise RuntimeError("Qwen edit CUDA encode requires at least one reference image.")

    processor, text_model, device = _load_qwen_edit_torch_encoder(bundle_root, device)
    images: list[Image.Image] = []
    for source in sources:
        if isinstance(source, Image.Image):
            images.append(source.convert("RGB"))
        else:
            images.append(Image.open(source).convert("RGB"))

    def _encode_one(text: str) -> tuple[torch.Tensor, torch.Tensor]:
        txt = _format_qwen_edit_cuda_prompt(
            text,
            use_picture_prefix=use_picture_prefix,
            num_images=len(images),
        )
        model_inputs = processor(text=[txt], images=images, padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = text_model(
                input_ids=model_inputs["input_ids"],
                attention_mask=model_inputs["attention_mask"],
                pixel_values=model_inputs.get("pixel_values"),
                image_grid_thw=model_inputs.get("image_grid_thw"),
                output_hidden_states=True,
                return_dict=True,
            )
        hidden = outputs.hidden_states[-1]
        mask = model_inputs["attention_mask"].bool()
        valid_lengths = mask.sum(dim=1)
        selected = hidden[mask]
        split_result = torch.split(selected, valid_lengths.tolist(), dim=0)
        trimmed = [e[_QWEN_EDIT_DROP_IDX:] for e in split_result]
        attn_mask_list = [torch.ones(e.size(0), dtype=torch.long, device=device) for e in trimmed]
        max_seq_len = max(e.size(0) for e in trimmed)
        embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) for u in trimmed]
        )
        attn = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in attn_mask_list]
        )
        return embeds.to(dtype=torch.bfloat16), attn

    pos_e, pos_m = _encode_one(prompt)
    neg_e, neg_m = _encode_one(negative_prompt or "")
    return pos_e, pos_m, neg_e, neg_m
