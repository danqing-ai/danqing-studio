"""Qwen3-VL conditioning for Boogu-Image (T2I + TI2I)."""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import numpy as np
from PIL import Image

SYSTEM_T2I = (
    "You are a helpful assistant that generates high-quality images based "
    "on user instructions. The instructions are as follows."
)
SYSTEM_TI2I = (
    "Describe the key features of the input image (color, shape, size, texture, objects, "
    "background), then explain how the user's text instruction should alter or modify the "
    "image. Generate a new image that meets the user's requirements while maintaining "
    "consistency with the original input where appropriate."
)


class _Identity:
    def __call__(self, x):
        return x


class BooguQwen3VLEncoderMLX:
    """Extract last_hidden_state features for Boogu DiT (dim 4096)."""

    def __init__(
        self,
        *,
        mllm_dir: str,
        processor_dir: str,
        dtype: Any = mx.bfloat16,
    ) -> None:
        from mlx_vlm import load as vlm_load

        self._dtype = dtype
        self._qwen_model, self.processor = vlm_load(mllm_dir)
        _ = processor_dir

    def encode_t2i(self, text: str) -> mx.array:
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_T2I}]},
            {"role": "user", "content": [{"type": "text", "text": text}]},
        ]
        enc = self.processor.apply_chat_template(
            [messages],
            tokenize=True,
            return_dict=True,
            add_generation_prompt=False,
            padding=True,
            return_tensors="np",
        )
        ids = mx.array(np.asarray(enc["input_ids"]))
        lm = self._qwen_model.language_model
        base = lm.model if hasattr(lm, "model") else lm
        feats = base(ids)
        return feats.astype(self._dtype)

    def encode_ti2i(self, image: Image.Image, text: str) -> mx.array:
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_TI2I}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": text},
                ],
            },
        ]
        prompt = self.processor.apply_chat_template(
            [messages],
            tokenize=False,
            add_generation_prompt=False,
        )
        if isinstance(prompt, list):
            prompt = prompt[0]
        enc = self.processor(
            text=prompt,
            images=[image],
            return_tensors="np",
            padding=True,
        )
        orig = self._qwen_model.language_model.lm_head
        self._qwen_model.language_model.lm_head = _Identity()
        h = self._qwen_model(
            mx.array(np.asarray(enc["input_ids"])),
            pixel_values=mx.array(np.asarray(enc["pixel_values"])),
            image_grid_thw=mx.array(np.asarray(enc["image_grid_thw"])),
            mask=None,
        )
        self._qwen_model.language_model.lm_head = orig
        h = h.logits if hasattr(h, "logits") else h
        return h.astype(self._dtype)

    def release_weights(self) -> None:
        self._qwen_model = None
        self.processor = None
        mx.clear_cache()
