"""Step1X-Edit Qwen2.5-VL conditioner (MLX) — prompt/token logic from ``modules/conditioner.py``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
from PIL import Image

STEP1X_PREFIX = (
    'Given a user prompt, generate an "Enhanced prompt" that provides detailed visual descriptions '
    "suitable for image generation. Evaluate the level of detail in the user prompt:\n"
    "- If the prompt is simple, focus on adding specifics about colors, shapes, sizes, textures, "
    "and spatial relationships to create vivid and concrete scenes.\n"
    "- If the prompt is already detailed, refine and enhance the existing details slightly "
    "without overcomplicating.\n\n"
    "Here are examples of how to transform or refine prompts:\n"
    "- User Prompt: A cat sleeping -> Enhanced: A small, fluffy white cat curled up in a round shape, "
    "sleeping peacefully on a warm sunny windowsill, surrounded by pots of blooming red flowers.\n"
    "- User Prompt: A busy city street -> Enhanced: A bustling city street scene at dusk, featuring "
    "glowing street lamps, a diverse crowd of people in colorful clothing, and a double-decker bus "
    "passing by towering glass skyscrapers.\n\n"
    "Please generate only the enhanced description for the prompt below and avoid including any "
    "additional commentary or evaluations:\n"
    "User Prompt:"
)

SPLICE_TOKEN = 151653
DROP_PREFIX_TOKENS = 217


def split_string(s: str) -> list[str]:
    s = s.replace("'", '"').replace("\u201c", '"').replace("\u201d", '"')
    result: list[str] = []
    in_quotes = False
    temp = ""
    for idx, char in enumerate(s):
        if char == '"' and idx > 155:
            temp += char
            if not in_quotes:
                result.append(temp)
                temp = ""
            in_quotes = not in_quotes
            continue
        if in_quotes:
            if char.isspace():
                pass
            else:
                result.append("\u201c" + char + "\u201d")
        else:
            temp += char
    if temp:
        result.append(temp)
    return result


def _chw_to_pil(img: np.ndarray) -> Image.Image:
    if img.ndim == 3 and img.shape[0] in (1, 3):
        arr = np.transpose(img, (1, 2, 0))
        if arr.max() <= 1.0:
            arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
        else:
            arr = arr.clip(0, 255).astype(np.uint8)
        return Image.fromarray(arr)
    raise ValueError(f"Expected CHW image array, got shape {img.shape}")


class Step1XQwen25VLEmbedderMLX:
    """Qwen2.5-VL trunk + Step1X-specific token splice (skip 217, max_length=640)."""

    hidden_size = 3584

    def __init__(self, model_path: Path, *, max_length: int = 640, ctx: Any) -> None:
        from transformers import AutoProcessor

        try:
            from qwen_vl_utils import process_vision_info
        except ImportError as exc:
            raise RuntimeError(
                "Step1X-Edit MLX conditioner requires qwen-vl-utils. "
                "Install with: pip install qwen-vl-utils"
            ) from exc

        self._ctx = ctx
        self._process_vision_info = process_vision_info
        self.max_length = max_length
        self.prefix = STEP1X_PREFIX
        root = Path(model_path)
        self.processor = AutoProcessor.from_pretrained(
            str(root),
            min_pixels=256 * 28 * 28,
            max_pixels=324 * 28 * 28,
        )
        from backend.engine.families.qwen.text_encoder_mlx import load_qwen25vl_mlx_encoder

        self.encoder = load_qwen25vl_mlx_encoder(
            root,
            ctx=ctx,
            load_fn=getattr(ctx, "load_weights", None),
            strip_visual=False,
        )

    def _build_input_ids(self, text: str, image: Image.Image) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        messages = [{"role": "user", "content": [{"type": "text", "text": self.prefix}, {"type": "image", "image": image}, {"type": "text", "text": text}]}]
        chat_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, add_vision_id=True
        )
        image_inputs, _video_inputs = self._process_vision_info(messages)
        inputs = self.processor(text=[chat_text], images=image_inputs, padding=True, return_tensors="np")
        old_ids = inputs["input_ids"][0]
        token_list: list[np.ndarray] = []
        for text_each in split_string(chat_text):
            part = self.processor(text=text_each, images=None, videos=None, padding=True, return_tensors="np")
            token_each = part["input_ids"][0:1]
            if token_each.shape[1] >= 2 and token_each[0, 0] == 2073 and token_each[0, -1] == 854:
                token_each = token_each[:, 1:-1]
            token_list.append(token_each)
        new_ids = np.concatenate(token_list, axis=1)[0]
        idx1 = int(np.where(old_ids == SPLICE_TOKEN)[0][0])
        idx2 = int(np.where(new_ids == SPLICE_TOKEN)[0][0])
        merged = np.concatenate([old_ids[:idx1], new_ids[idx2:]], axis=0)
        attn = (merged > 0).astype(np.int64)
        return merged[None], attn[None], inputs["pixel_values"], inputs["image_grid_thw"]

    def __call__(self, captions: list[str], ref_images: np.ndarray) -> tuple[mx.array, mx.array]:
        batch = len(captions)
        embs = mx.zeros((batch, self.max_length, self.hidden_size), dtype=mx.bfloat16)
        masks = mx.zeros((batch, self.max_length), dtype=mx.int32)
        for idx, (txt, imgs) in enumerate(zip(captions, ref_images)):
            pil = _chw_to_pil(np.asarray(imgs))
            input_ids, attention_mask, pixel_values, image_grid_thw = self._build_input_ids(txt, pil)
            hidden = self.encoder(
                input_ids=mx.array(input_ids),
                attention_mask=mx.array(attention_mask),
                pixel_values=mx.array(pixel_values),
                image_grid_thw=mx.array(image_grid_thw),
            )
            if hasattr(self._ctx, "eval"):
                self._ctx.eval(hidden)
            seq_len = int(hidden.shape[1])
            take = min(self.max_length, max(0, seq_len - DROP_PREFIX_TOKENS))
            if take > 0:
                sl = hidden[0, DROP_PREFIX_TOKENS : DROP_PREFIX_TOKENS + take]
                embs[idx, :take] = sl.astype(mx.bfloat16)
                masks[idx, :take] = mx.ones((take,), dtype=mx.int32)
        return embs, masks
