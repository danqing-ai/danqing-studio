"""HunyuanVideo Qwen2.5-VL text trunk — pure MLX (reuses Qwen-Image encoder stack)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from backend.engine.common.bundle.hf_tokenizer_json import load_hf_tokenizer, render_qwen_chat_messages
from backend.engine.runtime.mlx_runtime import run_clear_cache, run_eval
from backend.engine._transformer_registry import load_mlx_encoder_stack


class HunyuanQwen25VLEncoder:
    """Qwen2.5-VL MLX trunk for Hunyuan conditioning (``hidden_states[-3]``, crop after template)."""

    def __init__(
        self,
        enc_dir: Path,
        tok_dir: Path,
        *,
        weight_dtype: mx.Dtype = mx.bfloat16,
        ctx: Any | None = None,
    ):
        self._enc_dir = Path(enc_dir)
        self._tok_dir = Path(tok_dir)
        self._weight_dtype = weight_dtype
        self._ctx = ctx
        self._tokenizer = load_hf_tokenizer(str(self._tok_dir))
        self._eval_fn = getattr(self._ctx, "eval", None) if self._ctx is not None else None
        self._clear_cache_fn = (
            getattr(self._ctx, "clear_cache", None) if self._ctx is not None else None
        )
        self._load_fn = getattr(self._ctx, "load_weights", None) if self._ctx is not None else None
        self._encoder = load_mlx_encoder_stack(
            "qwen25vl",
            self._enc_dir,
            weight_dtype=weight_dtype,
            skip_lm_head=True,
            eval_fn=self._eval_fn,
            load_fn=self._load_fn,
            ctx=self._ctx,
        )

    def release_weights(self) -> None:
        self._encoder = None
        run_clear_cache(self._clear_cache_fn)

    def encode_batch(
        self,
        chats: list[list[dict]],
        *,
        max_length: int,
        crop_start: int,
        layer_index: int = -3,
    ) -> tuple[np.ndarray, np.ndarray]:
        texts = [render_qwen_chat_messages(chat) for chat in chats]
        input_ids, attn_mask = self._tokenizer.encode_batch(
            texts, max_length=max_length, add_special_tokens=False,
        )
        array_fn = mx.array
        if self._ctx is not None and hasattr(self._ctx, "array"):
            array_fn = self._ctx.array
        input_ids_mx = array_fn(input_ids)
        attn_mask_mx = array_fn(attn_mask)
        hidden = self._encoder.encode_hidden_at(input_ids_mx, attn_mask_mx, layer_index=layer_index)
        run_eval(self._eval_fn, hidden)
        emb = np.asarray(hidden[:, crop_start:], dtype=np.float32)
        mask = np.asarray(attn_mask[:, crop_start:], dtype=np.int32).astype(bool)
        run_clear_cache(self._clear_cache_fn)
        return emb, mask
