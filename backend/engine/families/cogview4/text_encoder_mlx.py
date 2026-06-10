"""CogView4 GLM-4-9B text encoder — native MLX ``GlmModel`` (penultimate hidden state)."""
from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from backend.engine.families.cogview4.text_encoder_glm_mlx import (
    GlmEncoderModel,
    build_glm_encoder_mlx,
)


class CogView4TextEncoder:
    """GLM-4 encoder for CogView4 — returns prompt embeddings only."""

    def __init__(
        self,
        ctx: Any,
        model_path: str,
        max_seq_len: int = 1024,
        tokenizer_path: str = "",
        **_kw: Any,
    ):
        self.ctx = ctx
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path or model_path
        self.max_seq_len = int(max_seq_len)
        self._tokenizer = None
        self._model: GlmEncoderModel | None = None
        self._compiled_penultimate = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            tok_path = Path(self.tokenizer_path)
            tok_json = tok_path / "tokenizer.json"
            if tok_json.is_file():
                from transformers import PreTrainedTokenizerFast

                self._tokenizer = PreTrainedTokenizerFast.from_pretrained(str(tok_path))
            else:
                from transformers import AutoTokenizer

                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.tokenizer_path,
                    trust_remote_code=True,
                )
        return self._tokenizer

    def _tokenize_glm_np(self, texts: list[str]) -> np.ndarray:
        """Match diffusers ``_get_glm_embeds``: longest padding + prepend pad to multiple of 16."""
        tokenizer = self.tokenizer
        rows: list[list[int]] = []
        for text in texts:
            ids = tokenizer(
                text,
                padding="longest",
                max_length=self.max_seq_len,
                truncation=True,
                add_special_tokens=True,
                return_tensors=None,
            )["input_ids"]
            if isinstance(ids[0], list):
                ids = ids[0]
            rows.append(list(ids))
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
        max_len = max(len(r) for r in rows) if rows else 1
        pad_prefix = (16 - (max_len % 16)) % 16
        if pad_prefix > 0:
            prefix = [int(pad_id)] * pad_prefix
            rows = [prefix + r for r in rows]
        return np.asarray(rows, dtype=np.int32)

    def _tokenize_glm(self, texts: list[str]) -> mx.array:
        return self.ctx.array(self._tokenize_glm_np(texts), dtype=mx.int32)

    def _ensure_model(self) -> GlmEncoderModel:
        if self._model is None:
            self._model = build_glm_encoder_mlx(
                self.model_path,
                self.ctx,
                load_fn=getattr(self.ctx, "load_weights", None),
            )
            self._refresh_compiled_penultimate()
        return self._model

    def _refresh_compiled_penultimate(self) -> None:
        self._compiled_penultimate = None
        if getattr(self.ctx, "backend", None) != "mlx" or self._model is None:
            return
        model = self._model
        penultimate_layers = model.layers[:-1]

        def _run_penultimate(hidden_states, mask):
            for layer in penultimate_layers:
                hidden_states = layer(hidden_states, mask)
            return hidden_states

        try:
            self._compiled_penultimate = mx.compile(_run_penultimate)
        except Exception:
            self._compiled_penultimate = None

    def encode(self, texts: list[str]) -> mx.array:
        input_ids = self._tokenize_glm(texts)
        model = self._ensure_model()
        hidden = model.embed_tokens(input_ids)
        from mlx_lm.models.base import create_attention_mask

        mask = create_attention_mask(hidden, None)
        if self._compiled_penultimate is not None:
            hidden = self._compiled_penultimate(hidden, mask)
        else:
            for layer in model.layers[:-1]:
                hidden = layer(hidden, mask)
        return hidden.astype(mx.bfloat16)

    def release_weights(self) -> None:
        self._model = None
        self._compiled_penultimate = None
        clear_cache_fn = getattr(self.ctx, "clear_cache", None)
        if clear_cache_fn is not None:
            clear_cache_fn()
        else:
            importlib.import_module("mlx.core").clear_cache()
