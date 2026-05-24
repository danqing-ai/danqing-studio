"""FIBO text encoder — SmolLM3 hidden states (last + penultimate → 4096-d)."""
from __future__ import annotations

from typing import Any


class FiboTextEncoder:
    """Encode JSON prompts with bundled SmolLM3 weights (HF torch bridge on MLX builds)."""

    def __init__(
        self,
        ctx: Any,
        model_path: str,
        *,
        tokenizer_path: str | None = None,
        max_seq_len: int = 2048,
        **kwargs: Any,
    ):
        self.ctx = ctx
        self.model_path = model_path
        self._tokenizer_path = tokenizer_path or model_path
        self.max_seq_len = int(max_seq_len)
        self._tokenizer = None
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self._tokenizer_path, use_fast=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            dtype=torch.bfloat16,
            device_map="cpu",
        )
        self._model.eval()

    def encode(self, texts: list[str]) -> Any:
        import json

        import numpy as np
        import torch

        if not texts:
            raise ValueError("FiboTextEncoder.encode requires non-empty texts")
        for t in texts:
            json.loads(t)

        self._load()
        assert self._tokenizer is not None and self._model is not None
        batch = self._tokenizer(
            texts,
            padding="max_length",
            max_length=self.max_seq_len,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            out = self._model(**batch, output_hidden_states=True)
        last = out.hidden_states[-1]
        prev = out.hidden_states[-2]
        embeds = torch.cat([last, prev], dim=-1).to(torch.float32).numpy()
        return self.ctx.array(embeds)
