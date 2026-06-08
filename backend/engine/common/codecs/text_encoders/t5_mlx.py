"""T5-XXL 文本编码器 — MLX 前向；PyTorch 见 ``t5_cuda``。"""
from __future__ import annotations
import importlib
from typing import Any


class T5Encoder:
    """T5-XXL 文本编码器。

    用于: Flux1 / Flux2 / LTX / Wan 全系列。
    文本 → [B, seq_len, text_dim] 嵌入。
    """

    def __init__(
        self,
        ctx: Any,
        model_path: str,
        max_seq_len: int = 512,
        text_dim: int = 4096,
        *,
        tokenizer_path: str | None = None,
        weight_dtype: Any | None = None,
        native_mlx_only: bool = False,
    ):
        self.ctx = ctx
        self.model_path = model_path
        self._tokenizer_path = tokenizer_path or model_path
        self.max_seq_len = max_seq_len
        self.text_dim = text_dim
        self._weight_dtype = weight_dtype
        self._native_mlx_only = native_mlx_only
        self._tokenizer = None
        self._model = None
        self._torch_bridge_model = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                self._tokenizer_path, use_fast=False,
            )
        return self._tokenizer

    def encode(self, texts: list[str]) -> Any:
        tokenizer = self.tokenizer
        ctx = self.ctx
        tokens = tokenizer(
            texts, padding="max_length", max_length=self.max_seq_len,
            truncation=True, return_tensors="np",
        )
        input_ids = tokens["input_ids"]
        attention_mask = tokens["attention_mask"]
        if ctx.backend != "mlx":
            from backend.engine.common.codecs.text_encoders.t5_cuda import t5_forward_torch, t5_prepare_torch_tensors
            tid, tam = t5_prepare_torch_tensors(ctx, input_ids, attention_mask)
            return t5_forward_torch(self, tid, tam)
        import mlx.core as mx
        input_ids_mx = ctx.array(input_ids, dtype=mx.int32)
        attention_mask_mx = ctx.array(attention_mask, dtype=mx.float32)
        return self._forward_mlx(input_ids_mx, attention_mask_mx)

    def encode_with_mask(self, texts: list[str]) -> tuple[Any, Any]:
        """Encode texts → ``(hidden_states, attention_mask_bool)`` as MLX arrays."""
        tokenizer = self.tokenizer
        ctx = self.ctx
        tokens = tokenizer(
            texts, padding="max_length", max_length=self.max_seq_len,
            truncation=True, return_tensors="np",
        )
        input_ids = tokens["input_ids"]
        attention_mask = tokens["attention_mask"]
        if ctx.backend != "mlx":
            from backend.engine.common.codecs.text_encoders.t5_cuda import t5_prepare_torch_tensors, t5_forward_torch
            tid, tam = t5_prepare_torch_tensors(ctx, input_ids, attention_mask)
            hidden = t5_forward_torch(self, tid, tam)
            mask = ctx.array(attention_mask.astype(bool))
            return hidden, mask
        import mlx.core as mx
        input_ids_mx = ctx.array(input_ids, dtype=mx.int32)
        attention_mask_mx = ctx.array(attention_mask, dtype=mx.float32)
        hidden = self._forward_mlx(input_ids_mx, attention_mask_mx)
        mask = ctx.array(attention_mask.astype(bool))
        return hidden, mask

    def encode_tokenized_np(
        self,
        input_ids: Any,
        attention_mask: Any,
    ) -> tuple[Any, Any]:
        """Encode pre-tokenized numpy batches → MLX ``(hidden, bool_mask)``."""
        import numpy as np
        import mlx.core as mx

        ctx = self.ctx
        if ctx.backend != "mlx":
            from backend.engine.common.codecs.text_encoders.t5_cuda import t5_prepare_torch_tensors, t5_forward_torch
            tid, tam = t5_prepare_torch_tensors(ctx, input_ids, attention_mask)
            hidden = t5_forward_torch(self, tid, tam)
            mask = ctx.array(np.asarray(attention_mask).astype(bool))
            return hidden, mask
        input_ids_mx = ctx.array(input_ids, dtype=mx.int32)
        attention_mask_mx = ctx.array(attention_mask, dtype=mx.float32)
        hidden = self._forward_mlx(input_ids_mx, attention_mask_mx)
        mask = ctx.array(np.asarray(attention_mask).astype(bool))
        return hidden, mask

    def _forward_mlx(self, input_ids, attention_mask):
        import mlx.core as mx
        try:
            from mlx_lm.models.t5 import T5Model, T5Config
        except ImportError as e:
            if self._native_mlx_only:
                raise RuntimeError(
                    "T5 MLX forward requires mlx_lm.models.t5; torch bridge disabled for this encoder."
                ) from e
            return self._forward_mlx_via_torch_bridge(input_ids, attention_mask)
        if self._model is None:
            config = T5Config.from_pretrained(self.model_path)
            self._model = T5Model(config)
            self._model.load_weights(str(self.model_path))
            if self._weight_dtype is not None:
                from backend.engine.runtime.mlx_dtype import cast_module_parameters

                cast_module_parameters(
                    self._model, self._weight_dtype, eval_fn=self.ctx.eval
                )
            self.ctx.eval(self._model.parameters())
        return self._model(input_ids, attention_mask)

    def _forward_mlx_via_torch_bridge(self, input_ids, attention_mask):
        """mlx-lm 已移除 ``models.t5`` 时：HF T5Encoder → numpy → MLX（数值与原生 MLX T5 不完全一致）。"""
        try:
            from backend.engine.common.codecs.text_encoders.t5_cuda import t5_cpu_torch_bridge_hidden_numpy
        except ImportError as e:
            raise RuntimeError(
                "T5 MLX forward failed: mlx-lm has no T5 module and the PyTorch CPU bridge "
                "is not available in this build (desktop MLX bundle excludes torch/*_cuda). "
                "Ensure mlx-lm provides T5 or use a full CUDA-capable install."
            ) from e

        hidden_np = t5_cpu_torch_bridge_hidden_numpy(self, input_ids, attention_mask)
        return self.ctx.array(hidden_np)

    def release_weights(self) -> None:
        """Drop loaded MLX / bridge weights (tokenizers unchanged)."""
        self._model = None
        self._torch_bridge_model = None
        clear_cache_fn = getattr(self.ctx, "clear_cache", None)
        if clear_cache_fn is not None:
            clear_cache_fn()
        else:
            importlib.import_module("mlx.core").clear_cache()
