"""T5-XXL 文本编码器 — MLX 前向；PyTorch 见 ``t5_cuda``。"""
from __future__ import annotations
from typing import Any


class T5Encoder:
    """T5-XXL 文本编码器。

    用于: Flux1 / Flux2 / LTX / Wan / CogVideoX 全系列。
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
    ):
        self.ctx = ctx
        self.model_path = model_path
        self._tokenizer_path = tokenizer_path or model_path
        self.max_seq_len = max_seq_len
        self.text_dim = text_dim
        self._tokenizer = None
        self._model = None
        self._torch_bridge_model = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import T5Tokenizer
            self._tokenizer = T5Tokenizer.from_pretrained(
                self._tokenizer_path, legacy=False,
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
            from backend.engine.common.text_encoders.t5_cuda import t5_forward_torch, t5_prepare_torch_tensors
            tid, tam = t5_prepare_torch_tensors(ctx, input_ids, attention_mask)
            return t5_forward_torch(self, tid, tam)
        import mlx.core as mx
        input_ids_mx = mx.array(input_ids, dtype=mx.int32)
        attention_mask_mx = mx.array(attention_mask, dtype=mx.float32)
        return self._forward_mlx(input_ids_mx, attention_mask_mx)

    def _forward_mlx(self, input_ids, attention_mask):
        import mlx.core as mx
        try:
            from mlx_lm.models.t5 import T5Model, T5Config
        except ImportError:
            return self._forward_mlx_via_torch_bridge(input_ids, attention_mask)
        if self._model is None:
            config = T5Config.from_pretrained(self.model_path)
            self._model = T5Model(config)
            self._model.load_weights(str(self.model_path))
            mx.eval(self._model.parameters())
        return self._model(input_ids, attention_mask)

    def _forward_mlx_via_torch_bridge(self, input_ids, attention_mask):
        """mlx-lm 已移除 ``models.t5`` 时：HF T5Encoder → numpy → MLX（数值与原生 MLX T5 不完全一致）。"""
        import mlx.core as mx
        from backend.engine.common.text_encoders.t5_cuda import t5_cpu_torch_bridge_hidden_numpy

        hidden_np = t5_cpu_torch_bridge_hidden_numpy(self, input_ids, attention_mask)
        return mx.array(hidden_np)
