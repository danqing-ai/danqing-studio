"""T5-XXL 文本编码器。"""
from __future__ import annotations
from typing import Any


class T5Encoder:
    """T5-XXL 文本编码器。

    用于: Flux1 / Flux2 / LTX / Wan / CogVideoX 全系列。
    文本 → [B, seq_len, text_dim] 嵌入。
    """

    def __init__(self, ctx: Any, model_path: str,
                 max_seq_len: int = 512, text_dim: int = 4096):
        self.ctx = ctx
        self.model_path = model_path
        self.max_seq_len = max_seq_len
        self.text_dim = text_dim
        self._tokenizer = None
        self._model = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import T5Tokenizer
            self._tokenizer = T5Tokenizer.from_pretrained(
                self.model_path, legacy=False,
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
        if ctx.backend == "mlx":
            import mlx.core as mx
            input_ids = mx.array(input_ids, dtype=mx.int32)
            attention_mask = mx.array(attention_mask, dtype=mx.float32)
        else:
            import torch
            input_ids = torch.tensor(input_ids, dtype=torch.int32, device=ctx._device)
            attention_mask = torch.tensor(attention_mask, dtype=torch.float32, device=ctx._device)
        return self._forward(input_ids, attention_mask)

    def _forward(self, input_ids, attention_mask):
        if self.ctx.backend == "mlx":
            return self._forward_mlx(input_ids, attention_mask)
        else:
            return self._forward_torch(input_ids, attention_mask)

    def _forward_mlx(self, input_ids, attention_mask):
        import mlx.core as mx
        try:
            from mlx_lm.models.t5 import T5Model, T5Config
        except ImportError:
            raise ImportError("mlx_lm not installed. Install with: pip install mlx-lm")
        if self._model is None:
            config = T5Config.from_pretrained(self.model_path)
            self._model = T5Model(config)
            self._model.load_weights(str(self.model_path))
            mx.eval(self._model.parameters())
        return self._model(input_ids, attention_mask)

    def _forward_torch(self, input_ids, attention_mask):
        import torch
        from transformers import T5EncoderModel
        if self._model is None:
            self._model = T5EncoderModel.from_pretrained(
                self.model_path, torch_dtype=torch.float32
            ).to(self.ctx._device)
            self._model.eval()
        with torch.no_grad():
            outputs = self._model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state
