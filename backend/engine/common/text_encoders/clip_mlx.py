"""CLIP 文本编码器 — MLX 前向；PyTorch 见 ``clip_cuda``。"""
from __future__ import annotations
from typing import Any


class CLIPEncoder:
    """CLIP 文本/图像编码器。用于: Flux1 系列的双编码器（T5 + CLIP）。"""

    def __init__(self, ctx: Any, model_path: str,
                 max_seq_len: int = 77, embed_dim: int = 768):
        self.ctx = ctx
        self.model_path = model_path
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        self._tokenizer = None
        self._model = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import CLIPTokenizer
            self._tokenizer = CLIPTokenizer.from_pretrained(self.model_path)
        return self._tokenizer

    def encode(self, texts: list[str]) -> tuple[Any, Any]:
        tokenizer = self.tokenizer
        ctx = self.ctx
        tokens = tokenizer(
            texts, padding="max_length", max_length=self.max_seq_len,
            truncation=True, return_tensors="np",
        )
        input_ids = tokens["input_ids"]
        if ctx.backend == "mlx":
            import mlx.core as mx
            input_ids_mx = mx.array(input_ids, dtype=mx.int32)
            return self._forward_mlx(input_ids_mx)
        from backend.engine.common.text_encoders.clip_cuda import clip_encoder_encode_from_numpy
        return clip_encoder_encode_from_numpy(self, input_ids)

    def _forward_mlx(self, input_ids):
        import mlx.core as mx
        from mlx_lm.models.clip import CLIPTextModel, CLIPTextConfig
        if self._model is None:
            config = CLIPTextConfig.from_pretrained(self.model_path)
            self._model = CLIPTextModel(config)
            self._model.load_weights(str(self.model_path))
            mx.eval(self._model.parameters())
        pooled, hidden = self._model(input_ids)
        return pooled, hidden
