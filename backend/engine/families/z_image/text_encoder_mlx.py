"""Z-Image / Flux2 共享：Qwen3 MLX 栈 + ``ZImageTextEncoder``（MLX / CUDA 双路径）。"""
from __future__ import annotations

from typing import Any

import mlx.core as mx

from backend.engine.common.codecs.text_encoders.qwen3_mlx import (
    Float32RMSNorm,
    MlxRMSNorm,
    MlxSwiGLUMLP,
    MlxTimestepEmbeddingMLP,
    Qwen3EncoderModel,
    _ZImageEncoderAttention,
    _ZImageEncoderLayer,
    _ZImageEncoderMLP,
    _ZImageEncoderModel,
    _ZImageEncoderRotaryEmbedding,
    build_zimage_mlx_encoder,
)

__all__ = [
    "Float32RMSNorm",
    "MlxRMSNorm",
    "MlxSwiGLUMLP",
    "MlxTimestepEmbeddingMLP",
    "ZImageTextEncoder",
    "_ZImageEncoderAttention",
    "_ZImageEncoderLayer",
    "_ZImageEncoderMLP",
    "_ZImageEncoderModel",
    "_ZImageEncoderRotaryEmbedding",
    "build_zimage_mlx_encoder",
]


class ZImageTextEncoder:
    """Z-Image text encoder — matches reference z_image TextEncoder.

    Two output modes (controlled by hidden_state_layers):
    - None (z_image): returns penultimate hidden state, shape [B, seq_len, hidden_size]
    - (9, 18, 27) (flux2 already split to Flux2TextEncoder)
    """

    def __init__(
        self,
        ctx: Any,
        model_path: str,
        max_seq_len: int = 512,
        tokenizer_path: str = "",
        hidden_state_layers: tuple[int, ...] | None = None,
        enable_thinking: bool = False,
        **_kw: Any,
    ):
        self.ctx = ctx
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path or model_path
        self.max_seq_len = max_seq_len
        self.hidden_state_layers = hidden_state_layers
        self.enable_thinking = enable_thinking
        self._tokenizer = None
        self._model: Qwen3EncoderModel | None = None
        self._compiled_layers = None

    def _refresh_compiled_layers(self) -> None:
        self._compiled_layers = None
        if getattr(self.ctx, "backend", None) != "mlx" or self._model is None:
            return
        penultimate_layers = self._model.layers[:-1]

        def _run_penultimate(hidden_states, causal_mask, position_embeddings):
            for layer in penultimate_layers:
                hidden_states = layer(
                    hidden_states=hidden_states,
                    attention_mask=causal_mask,
                    position_embeddings=position_embeddings,
                )
            return hidden_states

        try:
            self._compiled_layers = mx.compile(_run_penultimate)
        except Exception:
            self._compiled_layers = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_path,
                trust_remote_code=True,
            )
        return self._tokenizer

    def encode(self, texts: list[str]) -> Any:
        tokenizer = self.tokenizer
        ctx = self.ctx

        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            chat_texts = []
            for text in texts:
                chat = [{"role": "user", "content": text}]
                chat_text = tokenizer.apply_chat_template(
                    chat,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=self.enable_thinking,
                )
                chat_texts.append(chat_text)
            tokens = tokenizer(
                chat_texts,
                padding="max_length",
                max_length=self.max_seq_len,
                truncation=True,
                return_tensors="np",
            )
        else:
            tokens = tokenizer(
                texts,
                padding="max_length",
                max_length=self.max_seq_len,
                truncation=True,
                return_tensors="np",
            )
        input_ids = tokens["input_ids"]
        attention_mask = tokens["attention_mask"]
        num_valid = int(attention_mask.sum())

        if ctx.backend == "mlx":
            input_ids = ctx.array(input_ids, dtype=mx.int32)
            attention_mask = ctx.array(attention_mask, dtype=mx.float32)
            return self._forward_mlx(input_ids, attention_mask, num_valid)
        from backend.engine.families.z_image.text_encoder_cuda import (
            zimage_prepare_torch_ids,
            zimage_text_encoder_forward_torch,
        )

        tid, tam = zimage_prepare_torch_ids(ctx, input_ids, attention_mask)
        return zimage_text_encoder_forward_torch(self, tid, tam, num_valid)

    def _forward_mlx(self, input_ids, attention_mask, num_valid: int):
        if self._model is None:
            self._model = build_zimage_mlx_encoder(
                self.model_path, self.ctx, load_fn=getattr(self.ctx, "load_weights", None)
            )
            self._refresh_compiled_layers()

        batch_size, seq_len = input_ids.shape
        hidden_states = self._model.embed_tokens(input_ids).astype(mx.bfloat16)
        causal_mask = self._model._get_causal_mask(attention_mask, batch_size, hidden_states, seq_len)
        position_ids = self._model._build_position_ids(batch_size, seq_len)
        position_embeddings = self._model.rotary_emb(hidden_states, position_ids)

        if self.hidden_state_layers is not None:
            all_hidden_states = [hidden_states]
            for layer in self._model.layers:
                hidden_states = layer(
                    hidden_states=hidden_states,
                    attention_mask=causal_mask,
                    position_embeddings=position_embeddings,
                )
                all_hidden_states.append(hidden_states)
            layer_outputs = [all_hidden_states[i] for i in self.hidden_state_layers]
            stacked = mx.stack(layer_outputs, axis=1)
            B, L, S, D = stacked.shape
            result = mx.transpose(stacked, (0, 2, 1, 3)).reshape(B, S, L * D)
        else:
            penultimate_layers = self._model.layers[:-1]
            if self._compiled_layers is not None:
                hidden_states = self._compiled_layers(
                    hidden_states, causal_mask, position_embeddings,
                )
            else:
                for layer in penultimate_layers:
                    hidden_states = layer(
                        hidden_states=hidden_states,
                        attention_mask=causal_mask,
                        position_embeddings=position_embeddings,
                    )
            result = hidden_states

        result = result[:, :num_valid, :]
        return result.astype(mx.bfloat16)
