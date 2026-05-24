"""Flux2 Klein text encoder — MlxRMSNorm + float32 SDPA + 3-layer concatenation."""
from __future__ import annotations
from json import load as json_load
from pathlib import Path
from typing import Any
import mlx.core as mx
import mlx.nn as nn
from backend.engine.common.attention import build_causal_with_padding_bias
from backend.engine.common.mlx_runtime_fallback import load_weights_dict
from backend.engine.common.text_encoders.qwen3_mlx import MlxRMSNorm
from backend.engine.families.z_image.text_encoder_mlx import (
    _ZImageEncoderLayer,
    _ZImageEncoderRotaryEmbedding,
)


def _flux2_rms_norm(dims: int, eps: float = 1e-6):
    return MlxRMSNorm(dims, eps=eps)


class _Flux2DecoderLayer(_ZImageEncoderLayer):
    """Flux2 Qwen3 layer — MlxRMSNorm + tuple return for API compat."""

    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        num_key_value_heads,
        intermediate_size,
        head_dim,
        rms_norm_eps=1e-6,
    ):
        super().__init__(
            hidden_size,
            num_attention_heads,
            num_key_value_heads,
            intermediate_size,
            head_dim,
            rms_norm_eps=rms_norm_eps,
            rms_norm_factory=_flux2_rms_norm,
        )

    def __call__(self, hidden_states, attention_mask, position_embeddings, past_key_value=None):
        return super().__call__(hidden_states, attention_mask, position_embeddings), None


class _Flux2EncoderModel(nn.Module):
    """Flux2 Qwen3 编码器模型。"""

    def __init__(
        self,
        vocab_size=151936,
        hidden_size=4096,
        num_hidden_layers=36,
        num_attention_heads=32,
        num_key_value_heads=8,
        intermediate_size=12288,
        head_dim=128,
        rope_theta=1000000.0,
        rms_norm_eps=1e-6,
    ):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = [
            _Flux2DecoderLayer(
                hidden_size,
                num_attention_heads,
                num_key_value_heads,
                intermediate_size,
                head_dim,
                rms_norm_eps,
            )
            for _ in range(num_hidden_layers)
        ]
        self.rotary_emb = _ZImageEncoderRotaryEmbedding(dim=head_dim, base=rope_theta)


class Flux2TextEncoder:
    """Flux2 Klein text encoder — matches reference qwen3_text_encoder.

    - MlxRMSNorm instead of nn.RMSNorm
    - float32 SDPA (via shared Z-Image attention path)
    - j > i causal mask
    - get_prompt_embeds logic (takes (9,18,27) three-layer concat)
    - enable_thinking=False
    """

    def __init__(self, ctx: Any, model_path: str, tokenizer_path: str = "", max_seq_len: int = 512, **_kw: Any):
        self.ctx = ctx
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path or model_path
        self.max_seq_len = max_seq_len
        self._tokenizer = None
        self._model = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path, trust_remote_code=True)
        return self._tokenizer

    def encode(self, texts: list[str]) -> Any:
        tokenizer = self.tokenizer
        chat_texts = []
        for text in texts:
            chat = [{"role": "user", "content": text}]
            chat_texts.append(tokenizer.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True, enable_thinking=False,
            ))
        tokens = tokenizer(chat_texts, padding="max_length", max_length=self.max_seq_len,
                          truncation=True, return_tensors="np")
        input_ids = tokens["input_ids"]
        attention_mask = tokens["attention_mask"]
        input_ids = self.ctx.array(input_ids, dtype=mx.int32)
        attention_mask = self.ctx.array(attention_mask, dtype=mx.float32)
        return self._forward(input_ids, attention_mask)

    def _forward(self, input_ids, attention_mask):
        if self._model is None:
            self._model = self._build_model()
        B, S = input_ids.shape
        hidden = self._model.embed_tokens(input_ids).astype(mx.float32)
        pos_ids = mx.broadcast_to(mx.arange(S, dtype=mx.int32)[None, :], (B, S))
        pos_emb = self._model.rotary_emb(hidden, pos_ids)

        mask_dtype = hidden.dtype
        attn_mask = build_causal_with_padding_bias(
            mx,
            attention_mask,
            S,
            mask_dtype,
            valid_value=1,
            neg_value=float("-inf"),
            batch_size=B,
        )

        all_hidden = [hidden]
        for layer in self._model.layers:
            hidden, _ = layer(hidden, attn_mask, pos_emb, None)
            all_hidden.append(hidden)

        outputs = [all_hidden[i] for i in (9, 18, 27)]
        stacked = mx.stack(outputs, axis=1)
        _, L, _, D = stacked.shape
        result = mx.transpose(stacked, (0, 2, 1, 3)).reshape(B, -1, L * D)
        return result.astype(mx.bfloat16)

    def _build_model(self):
        d = Path(self.model_path)
        w = {}
        load_fn = getattr(self.ctx, "load_weights", None)
        for sf in sorted(d.glob("*.safetensors")):
            w.update(load_weights_dict(load_fn, str(sf)))
        cfg = {}
        if (d / "config.json").exists():
            with open(d / "config.json", encoding="utf-8") as f:
                cfg = json_load(f)
        model = _Flux2EncoderModel(
            vocab_size=cfg.get("vocab_size", 151936),
            hidden_size=cfg.get("hidden_size", 4096),
            num_hidden_layers=cfg.get("num_hidden_layers", 36),
            num_attention_heads=cfg.get("num_attention_heads", 32),
            num_key_value_heads=cfg.get("num_key_value_heads", 8),
            intermediate_size=cfg.get("intermediate_size", 12288),
            head_dim=cfg.get("head_dim", 128),
            rope_theta=cfg.get("rope_theta", 1000000.0),
            rms_norm_eps=cfg.get("rms_norm_eps", 1e-6),
        )
        rw = {k[6:] if k.startswith("model.") else k: v for k, v in w.items()}
        model.load_weights(list(rw.items()), strict=False)
        self.ctx.eval(model.parameters())
        return model
