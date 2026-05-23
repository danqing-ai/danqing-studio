"""HunyuanVideo Qwen2.5-VL text trunk — pure MLX (reuses Qwen-Image encoder stack)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
from mlx import nn

from backend.engine.common.hf_tokenizer_json import load_hf_tokenizer, render_qwen_chat_messages
from backend.engine.common.mlx_dtype import cast_floating_mx_tree
from backend.engine.families.qwen.text_encoder_mlx import QwenEncoder
from backend.engine.families.qwen.weights import apply_qwen_text_encoder_weights


def _read_qwen_text_config(model_dir: Path) -> dict[str, Any]:
    cfg_path = model_dir / "config.json"
    if not cfg_path.is_file():
        raise RuntimeError(f"HunyuanVideo Qwen: missing config.json under {model_dir}")
    data = json.loads(cfg_path.read_text(encoding="utf-8"))
    if isinstance(data.get("text_config"), dict):
        return data["text_config"]
    return data


def _load_qwen_encoder_weights(model_dir: Path, *, weight_dtype: mx.Dtype) -> QwenEncoder:
    cfg = _read_qwen_text_config(model_dir)
    encoder = QwenEncoder(
        vocab_size=int(cfg.get("vocab_size", 152064)),
        hidden_size=int(cfg.get("hidden_size", 3584)),
        num_hidden_layers=int(cfg.get("num_hidden_layers", 28)),
        max_position_embeddings=int(cfg.get("max_position_embeddings", 128000)),
        rope_theta=float(cfg.get("rope_theta", 1_000_000.0)),
    )
    raw: dict[str, Any] = {}
    globs = sorted(model_dir.glob("*.safetensors"))
    if not globs:
        raise RuntimeError(f"HunyuanVideo Qwen: no *.safetensors under {model_dir}")
    for sf in globs:
        part = dict(mx.load(str(sf)))
        for key, val in part.items():
            if key == "lm_head.weight" or key.startswith("lm_head."):
                continue
            raw[key] = val

    nested = apply_qwen_text_encoder_weights(raw)
    enc_nested = nested.get("encoder")
    if not isinstance(enc_nested, dict):
        raise RuntimeError("HunyuanVideo Qwen: weight remap did not produce encoder.* tree.")
    enc_nested = {k: v for k, v in enc_nested.items() if k != "visual"}
    enc_nested = cast_floating_mx_tree(enc_nested, weight_dtype)
    shell = nn.Module()
    shell.encoder = encoder
    shell.update({"encoder": enc_nested})
    mx.eval(shell.parameters())
    return shell.encoder


class HunyuanQwen25VLEncoder:
    """Qwen2.5-VL MLX trunk for Hunyuan conditioning (``hidden_states[-3]``, crop after template)."""

    def __init__(self, enc_dir: Path, tok_dir: Path, *, weight_dtype: mx.Dtype = mx.bfloat16):
        self._enc_dir = Path(enc_dir)
        self._tok_dir = Path(tok_dir)
        self._weight_dtype = weight_dtype
        self._tokenizer = load_hf_tokenizer(str(self._tok_dir))
        self._encoder = _load_qwen_encoder_weights(self._enc_dir, weight_dtype=weight_dtype)

    def release_weights(self) -> None:
        self._encoder = None
        mx.clear_cache()

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
        input_ids_mx = mx.array(input_ids)
        attn_mask_mx = mx.array(attn_mask)
        hidden = self._encoder.encode_hidden_at(input_ids_mx, attn_mask_mx, layer_index=layer_index)
        mx.eval(hidden)
        emb = np.asarray(hidden[:, crop_start:], dtype=np.float32)
        mask = np.asarray(attn_mask[:, crop_start:], dtype=np.int32).astype(bool)
        mx.clear_cache()
        return emb, mask
