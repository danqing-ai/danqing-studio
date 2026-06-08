"""
DiffRhythm 2 MuQ-MuLan text style encoder — MLX inference path.

Loads text-tower weights from the bundled ``pytorch_model.bin`` (XLM-RoBERTa + MuQ
transformer head + ``text_to_latents``). Tokenizer uses HuggingFace ``xlm-roberta-base``
(numpy tokenization only).
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Callable

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from backend.engine.common.ops.attention import scaled_dot_product_attention_bhsd_mx
from backend.engine.runtime.mlx_runtime import run_eval

logger = logging.getLogger(__name__)

_STYLE_LATENT_DIM = 512
_ROBERTA_HEADS = 12
_ROBERTA_HEAD_DIM = 64
_ROBERTA_LAYERS = 12
_MUQ_TF_DEPTH = 8
_MUQ_TF_HEADS = 8
_MUQ_TF_HEAD_DIM = 64
_MUQ_ATTN_SCALE = 8.0
_ROBERTA_PADDING_IDX = 1
_TOKENIZER_NAME = "xlm-roberta-base"


def _eval(*vals: Any) -> None:
    run_eval(None, *vals)


def _l2norm(x: mx.array, axis: int = -1, eps: float = 1e-9) -> mx.array:
    denom = mx.sqrt(mx.sum(x * x, axis=axis, keepdims=True) + eps)
    return x / denom


def resolve_mulan_checkpoint(cache_dir: Path) -> Path:
    """Locate ``pytorch_model.bin`` under the MuQ-MuLan HF cache directory."""
    root = Path(cache_dir)
    candidates = sorted(root.rglob("pytorch_model.bin"))
    for path in candidates:
        if "MuQ-MuLan" in str(path):
            return path
    if candidates:
        return candidates[0]
    raise RuntimeError(
        f"MuQ-MuLan checkpoint not found under {root}. "
        "Install diffrhythm-v2 (fp16) so mulan/MuQ-MuLan-large weights are cached."
    )


def _remap_text_weight_key(key: str) -> str | None:
    if key.startswith("mulan.text.model."):
        sub = key[len("mulan.text.model.") :]
        if sub.startswith("pooler."):
            return None
        return f"roberta.{sub}"
    if key.startswith("mulan.text.transformer."):
        return "transformer." + key[len("mulan.text.transformer.") :]
    if key.startswith("mulan.text_to_latents."):
        return key[len("mulan.") :]
    return None


def _torch_to_mx(tensor: Any, array_fn: Callable[[Any], mx.array]) -> mx.array:
    arr = np.asarray(tensor)
    return array_fn(arr)


class MuQLayerNorm(nn.Module):
    """MuQ bias-less LayerNorm with optional ``learned_gamma``."""

    def __init__(self, dim: int, *, scale: bool = True):
        super().__init__()
        self.learned_gamma = mx.ones((dim,)) if scale else None

    def __call__(self, x: mx.array) -> mx.array:
        gamma = self.learned_gamma if self.learned_gamma is not None else mx.ones((x.shape[-1],))
        return mx.fast.layer_norm(x, gamma, mx.zeros((x.shape[-1],)), eps=1e-5)


class MuQGEGLU(nn.Module):
    def __call__(self, x: mx.array) -> mx.array:
        x, gate = mx.split(x, 2, axis=-1)
        return nn.gelu(gate) * x


class MuQFeedForward(nn.Module):
    def __init__(self, dim: int, mult: int = 4):
        super().__init__()
        hidden = int(dim * mult * 2 / 3)
        self.norm = MuQLayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden * 2, bias=False)
        self.act = MuQGEGLU()
        self.fc2 = nn.Linear(hidden, dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.norm(x)
        x = self.act(self.fc1(x))
        return self.fc2(x)


class MuQAttention(nn.Module):
    def __init__(self, dim: int, heads: int = _MUQ_TF_HEADS, dim_head: int = _MUQ_TF_HEAD_DIM):
        super().__init__()
        inner = heads * dim_head
        self.heads = heads
        self.dim_head = dim_head
        self.scale = _MUQ_ATTN_SCALE
        self.norm = MuQLayerNorm(dim)
        self.to_q = nn.Linear(dim, inner, bias=False)
        self.to_kv = nn.Linear(dim, inner * 2, bias=False)
        self.to_out = nn.Linear(inner, dim, bias=False)
        self.q_scale = mx.ones((dim_head,))
        self.k_scale = mx.ones((dim_head,))

    def __call__(self, x: mx.array) -> mx.array:
        b, t, _ = x.shape
        h = self.norm(x)
        q = self.to_q(h).reshape(b, t, self.heads, self.dim_head).transpose(0, 2, 1, 3)
        kv = self.to_kv(h)
        k, v = mx.split(kv, 2, axis=-1)
        k = k.reshape(b, t, self.heads, self.dim_head).transpose(0, 2, 1, 3)
        v = v.reshape(b, t, self.heads, self.dim_head).transpose(0, 2, 1, 3)

        q = _l2norm(q, axis=-1) * self.q_scale.reshape(1, 1, 1, -1)
        k = _l2norm(k, axis=-1) * self.k_scale.reshape(1, 1, 1, -1)
        out = scaled_dot_product_attention_bhsd_mx(mx, q, k, v, scale=self.scale)
        out = out.transpose(0, 2, 1, 3).reshape(b, t, self.heads * self.dim_head)
        return self.to_out(out)


class MuQTransformerBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.attn = MuQAttention(dim)
        self.ff = MuQFeedForward(dim)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.attn(x) + x
        x = self.ff(x) + x
        return x


class MuQTextTransformer(nn.Module):
    def __init__(self, dim: int = 768, depth: int = _MUQ_TF_DEPTH):
        super().__init__()
        self.layers = [MuQTransformerBlock(dim) for _ in range(depth)]

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.layers:
            x = layer(x)
        return x


class XLMRobertaLayer(nn.Module):
    def __init__(self, hidden: int = 768, heads: int = _ROBERTA_HEADS, ff: int = 3072):
        super().__init__()
        self.attention = nn.MultiHeadAttention(hidden, heads, bias=True)
        self.attention_ln = nn.LayerNorm(hidden)
        self.intermediate = nn.Linear(hidden, ff, bias=True)
        self.output_dense = nn.Linear(ff, hidden, bias=True)
        self.output_ln = nn.LayerNorm(hidden)

    def __call__(self, x: mx.array, mask: mx.array | None) -> mx.array:
        attn_out = self.attention(x, x, x, mask)
        x = self.attention_ln(attn_out + x)
        ff = self.intermediate(x)
        ff = nn.gelu(ff)
        ff = self.output_dense(ff)
        return self.output_ln(ff + x)


class XLMRobertaMLX(nn.Module):
    """XLM-RoBERTa encoder (12L) for MuQ text tower."""

    def __init__(self, vocab_size: int = 250_002, hidden: int = 768, max_pos: int = 514):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden)
        self.position_embeddings = nn.Embedding(max_pos, hidden)
        self.token_type_embeddings = nn.Embedding(1, hidden)
        self.embed_ln = nn.LayerNorm(hidden)
        self.layers = [XLMRobertaLayer(hidden) for _ in range(_ROBERTA_LAYERS)]

    @staticmethod
    def _position_ids(input_ids: mx.array) -> mx.array:
        mask = (input_ids != _ROBERTA_PADDING_IDX).astype(mx.int32)
        pos = mx.cumsum(mask, axis=1) * mask
        return (pos + _ROBERTA_PADDING_IDX).astype(mx.int32)

    @staticmethod
    def _extended_mask(attention_mask: mx.array) -> mx.array:
        # (B, L) with 1=keep -> additive mask for MultiHeadAttention
        inv = (1.0 - attention_mask.astype(mx.float32)) * -1e4
        return inv.reshape(attention_mask.shape[0], 1, 1, attention_mask.shape[1])

    def __call__(self, input_ids: mx.array, attention_mask: mx.array) -> mx.array:
        b, seq = input_ids.shape
        pos_ids = self._position_ids(input_ids)
        tok_type = mx.zeros((b, seq), dtype=mx.int32).astype(mx.int32)
        x = (
            self.word_embeddings(input_ids)
            + self.position_embeddings(pos_ids)
            + self.token_type_embeddings(tok_type)
        )
        x = self.embed_ln(x)
        mask = self._extended_mask(attention_mask)
        for layer in self.layers:
            x = layer(x, mask)
        return x


class MuQStyleEncoderMLX(nn.Module):
    """MLX MuQ-MuLan text encoder → L2-normalized 512-d style latent."""

    def __init__(self, cache_dir: Path, mulan_repo_id: str, ctx: Any):
        super().__init__()
        del mulan_repo_id
        self._cache_dir = Path(cache_dir)
        self._ctx = ctx
        self.roberta = XLMRobertaMLX()
        self.transformer = MuQTextTransformer()
        self.text_to_latents = nn.Linear(768, _STYLE_LATENT_DIM, bias=True)
        self._tokenizer: Any = None
        self._loaded = False

    def _tokenizer_hf(self) -> Any:
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                _TOKENIZER_NAME,
                cache_dir=str(self._cache_dir),
            )
        return self._tokenizer

    def load(self) -> None:
        ckpt = resolve_mulan_checkpoint(self._cache_dir)
        array_fn = getattr(self._ctx, "array", mx.array)
        load_mulan_text_weights(self, str(ckpt), array_fn=array_fn)
        self._loaded = True
        logger.info("MuQ-MuLan MLX text tower loaded from %s", ckpt)

    def encode_text(self, style_prompt: str, *, array_fn: Any | None = None) -> mx.array:
        if not self._loaded:
            raise RuntimeError("MuQStyleEncoderMLX.load() must be called first")
        if array_fn is None:
            array_fn = getattr(self._ctx, "array", mx.array)

        text = (style_prompt or "").strip()
        if not text:
            raise RuntimeError("DiffRhythm 2 style prompt must be non-empty")

        tok = self._tokenizer_hf()
        batch = tok([text], return_tensors="np", padding=True)
        input_ids = array_fn(batch["input_ids"].astype(np.int32)).astype(mx.int32)
        attention_mask = array_fn(batch["attention_mask"].astype(np.int32))

        hidden = self.roberta(input_ids, attention_mask)
        hidden = self.transformer(hidden)
        pooled = mx.mean(hidden, axis=1)
        latent = self.text_to_latents(pooled)
        latent = _l2norm(latent, axis=-1)
        _eval(latent)
        if int(latent.shape[-1]) != _STYLE_LATENT_DIM:
            raise RuntimeError(
                f"MuQ MLX latent dim must be {_STYLE_LATENT_DIM}, got {latent.shape}"
            )
        return latent[0] if latent.ndim == 2 else latent


# Remap MuQ transformer keys (PyTorch ModuleList indices) to MLX module paths.
_MUQ_TF_KEY_RE = re.compile(
    r"^transformer\.layers\.(\d+)\.(\d+)\.(.*)$"
)


def _patch_roberta_keys(key: str) -> str | None:
    if not key.startswith("roberta."):
        return key
    k = key[len("roberta.") :]
    k = k.replace("embeddings.word_embeddings", "word_embeddings")
    k = k.replace("embeddings.position_embeddings", "position_embeddings")
    k = k.replace("embeddings.token_type_embeddings", "token_type_embeddings")
    k = k.replace("embeddings.LayerNorm", "embed_ln")
    k = re.sub(
        r"encoder\.layer\.(\d+)\.attention\.self\.query\.(weight|bias)",
        r"layers.\1.attention.query_proj.\2",
        k,
    )
    k = re.sub(
        r"encoder\.layer\.(\d+)\.attention\.self\.key\.(weight|bias)",
        r"layers.\1.attention.key_proj.\2",
        k,
    )
    k = re.sub(
        r"encoder\.layer\.(\d+)\.attention\.self\.value\.(weight|bias)",
        r"layers.\1.attention.value_proj.\2",
        k,
    )
    k = re.sub(
        r"encoder\.layer\.(\d+)\.attention\.output\.dense\.(weight|bias)",
        r"layers.\1.attention.out_proj.\2",
        k,
    )
    k = re.sub(
        r"encoder\.layer\.(\d+)\.attention\.output\.LayerNorm\.(weight|bias)",
        r"layers.\1.attention_ln.\2",
        k,
    )
    k = re.sub(
        r"encoder\.layer\.(\d+)\.intermediate\.dense\.(weight|bias)",
        r"layers.\1.intermediate.\2",
        k,
    )
    k = re.sub(
        r"encoder\.layer\.(\d+)\.output\.dense\.(weight|bias)",
        r"layers.\1.output_dense.\2",
        k,
    )
    k = re.sub(
        r"encoder\.layer\.(\d+)\.output\.LayerNorm\.(weight|bias)",
        r"layers.\1.output_ln.\2",
        k,
    )
    return f"roberta.{k}"


def _patch_muq_transformer_keys(weights: list[tuple[str, mx.array]]) -> list[tuple[str, mx.array]]:
    out: list[tuple[str, mx.array]] = []
    for key, val in weights:
        key = _patch_roberta_keys(key) or key
        m = _MUQ_TF_KEY_RE.match(key)
        if not m:
            out.append((key, val))
            continue
        layer, slot, rest = m.group(1), int(m.group(2)), m.group(3)
        rest = rest.replace("to_out.0.", "to_out.")
        if slot == 0:
            new_key = f"transformer.layers.{layer}.attn.{rest}"
        elif slot == 1:
            mapping = {
                "0.learned_gamma": "ff.norm.learned_gamma",
                "1.weight": "ff.fc1.weight",
                "4.weight": "ff.fc2.weight",
            }
            new_key = f"transformer.layers.{layer}.{mapping.get(rest, rest)}"
        else:
            new_key = key
        out.append((new_key, val))
    return out


def load_mulan_text_weights(module: nn.Module, ckpt_path: str, *, array_fn: Callable[[Any], mx.array]) -> None:
    """Load MuQ text-tower weights from ``pytorch_model.bin`` (numpy unpickle, no torch)."""
    from backend.engine.common.bundle.pytorch_bin_numpy import load_pytorch_bin

    path = Path(ckpt_path)
    if not path.is_file():
        raise RuntimeError(f"MuQ-MuLan checkpoint missing: {path}")

    raw_sd = load_pytorch_bin(path)
    if not isinstance(raw_sd, dict):
        raise RuntimeError(f"MuQ-MuLan checkpoint at {path} must be a state dict")

    mlx_weights: list[tuple[str, mx.array]] = []
    for pt_key, tensor in raw_sd.items():
        mlx_key = _remap_text_weight_key(pt_key)
        if mlx_key is None:
            continue
        mlx_weights.append((mlx_key, _torch_to_mx(tensor, array_fn)))

    mlx_weights = _patch_muq_transformer_keys(mlx_weights)
    module.load_weights(mlx_weights, strict=False)
    _eval(module)
