"""Qwen-Image 文本编码（MLX）— Qwen2.5-VL trunk + Image 提示模板。"""
from __future__ import annotations

import math
import json
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
from mlx import nn

from backend.engine.common.ops.attention import (
    rotate_half,
    scaled_dot_product_attention_bhsd_mx,
)
from backend.engine.runtime.mlx import MLXContext
from backend.engine.common.ops.embeddings import (
    pad_ragged_1d_sequences,
    pad_ragged_2d_sequences,
)
from backend.engine.runtime.mlx_runtime import load_weights_dict, run_eval
from backend.engine.families.qwen.weights_mlx import apply_qwen_text_encoder_weights
from backend.engine.common.codecs.text_encoders.qwen3_mlx import MlxSwiGLUMLP

_MLX_CTX = MLXContext()


class QwenRMSNorm(nn.Module):
    """Match reference QwenRMSNorm exactly (fp32 variance path)."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((hidden_size,))
        self.eps = eps

    def __call__(self, hidden_states: mx.array) -> mx.array:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.astype(mx.float32)
        variance = mx.mean(mx.square(hidden_states), axis=-1, keepdims=True)
        hidden_states = hidden_states * mx.rsqrt(variance + self.eps)
        result = self.weight.astype(mx.float32) * hidden_states
        return result.astype(input_dtype)

class QwenRotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 128000,
        base: float = 1000000.0,
        device: str = None,
        scaling_factor: float = 1.0,
        rope_type: str = "default",
        config=None,
    ):
        super().__init__()
        self.inv_freq = 1.0 / (base ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))
        self.attention_scaling = scaling_factor

    def __call__(self, x: mx.array, position_ids: mx.array) -> tuple[mx.array, mx.array]:
        if len(position_ids.shape) == 2:
            batch_size, seq_len = position_ids.shape
            position_ids = mx.broadcast_to(mx.expand_dims(position_ids, axis=0), (3, batch_size, seq_len))

        inv_freq_expanded = mx.expand_dims(mx.expand_dims(self.inv_freq, axis=0), axis=0)
        inv_freq_expanded = mx.expand_dims(inv_freq_expanded, axis=-1)
        inv_freq_expanded = mx.broadcast_to(inv_freq_expanded, (3, position_ids.shape[1], self.inv_freq.shape[0], 1))
        inv_freq_expanded = inv_freq_expanded.astype(mx.float32)
        position_ids_expanded = mx.expand_dims(position_ids, axis=2)
        position_ids_expanded = position_ids_expanded.astype(mx.float32)
        freqs = mx.matmul(inv_freq_expanded, position_ids_expanded)
        freqs = mx.transpose(freqs, (0, 1, 3, 2))
        emb = mx.concatenate([freqs, freqs], axis=-1)
        cos = mx.cos(emb) * self.attention_scaling
        sin = mx.sin(emb) * self.attention_scaling
        return cos.astype(x.dtype), sin.astype(x.dtype)

class QwenAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int = None,
        max_position_embeddings: int = 128000,
        rope_theta: float = 1000000.0,
        rope_scaling: dict = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads or num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.num_key_value_groups = num_attention_heads // self.num_key_value_heads
        self.scaling = 1.0 / math.sqrt(self.head_dim)
        self.q_proj = nn.Linear(hidden_size, num_attention_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(num_attention_heads * self.head_dim, hidden_size, bias=False)
        self.rotary_emb = QwenRotaryEmbedding(
            dim=self.head_dim,
            max_position_embeddings=max_position_embeddings,
            base=rope_theta,
            rope_type="default",
        )
        self.rope_scaling = rope_scaling or {"mrope_section": [16, 24, 24]}

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: mx.array | None = None,
        position_embeddings: tuple[mx.array, mx.array] | None = None,
    ) -> mx.array:
        bsz, q_len, _ = hidden_states.shape
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.reshape(bsz, q_len, self.num_attention_heads, self.head_dim).transpose(0, 2, 1, 3)
        key_states = key_states.reshape(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(0, 2, 1, 3)
        value_states = value_states.reshape(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(0, 2, 1, 3)

        query_states, key_states = QwenAttention._apply_multimodal_rotary_pos_emb(
            q=query_states,
            k=key_states,
            position_embeddings=position_embeddings,
            mrope_section=self.rope_scaling["mrope_section"],
        )

        if self.num_key_value_heads != self.num_attention_heads:
            key_states = QwenAttention._repeat_kv(key_states, self.num_key_value_groups)
            value_states = QwenAttention._repeat_kv(value_states, self.num_key_value_groups)

        mask = attention_mask[:, :, :, : key_states.shape[-2]].astype(query_states.dtype) if attention_mask is not None else None
        attn_output = scaled_dot_product_attention_bhsd_mx(
            mx, query_states, key_states, value_states, scale=self.scaling, mask=mask,
        )
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        return attn_output

    @staticmethod
    def _repeat_kv(hidden_states: mx.array, n_rep: int) -> mx.array:
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        hidden_states = mx.expand_dims(hidden_states, axis=2)
        hidden_states = mx.broadcast_to(hidden_states, (batch, num_key_value_heads, n_rep, slen, head_dim))
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

    @staticmethod
    def _apply_multimodal_rotary_pos_emb(
        q: mx.array,
        k: mx.array,
        position_embeddings: tuple[mx.array, mx.array],
        mrope_section: list[int],
        unsqueeze_dim: int = 1,
    ) -> tuple[mx.array, mx.array]:
        mrope_section_doubled = [s * 2 for s in mrope_section]

        cos, sin = position_embeddings
        cos_chunks = []
        sin_chunks = []
        start_idx = 0
        for section_size in mrope_section_doubled:
            end_idx = start_idx + section_size
            cos_chunk = cos[..., start_idx:end_idx]
            sin_chunk = sin[..., start_idx:end_idx]
            cos_chunks.append(cos_chunk)
            sin_chunks.append(sin_chunk)
            start_idx = end_idx

        cos_selected = [chunk[i % 3] for i, chunk in enumerate(cos_chunks)]
        sin_selected = [chunk[i % 3] for i, chunk in enumerate(sin_chunks)]

        cos_combined = mx.concatenate(cos_selected, axis=-1)
        sin_combined = mx.concatenate(sin_selected, axis=-1)

        if unsqueeze_dim == 1:
            cos_combined = mx.expand_dims(cos_combined, axis=1)
            sin_combined = mx.expand_dims(sin_combined, axis=1)

        orig_q_dtype = q.dtype
        orig_k_dtype = k.dtype
        q = q.astype(mx.float32)
        k = k.astype(mx.float32)
        cos_combined = cos_combined.astype(mx.float32)
        sin_combined = sin_combined.astype(mx.float32)

        q_embed = (q * cos_combined) + (rotate_half(_MLX_CTX, q) * sin_combined)
        k_embed = (k * cos_combined) + (rotate_half(_MLX_CTX, k) * sin_combined)

        q_embed = q_embed.astype(orig_q_dtype)
        k_embed = k_embed.astype(orig_k_dtype)

        return q_embed, k_embed

class QwenEncoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int = 3584,
        num_attention_heads: int = 28,
        num_key_value_heads: int = 4,
        intermediate_size: int = 18944,
        rms_norm_eps: float = 1e-6,
        max_position_embeddings: int = 128000,
        rope_theta: float = 1000000.0,
    ):
        super().__init__()
        self.input_layernorm = QwenRMSNorm(hidden_size, eps=rms_norm_eps)
        self.self_attn = QwenAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
            rope_scaling={"mrope_section": [16, 24, 24]},
        )
        self.post_attention_layernorm = QwenRMSNorm(hidden_size, eps=rms_norm_eps)
        self.mlp = MlxSwiGLUMLP(hidden_size, intermediate_size)

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: mx.array | None = None,
        position_embeddings: tuple[mx.array, mx.array] | None = None,
    ) -> mx.array:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

class QwenEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int = 152064,
        hidden_size: int = 3584,
        num_hidden_layers: int = 28,
        max_position_embeddings: int = 128000,
        rope_theta: float = 1000000.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.image_token_id = 151655

        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = [QwenEncoderLayer() for i in range(num_hidden_layers)]
        self.norm = QwenRMSNorm(hidden_size, eps=1e-6)
        self.rotary_emb = QwenRotaryEmbedding(
            dim=hidden_size // 28,
            max_position_embeddings=max_position_embeddings,
            base=rope_theta,
            rope_type="default",
        )

        self.visual = None

    def get_image_features(self, pixel_values: mx.array, image_grid_thw: mx.array) -> mx.array:
        if self.visual is None:
            raise RuntimeError("Vision transformer not initialized. Call load_visual_weights() first.")

        pixel_values = pixel_values.astype(mx.float32)
        image_embeds = self.visual(pixel_values, image_grid_thw)
        original_split_sizes = image_grid_thw.prod(axis=-1).astype(mx.int32)
        split_sizes = (original_split_sizes // 4).astype(mx.int32)
        split_sizes = [int(s) for s in split_sizes.tolist()]
        split_sizes = [s for s in split_sizes if s > 0]
        image_embeds_split = []
        start_idx = 0
        for split_size in split_sizes:
            end_idx = start_idx + split_size
            image_embeds_split.append(image_embeds[start_idx:end_idx])
            start_idx = end_idx
        return image_embeds_split

    @staticmethod
    def _build_position_ids(batch_size: int, seq_len: int) -> mx.array:
        cache_position = mx.arange(seq_len, dtype=mx.int32)
        position_ids = mx.expand_dims(mx.expand_dims(cache_position, axis=0), axis=0)
        return mx.broadcast_to(position_ids, (3, batch_size, seq_len))

    @staticmethod
    def _build_attention_mask_4d(attention_mask: mx.array, batch_size: int, seq_len: int) -> mx.array:
        padding_mask = mx.where(
            attention_mask == 1,
            mx.zeros_like(attention_mask).astype(mx.float32),
            mx.ones_like(attention_mask).astype(mx.float32) * (-float("inf")),
        )
        padding_mask = mx.expand_dims(mx.expand_dims(padding_mask, axis=1), axis=1)
        idx = mx.arange(seq_len, dtype=mx.int32)
        j = mx.expand_dims(idx, axis=0)
        i = mx.expand_dims(idx, axis=1)
        tri_bool = j > i
        zeros_2d = mx.zeros((seq_len, seq_len)).astype(mx.float32)
        neginf_2d = mx.ones((seq_len, seq_len)).astype(mx.float32) * (-float("inf"))
        causal_tri_mask = mx.where(tri_bool, neginf_2d, zeros_2d)
        causal_tri_mask = mx.expand_dims(mx.expand_dims(causal_tri_mask, axis=0), axis=0)
        causal_tri_mask = mx.broadcast_to(causal_tri_mask, (batch_size, 1, seq_len, seq_len))
        return causal_tri_mask + padding_mask

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: mx.array,
        pixel_values: mx.array | None = None,
        image_grid_thw: mx.array | None = None,
    ) -> mx.array:
        batch_size, seq_len = input_ids.shape
        inputs_embeds = self.embed_tokens(input_ids)

        if pixel_values is not None and image_grid_thw is not None:
            image_embeds_split = self.get_image_features(pixel_values, image_grid_thw)
            image_embeds = mx.concatenate(image_embeds_split, axis=0)

            image_positions = input_ids == self.image_token_id
            n_image_tokens = mx.sum(image_positions).item()

            if n_image_tokens > 0 and image_embeds.shape[0] >= n_image_tokens:
                image_positions_flat = image_positions.flatten()
                inputs_embeds_flat = inputs_embeds.reshape(-1, inputs_embeds.shape[-1])
                image_embeds_to_use = image_embeds

                new_embeds_list = []
                image_idx = 0
                for i in range(len(image_positions_flat)):
                    if image_positions_flat[i] and image_idx < image_embeds_to_use.shape[0]:
                        new_embeds_list.append(image_embeds_to_use[image_idx])
                        image_idx += 1
                    else:
                        new_embeds_list.append(inputs_embeds_flat[i])

                new_embeds = mx.stack(new_embeds_list, axis=0)
                inputs_embeds = new_embeds.reshape(inputs_embeds.shape)
        position_ids = self._build_position_ids(batch_size, seq_len)
        attention_mask_4d = self._build_attention_mask_4d(attention_mask, batch_size, seq_len)
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for i, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states, attention_mask_4d, position_embeddings)

        hidden_states = self.norm(hidden_states)
        return hidden_states

    def encode_hidden_at(
        self,
        input_ids: mx.array,
        attention_mask: mx.array,
        layer_index: int = -3,
    ) -> mx.array:
        """HF ``output_hidden_states[layer_index]`` (0=embeddings, -1=last layer output)."""
        batch_size, seq_len = input_ids.shape
        inputs_embeds = self.embed_tokens(input_ids)
        position_ids = self._build_position_ids(batch_size, seq_len)
        attention_mask_4d = self._build_attention_mask_4d(attention_mask, batch_size, seq_len)

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        all_hidden_states = [hidden_states]
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask_4d, position_embeddings)
            all_hidden_states.append(hidden_states)

        pick = layer_index if layer_index >= 0 else len(all_hidden_states) + layer_index
        if pick < 0 or pick >= len(all_hidden_states):
            raise RuntimeError(
                f"QwenEncoder layer_index {layer_index} out of range "
                f"(n_hidden_states={len(all_hidden_states)})."
            )
        return all_hidden_states[pick]

class QwenTextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = QwenEncoder()

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: mx.array,
    ) -> tuple[mx.array, mx.array]:
        hidden_states = self.encoder(input_ids, attention_mask)

        prompt_embeds, encoder_attention_mask = QwenTextEncoder._process_text_embeddings_mlx(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            drop_idx=34,
            dtype=mx.bfloat16,
        )

        return prompt_embeds, encoder_attention_mask

    @staticmethod
    def _process_text_embeddings_mlx(hidden_states, attention_mask, drop_idx=1, dtype=mx.float32):
        split_hidden_states = QwenTextEncoder._extract_masked_hidden(hidden_states, attention_mask)
        split_hidden_states = [e[drop_idx:] for e in split_hidden_states]
        attn_mask_list = [mx.ones(e.shape[0], dtype=mx.int32) for e in split_hidden_states]
        max_seq_len = max(int(e.shape[0]) for e in split_hidden_states)
        prompt_embeds = pad_ragged_2d_sequences(
            mx,
            split_hidden_states,
            target_len=max_seq_len,
            dtype=dtype,
            pad_value=0.0,
        )
        encoder_attention_mask = pad_ragged_1d_sequences(
            mx,
            attn_mask_list,
            target_len=max_seq_len,
            dtype=mx.int32,
            pad_value=0.0,
        )
        return prompt_embeds, encoder_attention_mask

    @staticmethod
    def _extract_masked_hidden(hidden_states, attention_mask):
        batch_size = hidden_states.shape[0]
        split_hidden_states = []
        for i in range(batch_size):
            mask = attention_mask[i]
            valid_length = mx.sum(mask).item()
            valid_length = int(valid_length)
            valid_hidden = hidden_states[i, :valid_length, :]
            split_hidden_states.append(valid_hidden)
        return split_hidden_states



def _nested_without_encoder_visual(nested: dict) -> dict:
    """Drop ``encoder.visual`` — bundle 里虽有 vision 权重，但本引擎 ``QwenEncoder`` 未装配 ``visual`` MLX 子模块（仅文本编码）。"""
    out = dict(nested)
    enc = out.get("encoder")
    if isinstance(enc, dict) and "visual" in enc:
        enc = {k: v for k, v in enc.items() if k != "visual"}
        out["encoder"] = enc
    return out


def _read_qwen25vl_encoder_config(
    model_dir: Path,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if config is not None:
        return config
    cfg_path = model_dir / "config.json"
    if cfg_path.is_file():
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
        if isinstance(data.get("text_config"), dict):
            return data["text_config"]
        return data
    return {}


def _glob_qwen25vl_safetensors(model_dir: Path) -> list[Path]:
    globs = sorted(model_dir.glob("*.safetensors"))
    if globs:
        return globs
    te_dir = model_dir / "text_encoder"
    if te_dir.is_dir():
        return sorted(te_dir.glob("*.safetensors"))
    return []


def load_qwen25vl_mlx_encoder(
    model_dir: Path,
    *,
    config: dict[str, Any] | None = None,
    weight_dtype: mx.Dtype | None = None,
    skip_lm_head: bool = False,
    strip_visual: bool = True,
    eval_fn: Any | None = None,
    load_fn: Any | None = None,
    ctx: Any | None = None,
) -> QwenEncoder:
    """Load Qwen2.5-VL MLX encoder trunk (shared by Qwen-Image and HunyuanVideo)."""
    root = Path(model_dir)
    cfg = _read_qwen25vl_encoder_config(root, config)
    encoder = QwenEncoder(
        vocab_size=int(cfg.get("vocab_size", 152064)),
        hidden_size=int(cfg.get("hidden_size", 3584)),
        num_hidden_layers=int(cfg.get("num_hidden_layers", 28)),
        max_position_embeddings=int(cfg.get("max_position_embeddings", 128000)),
        rope_theta=float(cfg.get("rope_theta", 1_000_000.0)),
    )
    globs = _glob_qwen25vl_safetensors(root)
    if not globs:
        raise RuntimeError(f"Qwen2.5-VL: no *.safetensors under {root}")

    raw: dict[str, Any] = {}
    for sf in globs:
        part = load_weights_dict(load_fn, str(sf))
        for key, val in part.items():
            if skip_lm_head and (key == "lm_head.weight" or key.startswith("lm_head.")):
                continue
            raw[key] = val

    nested = apply_qwen_text_encoder_weights(raw)
    if strip_visual:
        nested = _nested_without_encoder_visual(nested)
    enc_nested = nested.get("encoder")
    if not isinstance(enc_nested, dict):
        raise RuntimeError("Qwen2.5-VL: weight remap did not produce encoder.* tree.")
    if strip_visual:
        enc_nested = {k: v for k, v in enc_nested.items() if k != "visual"}
    elif "visual" in enc_nested:
        from backend.engine.families.qwen.edit_encoder_mlx import VisionTransformer

        encoder.visual = VisionTransformer()

    if weight_dtype is not None:
        from backend.engine.runtime.mlx_dtype import cast_floating_mx_tree

        enc_nested = cast_floating_mx_tree(enc_nested, weight_dtype)

    shell = nn.Module()
    shell.encoder = encoder
    shell.update({"encoder": enc_nested})

    if eval_fn is not None:
        run_eval(eval_fn, shell.parameters())
    elif ctx is not None and hasattr(ctx, "eval"):
        ctx.eval(shell.parameters())

    return shell.encoder


class QwenImageTextEncoder:
    """HF tokenizer + 上图 ``QwenTextEncoder`` MLX 模块；权重走 ``WeightMapper``。"""

    def __init__(self, ctx: Any, model_path: str | Path, tokenizer_path: str = "", **_kw: Any):
        self.ctx = ctx
        te_path = Path(model_path)
        self.bundle_root = te_path.parent if te_path.name == "text_encoder" else te_path
        raw_tok = tokenizer_path.strip() if tokenizer_path else ""
        tok_root = Path(raw_tok) if raw_tok and Path(raw_tok).is_dir() else self.bundle_root / "tokenizer"
        if not tok_root.is_dir():
            raise RuntimeError(f"Qwen Image: missing tokenizer directory: {tok_root}")
        # Align with reference tokenizer loader workaround for Qwen2Tokenizer
        # to avoid known vocab/merges loading inconsistencies.
        self._hf = self._load_qwen2_tokenizer(tok_root)
        self._prompt_template = (
            "<|im_start|>system\n"
            "Describe the image by detailing the color, shape, size, texture, quantity, text, "
            "spatial relationships of the objects and background:<|im_end|>\n"
            "<|im_start|>user\n{}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        self.model = QwenTextEncoder()
        self.model.encoder = load_qwen25vl_mlx_encoder(
            self.bundle_root,
            ctx=self.ctx,
            load_fn=getattr(self.ctx, "load_weights", None),
        )

    def encode(self, texts: list[str]) -> tuple[Any, Any]:
        prompt = texts[0] if texts else ""
        filled = self._prompt_template.format(prompt)
        batch = self._hf(
            filled,
            return_tensors="np",
            padding="longest",
            max_length=1058,
            truncation=True,
        )
        input_ids = self.ctx.array(np.asarray(batch["input_ids"], dtype=np.int32), dtype=mx.int32)
        attention_mask = self.ctx.array(
            np.asarray(batch["attention_mask"], dtype=np.int32), dtype=mx.int32
        )
        pe, pm = self.model(input_ids=input_ids, attention_mask=attention_mask)
        self.ctx.eval(pe, pm)
        return pe, pm

    def release_weights(self) -> None:
        """Drop Qwen MLX weights after encode (tokenizer kept)."""
        self.model = None
        clear_cache_fn = getattr(self.ctx, "clear_cache", None)
        if clear_cache_fn is not None:
            clear_cache_fn()
        else:
            import importlib
            importlib.import_module("mlx.core").clear_cache()

    @staticmethod
    def _load_qwen2_tokenizer(tok_root: Path):
        from tokenizers import AddedToken
        from transformers import Qwen2Tokenizer

        vocab_file = tok_root / "vocab.json"
        merges_file = tok_root / "merges.txt"
        cfg_file = tok_root / "tokenizer_config.json"
        if not vocab_file.exists() or not merges_file.exists():
            return Qwen2Tokenizer.from_pretrained(str(tok_root), trust_remote_code=False)

        with open(vocab_file, encoding="utf-8") as f:
            vocab = json.load(f)
        merges: list[tuple[str, str]] = []
        with open(merges_file, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) == 2:
                    merges.append((parts[0], parts[1]))

        init_kwargs: dict[str, Any] = {}
        chat_template = None
        if cfg_file.exists():
            with open(cfg_file, encoding="utf-8") as f:
                cfg = json.load(f)
            added_tokens_decoder = cfg.get("added_tokens_decoder", {})
            if isinstance(added_tokens_decoder, dict) and added_tokens_decoder:
                init_kwargs["added_tokens_decoder"] = {
                    int(k): AddedToken(
                        content=v["content"],
                        lstrip=v.get("lstrip", False),
                        rstrip=v.get("rstrip", False),
                        single_word=v.get("single_word", False),
                        normalized=v.get("normalized", False),
                        special=v.get("special", True),
                    )
                    for k, v in added_tokens_decoder.items()
                }
            chat_template = cfg.get("chat_template")

        tok = Qwen2Tokenizer(vocab=vocab, merges=merges, **init_kwargs)
        if chat_template:
            tok.chat_template = chat_template
        return tok
