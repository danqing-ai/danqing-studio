"""HunyuanVideo-1.5 I2V — SigLIP vision encoder (MLX, bundle ``image_encoder/``)."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from PIL import Image

from backend.engine.common.ops.attention import scaled_dot_product_attention_bhsd_mx
from backend.engine.runtime.mlx_runtime import run_eval

_HIDDEN = 1152
_HEADS = 16
_HEAD_DIM = _HIDDEN // _HEADS
_LAYERS = 27
_INTER = 4304
_PATCH = 14
_IMAGE = 384
_NUM_PATCHES = (_IMAGE // _PATCH) ** 2


def _gelu_pytorch_tanh(x: mx.array) -> mx.array:
    return x * mx.sigmoid(1.702 * x)


def _preprocess_siglip_rgb(pil: Image.Image) -> np.ndarray:
    """Return float32 NCHW ``[1,3,384,384]`` matching SiglipImageProcessor."""
    if pil.mode != "RGB":
        pil = pil.convert("RGB")
    pil = pil.resize((_IMAGE, _IMAGE), Image.Resampling.BICUBIC)
    arr = np.asarray(pil, dtype=np.float32) * (1.0 / 255.0)
    mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    std = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    arr = (arr - mean) / std
    arr = np.transpose(arr, (2, 0, 1))[None, ...]
    return arr.astype(np.float32)


def _conv2d_weight_torch_to_mlx(w: Any) -> mx.array:
    if isinstance(w, mx.array):
        if w.ndim == 4:
            return mx.transpose(w, (0, 2, 3, 1))
        return w
    arr = np.asarray(w)
    if arr.ndim == 4:
        return mx.array(np.transpose(arr, (0, 2, 3, 1)))
    return mx.array(arr)


class _SiglipEncoderLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(_HIDDEN)
        self.layer_norm2 = nn.LayerNorm(_HIDDEN)
        self.q_proj = nn.Linear(_HIDDEN, _HIDDEN, bias=True)
        self.k_proj = nn.Linear(_HIDDEN, _HIDDEN, bias=True)
        self.v_proj = nn.Linear(_HIDDEN, _HIDDEN, bias=True)
        self.out_proj = nn.Linear(_HIDDEN, _HIDDEN, bias=True)
        self.fc1 = nn.Linear(_HIDDEN, _INTER, bias=True)
        self.fc2 = nn.Linear(_INTER, _HIDDEN, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        b, seq, _ = x.shape
        norm = self.layer_norm1(x)
        q = self.q_proj(norm).reshape(b, seq, _HEADS, _HEAD_DIM).transpose(0, 2, 1, 3)
        k = self.k_proj(norm).reshape(b, seq, _HEADS, _HEAD_DIM).transpose(0, 2, 1, 3)
        v = self.v_proj(norm).reshape(b, seq, _HEADS, _HEAD_DIM).transpose(0, 2, 1, 3)
        attn = scaled_dot_product_attention_bhsd_mx(
            mx, q, k, v, scale=_HEAD_DIM**-0.5,
        )
        attn = attn.transpose(0, 2, 1, 3).reshape(b, seq, _HIDDEN)
        x = x + self.out_proj(attn)
        y = self.layer_norm2(x)
        y = _gelu_pytorch_tanh(self.fc1(y))
        y = self.fc2(y)
        return x + y


class _SiglipVisionMLX(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.patch_embedding = nn.Conv2d(3, _HIDDEN, kernel_size=_PATCH, stride=_PATCH, bias=True)
        self.position_embedding = nn.Embedding(_NUM_PATCHES, _HIDDEN)
        self.layers = [_SiglipEncoderLayer() for _ in range(_LAYERS)]
        self.post_layernorm = nn.LayerNorm(_HIDDEN)

    def __call__(self, pixel_values_nchw: mx.array) -> mx.array:
        x = pixel_values_nchw.transpose(0, 2, 3, 1)
        x = self.patch_embedding(x)
        b, h, w, c = x.shape
        x = x.reshape(b, h * w, c)
        pos = self.position_embedding.weight[: h * w]
        x = x + pos[None, :, :]
        for layer in self.layers:
            x = layer(x)
        return self.post_layernorm(x)


def resolve_hunyuan_image_encoder_dir(bundle_root: Path) -> Path:
    root = Path(bundle_root)
    enc = root / "image_encoder"
    if (enc / "model.safetensors").is_file():
        return enc
    raise RuntimeError(
        f"HunyuanVideo I2V bundle missing SigLIP vision weights under {enc}. "
        "Install the full HunyuanVideo-1.5 bundle (image_encoder/model.safetensors)."
    )


def _load_siglip_weights(model: _SiglipVisionMLX, encoder_dir: Path) -> None:
    st_path = encoder_dir / "model.safetensors"
    flat = dict(mx.load(str(st_path)))
    nested: dict[str, Any] = {
        "patch_embedding": {},
        "position_embedding": {},
        "post_layernorm": {},
        "layers": [{} for _ in range(_LAYERS)],
    }
    for key, tensor in flat.items():
        if not key.startswith("vision_model."):
            continue
        sub = key[len("vision_model.") :]
        if sub.startswith("embeddings.patch_embedding."):
            leaf = sub.split(".")[-1]
            if leaf == "weight":
                nested["patch_embedding"][leaf] = _conv2d_weight_torch_to_mlx(tensor)
            else:
                nested["patch_embedding"][leaf] = tensor
        elif sub == "embeddings.position_embedding.weight":
            nested["position_embedding"]["weight"] = tensor
        elif sub.startswith("post_layernorm."):
            nested["post_layernorm"][sub.split(".")[-1]] = tensor
        elif sub.startswith("encoder.layers."):
            parts = sub.split(".")
            layer_idx = int(parts[2])
            block = ".".join(parts[3:])
            layer = nested["layers"][layer_idx]
            if block.startswith("layer_norm1."):
                layer.setdefault("layer_norm1", {})[block.split(".")[-1]] = tensor
            elif block.startswith("layer_norm2."):
                layer.setdefault("layer_norm2", {})[block.split(".")[-1]] = tensor
            elif block.startswith("self_attn."):
                attn_key = block[len("self_attn.") :]
                proj, leaf = attn_key.split(".")
                name = {"q_proj": "q_proj", "k_proj": "k_proj", "v_proj": "v_proj", "out_proj": "out_proj"}[
                    proj
                ]
                layer.setdefault(name, {})[leaf] = tensor
            elif block.startswith("mlp."):
                mlp_key = block[len("mlp.") :]
                fc, leaf = mlp_key.split(".")
                name = "fc1" if fc == "fc1" else "fc2"
                layer.setdefault(name, {})[leaf] = tensor
    model.patch_embedding.weight = nested["patch_embedding"]["weight"]
    model.patch_embedding.bias = nested["patch_embedding"]["bias"]
    model.position_embedding.weight = nested["position_embedding"]["weight"]
    model.post_layernorm.weight = nested["post_layernorm"]["weight"]
    model.post_layernorm.bias = nested["post_layernorm"]["bias"]
    for i, layer_mod in enumerate(model.layers):
        src = nested["layers"][i]
        for attr in ("layer_norm1", "layer_norm2", "q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"):
            if attr in src:
                getattr(layer_mod, attr).update(src[attr])


@lru_cache(maxsize=4)
def _get_siglip_runtime(bundle_root_str: str) -> _SiglipVisionMLX:
    encoder_dir = resolve_hunyuan_image_encoder_dir(Path(bundle_root_str))
    model = _SiglipVisionMLX()
    _load_siglip_weights(model, encoder_dir)
    run_eval(None, model.parameters())
    return model


def encode_hunyuan_image_embeds(
    ctx: Any,
    image: Image.Image,
    bundle_root: Path | str,
) -> Any:
    """Encode reference frame to DiT ``image_embeds`` (SigLIP last_hidden_state)."""
    root = Path(bundle_root)
    if not root.is_dir():
        raise RuntimeError(f"HunyuanVideo I2V bundle not found: {root}")
    resolve_hunyuan_image_encoder_dir(root)
    siglip = _get_siglip_runtime(str(root.resolve()))
    pixels = _preprocess_siglip_rgb(image)
    array_fn = getattr(ctx, "array", mx.array)
    pixel_mx = array_fn(pixels)
    hidden = siglip(pixel_mx)
    if hasattr(ctx, "eval"):
        ctx.eval(hidden)
    elif hasattr(ctx, "clear_cache"):
        ctx.clear_cache()
    return hidden
