"""FLUX.1 Redux — SigLIP vision + redux MLP on MLX (no torch)."""
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

_REDUX_DIM = 1152
_TXT_DIM = 4096
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
        # pixel_values: [B,3,H,W] NCHW → patch embed expects NHWC internally in MLX Conv2d
        x = pixel_values_nchw.transpose(0, 2, 3, 1)
        x = self.patch_embedding(x)
        b, h, w, c = x.shape
        x = x.reshape(b, h * w, c)
        pos = self.position_embedding.weight[: h * w]
        x = x + pos[None, :, :]
        for layer in self.layers:
            x = layer(x)
        return self.post_layernorm(x)


class _ReduxProjectorMLX(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.redux_up = nn.Linear(_REDUX_DIM, _TXT_DIM * 3, bias=False)
        self.redux_down = nn.Linear(_TXT_DIM * 3, _TXT_DIM, bias=False)

    def __call__(self, hidden: mx.array) -> mx.array:
        return self.redux_down(nn.silu(self.redux_up(hidden)))


def _load_siglip_weights(model: _SiglipVisionMLX, encoder_dir: Path) -> None:
    st_path = encoder_dir / "model.safetensors"
    if not st_path.is_file():
        raise RuntimeError(f"SigLIP vision weights missing: {st_path}")
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
            leaf = sub.split(".")[-1]
            nested["post_layernorm"][leaf] = tensor
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


def _load_redux_weights(projector: _ReduxProjectorMLX, redux_path: Path) -> None:
    flat = dict(mx.load(str(redux_path)))
    up_key = down_key = None
    for k in flat:
        if "redux_up" in k and k.endswith(".weight"):
            up_key = k
        if "redux_down" in k and k.endswith(".weight"):
            down_key = k
    if up_key is None or down_key is None:
        raise RuntimeError(f"flux-redux bundle missing redux_up/redux_down under {redux_path.parent}")
    projector.redux_up.weight = flat[up_key]
    projector.redux_down.weight = flat[down_key]


def _find_redux_weights(bundle_root: Path) -> Path:
    candidates = [
        bundle_root / "flux1-redux-dev.safetensors",
        bundle_root / "redux" / "diffusion_pytorch_model.safetensors",
    ]
    for p in candidates:
        if p.is_file():
            return p
    for p in sorted(bundle_root.rglob("*.safetensors")):
        if "redux" in p.name.lower() or "siglip" not in p.name.lower():
            return p
    raise RuntimeError(f"no redux safetensors under {bundle_root}")


def _resolve_siglip_encoder_dir(bundle_root: Path) -> Path:
    local = bundle_root / "image_encoder"
    if (local / "model.safetensors").is_file():
        return local
    siglip = bundle_root / "siglip"
    if (siglip / "model.safetensors").is_file():
        return siglip
    raise RuntimeError(
        f"flux-redux bundle missing SigLIP vision weights under {bundle_root}/image_encoder "
        "(install flux-redux from Models → ControlNet)"
    )


@lru_cache(maxsize=4)
def _get_redux_runtime(bundle_root_str: str) -> tuple[_SiglipVisionMLX, _ReduxProjectorMLX]:
    bundle_root = Path(bundle_root_str)
    siglip = _SiglipVisionMLX()
    _load_siglip_weights(siglip, _resolve_siglip_encoder_dir(bundle_root))
    projector = _ReduxProjectorMLX()
    _load_redux_weights(projector, _find_redux_weights(bundle_root))
    run_eval(None, siglip.parameters(), projector.parameters())
    return siglip, projector


def encode_redux_context_tokens_mlx(
    pil: Image.Image,
    *,
    redux_bundle_root: Path,
    on_log: Any = None,
) -> np.ndarray:
    """Return ``[1, seq, 4096]`` float32 tokens for T5 context concat."""
    if not redux_bundle_root.is_dir():
        raise RuntimeError(
            f"flux-redux bundle not found at {redux_bundle_root}; "
            "install flux-redux from Models → ControlNet"
        )
    siglip, projector = _get_redux_runtime(str(redux_bundle_root.resolve()))
    pixels = _preprocess_siglip_rgb(pil)
    pixel_mx = mx.array(pixels)
    hidden = siglip(pixel_mx)
    projected = projector(hidden)
    run_eval(None, projected)
    out = np.array(projected, dtype=np.float32)
    if on_log:
        on_log(
            "info",
            f"redux_encode_mlx tokens shape={tuple(out.shape)} bundle={redux_bundle_root.name}",
        )
    return out
