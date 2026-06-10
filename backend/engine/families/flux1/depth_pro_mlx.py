"""Apple Depth Pro — native MLX inference (no torch)."""
from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from PIL import Image

from backend.engine.common.ops.attention import scaled_dot_product_attention_bhsd_mx
from backend.engine.runtime.mlx_runtime import run_eval

_INPUT_SIZE = 1536
_PATCH_ENCODER_PATCH = 384
_DINO_IMAGE = 384
_DINO_PATCH = 16
_DINO_HIDDEN = 1024
_DINO_HEADS = 16
_DINO_HEAD_DIM = _DINO_HIDDEN // _DINO_HEADS
_DINO_LAYERS = 24
_DINO_MLP = 4096
_DINO_SEQ = (_DINO_IMAGE // _DINO_PATCH) ** 2 + 1
_OUT_SIZE = _DINO_IMAGE // _DINO_PATCH

_SCALED_RATIOS = (0.25, 0.5, 1.0)
_SCALED_OVERLAP = (0.0, 0.5, 0.25)
_SCALED_DIMS = (1024, 1024, 512)
_INTER_HOOKS = (11, 5)
_INTER_DIMS = (256, 256)
_FUSION_DIM = 256
_MERGE_PAD = 3


def _gelu(x: mx.array) -> mx.array:
    return nn.gelu(x)


def _conv2d_weight_torch_to_mlx(w: Any) -> mx.array:
    arr = np.asarray(w)
    if arr.ndim == 4:
        return mx.array(np.transpose(arr, (0, 2, 3, 1)))
    return mx.array(arr)


def _conv_transpose2d_weight_torch_to_mlx(w: Any) -> mx.array:
    arr = np.asarray(w)
    if arr.ndim == 4:
        return mx.array(np.transpose(arr, (1, 2, 3, 0)))
    return mx.array(arr)


def _nchw_to_nhwc(x: mx.array) -> mx.array:
    return x.transpose(0, 2, 3, 1)


def _nhwc_to_nchw(x: mx.array) -> mx.array:
    return x.transpose(0, 3, 1, 2)


def _resize_bilinear_nchw(arr: np.ndarray, height: int, width: int) -> np.ndarray:
    b, c, _, _ = arr.shape
    out = np.empty((b, c, height, width), dtype=np.float32)
    for bi in range(b):
        for ci in range(c):
            im = Image.fromarray(arr[bi, ci], mode="F")
            im = im.resize((width, height), Image.Resampling.BILINEAR)
            out[bi, ci] = np.asarray(im, dtype=np.float32)
    return out


def _preprocess_depth_pro_rgb(pil: Image.Image) -> np.ndarray:
    """Return float32 NCHW ``[1,3,1536,1536]`` matching DepthProImageProcessor."""
    if pil.mode != "RGB":
        pil = pil.convert("RGB")
    pil = pil.resize((_INPUT_SIZE, _INPUT_SIZE), Image.Resampling.BICUBIC)
    arr = np.asarray(pil, dtype=np.float32) * (1.0 / 255.0)
    mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    std = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    arr = (arr - mean) / std
    return np.transpose(arr, (2, 0, 1))[None, ...].astype(np.float32)


def split_to_patches_np(pixel_values: np.ndarray, patch_size: int, overlap_ratio: float) -> np.ndarray:
    batch_size, num_channels, height, width = pixel_values.shape
    if height == width == patch_size:
        return pixel_values
    stride = int(patch_size * (1.0 - overlap_ratio))
    if stride <= 0:
        raise RuntimeError(f"invalid patch stride for patch_size={patch_size} overlap={overlap_ratio}")
    patches: list[np.ndarray] = []
    for y in range(0, height - patch_size + 1, stride):
        for x in range(0, width - patch_size + 1, stride):
            patches.append(pixel_values[:, :, y : y + patch_size, x : x + patch_size])
    if not patches:
        raise RuntimeError(f"no depth-pro patches from image {height}x{width}")
    return np.concatenate(patches, axis=0)


def reshape_features_np(hidden_states: np.ndarray) -> np.ndarray:
    n_samples, seq_len, hidden_size = hidden_states.shape
    size = int(seq_len**0.5)
    hidden_states = hidden_states[:, -(size * size) :, :]
    hidden_states = hidden_states.reshape(n_samples, size, size, hidden_size)
    return np.transpose(hidden_states, (0, 3, 1, 2))


def merge_patches_np(patches: np.ndarray, batch_size: int, padding: int) -> np.ndarray:
    n_patches, hidden_size, out_size, _ = patches.shape
    n_patches_per_batch = n_patches // batch_size
    if n_patches == batch_size:
        return patches
    sqrt_n = int(round(n_patches_per_batch**0.5))
    if sqrt_n * sqrt_n != n_patches_per_batch:
        sqrt_n = int(math.floor(n_patches_per_batch**0.5))
    new_out_size = sqrt_n * out_size
    if n_patches_per_batch < 4:
        padding = 0
    padding = min(out_size // 4, padding)
    if padding == 0:
        merged = patches.reshape(n_patches_per_batch, batch_size, hidden_size, out_size, out_size)
        merged = merged.transpose(1, 2, 0, 3, 4)
        merged = merged[:, :, : sqrt_n**2, :, :]
        merged = merged.reshape(batch_size, hidden_size, sqrt_n, sqrt_n, out_size, out_size)
        merged = merged.transpose(0, 1, 2, 4, 3, 5)
        return merged.reshape(batch_size, hidden_size, new_out_size, new_out_size)
    boxes: list[np.ndarray] = []
    i = 0
    for h in range(sqrt_n):
        row: list[np.ndarray] = []
        for w in range(sqrt_n):
            box = patches[batch_size * i : batch_size * (i + 1)]
            pad_top = 0 if h == 0 else padding
            pad_left = 0 if w == 0 else padding
            pad_bottom = 0 if h == sqrt_n - 1 else padding
            pad_right = 0 if w == sqrt_n - 1 else padding
            _, _, box_h, box_w = box.shape
            box = box[:, :, pad_top : box_h - pad_bottom, pad_left : box_w - pad_right]
            row.append(box)
            i += 1
        boxes.append(np.concatenate(row, axis=-1))
    return np.concatenate(boxes, axis=-2)


def reconstruct_feature_maps_np(
    hidden_state: np.ndarray,
    *,
    batch_size: int,
    padding: int,
    output_size: tuple[int, int],
) -> np.ndarray:
    features = reshape_features_np(hidden_state)
    features = merge_patches_np(features, batch_size=batch_size, padding=padding)
    return _resize_bilinear_nchw(features, output_size[0], output_size[1])


class _Dinov2LayerMLX(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(_DINO_HIDDEN)
        self.norm2 = nn.LayerNorm(_DINO_HIDDEN)
        self.query = nn.Linear(_DINO_HIDDEN, _DINO_HIDDEN, bias=True)
        self.key = nn.Linear(_DINO_HIDDEN, _DINO_HIDDEN, bias=True)
        self.value = nn.Linear(_DINO_HIDDEN, _DINO_HIDDEN, bias=True)
        self.out_dense = nn.Linear(_DINO_HIDDEN, _DINO_HIDDEN, bias=True)
        self.fc1 = nn.Linear(_DINO_HIDDEN, _DINO_MLP, bias=True)
        self.fc2 = nn.Linear(_DINO_MLP, _DINO_HIDDEN, bias=True)
        self.layer_scale1 = mx.zeros((_DINO_HIDDEN,))
        self.layer_scale2 = mx.zeros((_DINO_HIDDEN,))

    def __call__(self, x: mx.array) -> mx.array:
        y = self.norm1(x)
        b, seq, _ = y.shape
        q = self.query(y).reshape(b, seq, _DINO_HEADS, _DINO_HEAD_DIM).transpose(0, 2, 1, 3)
        k = self.key(y).reshape(b, seq, _DINO_HEADS, _DINO_HEAD_DIM).transpose(0, 2, 1, 3)
        v = self.value(y).reshape(b, seq, _DINO_HEADS, _DINO_HEAD_DIM).transpose(0, 2, 1, 3)
        attn = scaled_dot_product_attention_bhsd_mx(mx, q, k, v, scale=_DINO_HEAD_DIM**-0.5)
        attn = attn.transpose(0, 2, 1, 3).reshape(b, seq, _DINO_HIDDEN)
        attn = self.out_dense(attn) * self.layer_scale1
        x = x + attn
        y = self.norm2(x)
        y = _gelu(self.fc1(y))
        y = self.fc2(y) * self.layer_scale2
        return x + y


class _Dinov2MLX(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cls_token = mx.zeros((1, 1, _DINO_HIDDEN))
        self.patch_embedding = nn.Conv2d(3, _DINO_HIDDEN, kernel_size=_DINO_PATCH, stride=_DINO_PATCH, bias=True)
        self.position_embedding = mx.zeros((1, _DINO_SEQ, _DINO_HIDDEN))
        self.layers = [_Dinov2LayerMLX() for _ in range(_DINO_LAYERS)]

    def __call__(self, pixel_values_nchw: mx.array, *, return_hidden_states: bool = False) -> tuple[mx.array, list[mx.array] | None]:
        x = pixel_values_nchw.transpose(0, 2, 3, 1)
        x = self.patch_embedding(x)
        b, h, w, c = x.shape
        x = x.reshape(b, h * w, c)
        cls = mx.broadcast_to(self.cls_token, (b, 1, _DINO_HIDDEN))
        x = mx.concatenate([cls, x], axis=1)
        x = x + self.position_embedding[:, : x.shape[1], :]
        hidden_states: list[mx.array] = [x]
        for layer in self.layers:
            x = layer(x)
            hidden_states.append(x)
        if return_hidden_states:
            return x, hidden_states
        return x, None


class _DepthProUpsampleBlockMLX(nn.Module):
    def __init__(self, *, input_dims: int, intermediate_dims: int, output_dims: int, n_upsample: int, use_proj: bool, bias: bool) -> None:
        super().__init__()
        self.use_proj = use_proj
        if use_proj:
            self.proj = nn.Conv2d(input_dims, intermediate_dims, kernel_size=1, stride=1, padding=0, bias=bias)
        self.deconvs = [
            nn.ConvTranspose2d(
                intermediate_dims if i == 0 else output_dims,
                output_dims,
                kernel_size=2,
                stride=2,
                padding=0,
                bias=bias,
            )
            for i in range(n_upsample)
        ]

    def __call__(self, x_nhwc: mx.array) -> mx.array:
        if self.use_proj:
            x_nhwc = self.proj(x_nhwc)
        for deconv in self.deconvs:
            x_nhwc = deconv(x_nhwc)
        return x_nhwc


class _DepthProNeckMLX(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.image_block = _DepthProUpsampleBlockMLX(
            input_dims=_DINO_HIDDEN,
            intermediate_dims=_DINO_HIDDEN,
            output_dims=_SCALED_DIMS[0],
            n_upsample=1,
            use_proj=False,
            bias=True,
        )
        self.scaled_blocks = [
            _DepthProUpsampleBlockMLX(
                input_dims=_DINO_HIDDEN,
                intermediate_dims=dim,
                output_dims=dim,
                n_upsample=1,
                use_proj=True,
                bias=False,
            )
            for dim in _SCALED_DIMS
        ]
        self.intermediate_blocks = [
            _DepthProUpsampleBlockMLX(
                input_dims=_DINO_HIDDEN,
                intermediate_dims=_FUSION_DIM if i == 0 else dim,
                output_dims=dim,
                n_upsample=2 + i,
                use_proj=True,
                bias=False,
            )
            for i, dim in enumerate(_INTER_DIMS)
        ]
        self.fuse = nn.Conv2d(_SCALED_DIMS[0] * 2, _SCALED_DIMS[0], kernel_size=1, stride=1, padding=0, bias=True)
        self.projections = [
            nn.Conv2d(_SCALED_DIMS[0], _FUSION_DIM, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(_SCALED_DIMS[0], _FUSION_DIM, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(_SCALED_DIMS[1], _FUSION_DIM, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(_SCALED_DIMS[2], _FUSION_DIM, kernel_size=3, stride=1, padding=1, bias=False),
        ]

    def __call__(self, features_nchw: list[mx.array]) -> list[mx.array]:
        nhwc = [_nchw_to_nhwc(f) for f in features_nchw]
        nhwc[0] = self.image_block(nhwc[0])
        for i, block in enumerate(self.scaled_blocks):
            nhwc[i + 1] = block(nhwc[i + 1])
        for i, block in enumerate(self.intermediate_blocks):
            nhwc[len(_SCALED_RATIOS) + 1 + i] = block(nhwc[len(_SCALED_RATIOS) + 1 + i])
        global_feat = mx.concatenate([nhwc[1], nhwc[0]], axis=-1)
        global_feat = self.fuse(global_feat)
        nhwc = [global_feat, *nhwc[2:]]
        out: list[mx.array] = []
        for i, proj in enumerate(self.projections):
            out.append(_nhwc_to_nchw(proj(nhwc[i])))
        out.append(_nhwc_to_nchw(nhwc[4]))
        return out


class _DepthProResidualMLX(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(_FUSION_DIM, _FUSION_DIM, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(_FUSION_DIM, _FUSION_DIM, kernel_size=3, stride=1, padding=1, bias=True)

    def __call__(self, x_nhwc: mx.array) -> mx.array:
        residual = x_nhwc
        x_nhwc = nn.relu(x_nhwc)
        x_nhwc = self.conv1(x_nhwc)
        x_nhwc = nn.relu(x_nhwc)
        x_nhwc = self.conv2(x_nhwc)
        return x_nhwc + residual


class _DepthProFusionLayerMLX(nn.Module):
    def __init__(self, *, use_deconv: bool) -> None:
        super().__init__()
        self.res1 = _DepthProResidualMLX()
        self.res2 = _DepthProResidualMLX()
        self.use_deconv = use_deconv
        if use_deconv:
            self.deconv = nn.ConvTranspose2d(_FUSION_DIM, _FUSION_DIM, kernel_size=2, stride=2, padding=0, bias=False)
        self.projection = nn.Conv2d(_FUSION_DIM, _FUSION_DIM, kernel_size=1, stride=1, padding=0, bias=True)

    def __call__(self, hidden_nhwc: mx.array, residual_nhwc: mx.array | None = None) -> mx.array:
        if residual_nhwc is not None:
            hidden_nhwc = hidden_nhwc + self.res1(residual_nhwc)
        hidden_nhwc = self.res2(hidden_nhwc)
        if self.use_deconv:
            hidden_nhwc = self.deconv(hidden_nhwc)
        return self.projection(hidden_nhwc)


class _DepthProFusionStageMLX(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        n_inter = len(_SCALED_RATIOS) + len(_INTER_HOOKS) - 1
        self.intermediate = [_DepthProFusionLayerMLX(use_deconv=True) for _ in range(n_inter)]
        self.final = _DepthProFusionLayerMLX(use_deconv=False)

    def __call__(self, hidden_states_nchw: list[mx.array]) -> list[mx.array]:
        nhwc = [_nchw_to_nhwc(h) for h in hidden_states_nchw]
        fused: list[mx.array] = []
        fused_state: mx.array | None = None
        for hidden, layer in zip(nhwc[:-1], self.intermediate):
            if fused_state is None:
                fused_state = layer(hidden)
            else:
                fused_state = layer(fused_state, hidden)
            fused.append(fused_state)
        fused_state = self.final(fused_state, nhwc[-1])
        fused.append(fused_state)
        return [_nhwc_to_nchw(f) for f in fused]


class _DepthProHeadMLX(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv0 = nn.Conv2d(_FUSION_DIM, _FUSION_DIM // 2, kernel_size=3, stride=1, padding=1, bias=True)
        self.deconv = nn.ConvTranspose2d(_FUSION_DIM // 2, _FUSION_DIM // 2, kernel_size=2, stride=2, padding=0, bias=True)
        self.conv1 = nn.Conv2d(_FUSION_DIM // 2, 32, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0, bias=True)

    def __call__(self, x_nchw: mx.array) -> mx.array:
        x = _nchw_to_nhwc(x_nchw)
        x = self.conv0(x)
        x = self.deconv(x)
        x = self.conv1(x)
        x = nn.relu(x)
        x = self.conv2(x)
        x = nn.relu(x)
        return x[..., 0]


class _DepthProMLX(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.patch_encoder = _Dinov2MLX()
        self.image_encoder = _Dinov2MLX()
        self.neck = _DepthProNeckMLX()
        self.fusion = _DepthProFusionStageMLX()
        self.head = _DepthProHeadMLX()

    def _encode_patches(self, pixel_values_np: np.ndarray) -> list[np.ndarray]:
        batch_size, _, height, width = pixel_values_np.shape
        scaled_images: list[np.ndarray] = []
        for ratio in _SCALED_RATIOS:
            h = max(1, int(round(height * ratio)))
            w = max(1, int(round(width * ratio)))
            scaled_images.append(_resize_bilinear_nchw(pixel_values_np, h, w))
        n_patches_per_scale: list[int] = []
        patch_batches: list[np.ndarray] = []
        for i, scaled in enumerate(scaled_images):
            patches = split_to_patches_np(scaled, _PATCH_ENCODER_PATCH, _SCALED_OVERLAP[i])
            n_patches_per_scale.append(patches.shape[0] // batch_size)
            patch_batches.append(patches)
        all_patches = np.concatenate(list(reversed(patch_batches)), axis=0)
        enc_mx = self.patch_encoder(mx.array(all_patches), return_hidden_states=True)
        last_hidden, hidden_states = enc_mx
        run_eval(None, last_hidden)
        last_np = np.array(last_hidden, dtype=np.float32)
        hidden_np = [np.array(h, dtype=np.float32) for h in hidden_states or []]

        split_sizes = list(reversed(n_patches_per_scale))
        chunks = []
        start = 0
        for n in split_sizes:
            end = start + n * batch_size
            chunks.append(last_np[start:end])
            start = end
        chunks = list(reversed(chunks))

        exponent = int(round(math.log2(width / _OUT_SIZE)))
        base_h = height // (2**exponent)
        base_w = width // (2**exponent)

        scaled_features: list[np.ndarray] = []
        for i, hidden in enumerate(chunks):
            pad = int(_MERGE_PAD * (1.0 / _SCALED_RATIOS[i]))
            out_h = base_h * (2**i)
            out_w = base_w * (2**i)
            scaled_features.append(
                reconstruct_feature_maps_np(
                    hidden,
                    batch_size=batch_size,
                    padding=pad,
                    output_size=(out_h, out_w),
                )
            )

        intermediate_features: list[np.ndarray] = []
        for i, hook_id in enumerate(_INTER_HOOKS):
            hidden = hidden_np[hook_id + 1]
            pad = int(_MERGE_PAD * (1.0 / _SCALED_RATIOS[-1]))
            out_h = base_h * (2 ** (len(_SCALED_RATIOS) - 1))
            out_w = base_w * (2 ** (len(_SCALED_RATIOS) - 1))
            intermediate_features.append(
                reconstruct_feature_maps_np(
                    hidden,
                    batch_size=batch_size,
                    padding=pad,
                    output_size=(out_h, out_w),
                )
            )
        return scaled_features + intermediate_features

    def _encode_image(self, pixel_values_np: np.ndarray) -> np.ndarray:
        batch_size, _, height, width = pixel_values_np.shape
        resized = _resize_bilinear_nchw(pixel_values_np, _DINO_IMAGE, _DINO_IMAGE)
        last_hidden, _ = self.image_encoder(mx.array(resized), return_hidden_states=False)
        run_eval(None, last_hidden)
        last_np = np.array(last_hidden, dtype=np.float32)
        exponent = int(round(math.log2(width / _OUT_SIZE)))
        base_h = height // (2**exponent)
        base_w = width // (2**exponent)
        return reconstruct_feature_maps_np(
            last_np,
            batch_size=batch_size,
            padding=0,
            output_size=(base_h, base_w),
        )

    def __call__(self, pixel_values_np: np.ndarray) -> mx.array:
        image_feat = self._encode_image(pixel_values_np)
        patch_feats = self._encode_patches(pixel_values_np)
        features = [mx.array(image_feat), *[mx.array(f) for f in patch_feats]]
        features = self.neck(features)
        fused = self.fusion(features)
        return self.head(fused[-1])


def _assign_dinov2_weights(model: _Dinov2MLX, flat: dict[str, Any], prefix: str) -> None:
    pfx = prefix + "."
    model.cls_token = flat[pfx + "embeddings.cls_token"]
    model.position_embedding = flat[pfx + "embeddings.position_embeddings"]
    model.patch_embedding.weight = _conv2d_weight_torch_to_mlx(flat[pfx + "embeddings.patch_embeddings.projection.weight"])
    model.patch_embedding.bias = flat[pfx + "embeddings.patch_embeddings.projection.bias"]
    for i, layer in enumerate(model.layers):
        lp = pfx + f"encoder.layer.{i}."
        layer.norm1.weight = flat[lp + "norm1.weight"]
        layer.norm1.bias = flat[lp + "norm1.bias"]
        layer.norm2.weight = flat[lp + "norm2.weight"]
        layer.norm2.bias = flat[lp + "norm2.bias"]
        layer.query.weight = flat[lp + "attention.attention.query.weight"]
        layer.query.bias = flat[lp + "attention.attention.query.bias"]
        layer.key.weight = flat[lp + "attention.attention.key.weight"]
        layer.key.bias = flat[lp + "attention.attention.key.bias"]
        layer.value.weight = flat[lp + "attention.attention.value.weight"]
        layer.value.bias = flat[lp + "attention.attention.value.bias"]
        layer.out_dense.weight = flat[lp + "attention.output.dense.weight"]
        layer.out_dense.bias = flat[lp + "attention.output.dense.bias"]
        layer.fc1.weight = flat[lp + "mlp.fc1.weight"]
        layer.fc1.bias = flat[lp + "mlp.fc1.bias"]
        layer.fc2.weight = flat[lp + "mlp.fc2.weight"]
        layer.fc2.bias = flat[lp + "mlp.fc2.bias"]
        layer.layer_scale1 = flat[lp + "layer_scale1.lambda1"]
        layer.layer_scale2 = flat[lp + "layer_scale2.lambda1"]


def _load_depth_pro_weights(model: _DepthProMLX, bundle_root: Path) -> None:
    st_path = bundle_root / "model.safetensors"
    if not st_path.is_file():
        raise RuntimeError(
            f"depth-pro bundle missing model.safetensors at {st_path}; "
            "install depth-pro from Models → Tools"
        )
    flat = dict(mx.load(str(st_path)))

    _assign_dinov2_weights(
        model.patch_encoder,
        flat,
        "depth_pro.encoder.patch_encoder.model",
    )
    _assign_dinov2_weights(
        model.image_encoder,
        flat,
        "depth_pro.encoder.image_encoder.model",
    )

    neck = model.neck
    neck.image_block.deconvs[0].weight = _conv_transpose2d_weight_torch_to_mlx(
        flat["depth_pro.neck.feature_upsample.image_block.layers.0.weight"]
    )
    neck.image_block.deconvs[0].bias = flat["depth_pro.neck.feature_upsample.image_block.layers.0.bias"]

    for i in range(3):
        block = neck.scaled_blocks[i]
        block.proj.weight = _conv2d_weight_torch_to_mlx(
            flat[f"depth_pro.neck.feature_upsample.scaled_images.{i}.layers.0.weight"]
        )
        block.deconvs[0].weight = _conv_transpose2d_weight_torch_to_mlx(
            flat[f"depth_pro.neck.feature_upsample.scaled_images.{i}.layers.1.weight"]
        )

    for i in range(2):
        block = neck.intermediate_blocks[i]
        block.proj.weight = _conv2d_weight_torch_to_mlx(
            flat[f"depth_pro.neck.feature_upsample.intermediate.{i}.layers.0.weight"]
        )
        deconv_layer_ids = (1, 2) if i == 0 else (1, 2, 3)
        for j, key_idx in enumerate(deconv_layer_ids):
            block.deconvs[j].weight = _conv_transpose2d_weight_torch_to_mlx(
                flat[f"depth_pro.neck.feature_upsample.intermediate.{i}.layers.{key_idx}.weight"]
            )

    neck.fuse.weight = _conv2d_weight_torch_to_mlx(flat["depth_pro.neck.fuse_image_with_low_res.weight"])
    neck.fuse.bias = flat["depth_pro.neck.fuse_image_with_low_res.bias"]
    for i in range(4):
        neck.projections[i].weight = _conv2d_weight_torch_to_mlx(
            flat[f"depth_pro.neck.feature_projection.projections.{i}.weight"]
        )

    fusion = model.fusion
    for i, layer in enumerate(fusion.intermediate):
        pfx = f"fusion_stage.intermediate.{i}."
        for conv_name, attr in (("convolution1", "conv1"), ("convolution2", "conv2")):
            for res_name, res_mod in (("residual_layer1", layer.res1), ("residual_layer2", layer.res2)):
                key = pfx + f"{res_name}.{conv_name}."
                getattr(res_mod, attr).weight = _conv2d_weight_torch_to_mlx(flat[key + "weight"])
                getattr(res_mod, attr).bias = flat[key + "bias"]
        layer.deconv.weight = _conv_transpose2d_weight_torch_to_mlx(flat[pfx + "deconv.weight"])
        layer.projection.weight = _conv2d_weight_torch_to_mlx(flat[pfx + "projection.weight"])
        layer.projection.bias = flat[pfx + "projection.bias"]

    fpfx = "fusion_stage.final."
    for conv_name, attr in (("convolution1", "conv1"), ("convolution2", "conv2")):
        for res_name, res_mod in (("residual_layer1", fusion.final.res1), ("residual_layer2", fusion.final.res2)):
            key = fpfx + f"{res_name}.{conv_name}."
            getattr(res_mod, attr).weight = _conv2d_weight_torch_to_mlx(flat[key + "weight"])
            getattr(res_mod, attr).bias = flat[key + "bias"]
    fusion.final.projection.weight = _conv2d_weight_torch_to_mlx(flat[fpfx + "projection.weight"])
    fusion.final.projection.bias = flat[fpfx + "projection.bias"]

    head = model.head
    head.conv0.weight = _conv2d_weight_torch_to_mlx(flat["head.layers.0.weight"])
    head.conv0.bias = flat["head.layers.0.bias"]
    head.deconv.weight = _conv_transpose2d_weight_torch_to_mlx(flat["head.layers.1.weight"])
    head.deconv.bias = flat["head.layers.1.bias"]
    head.conv1.weight = _conv2d_weight_torch_to_mlx(flat["head.layers.2.weight"])
    head.conv1.bias = flat["head.layers.2.bias"]
    head.conv2.weight = _conv2d_weight_torch_to_mlx(flat["head.layers.4.weight"])
    head.conv2.bias = flat["head.layers.4.bias"]


def _assert_bundle(bundle_root: Path) -> None:
    if not bundle_root.is_dir():
        raise RuntimeError(
            f"depth-pro bundle not found at {bundle_root}; install depth-pro from Models → Tools"
        )
    st_path = bundle_root / "model.safetensors"
    if not st_path.is_file():
        raise RuntimeError(
            f"depth-pro bundle missing model.safetensors at {st_path}; "
            "install depth-pro from Models → Tools"
        )
    cfg_path = bundle_root / "config.json"
    if cfg_path.is_file():
        with cfg_path.open(encoding="utf-8") as fh:
            cfg = json.load(fh)
        if cfg.get("model_type") != "depth_pro":
            raise RuntimeError(f"expected depth_pro config at {cfg_path}, got {cfg.get('model_type')!r}")


@lru_cache(maxsize=2)
def _get_depth_pro_model(bundle_root_str: str) -> _DepthProMLX:
    bundle_root = Path(bundle_root_str)
    _assert_bundle(bundle_root)
    model = _DepthProMLX()
    _load_depth_pro_weights(model, bundle_root)
    run_eval(None, model.parameters())
    return model


def estimate_depth_pro_mlx(
    pil: Image.Image,
    *,
    depth_bundle_root: Path,
    on_log: Any = None,
) -> np.ndarray:
    """Run Depth Pro and return raw depth ``[H,W]`` float32 (model output resolution)."""
    _assert_bundle(depth_bundle_root)
    model = _get_depth_pro_model(str(depth_bundle_root.resolve()))
    pixels = _preprocess_depth_pro_rgb(pil)
    depth_mx = model(pixels)
    run_eval(None, depth_mx)
    depth = np.array(depth_mx[0], dtype=np.float32)
    if on_log:
        on_log(
            "info",
            f"depth_pro_mlx depth shape={tuple(depth.shape)} bundle={depth_bundle_root.name}",
        )
    return depth
