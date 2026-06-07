"""Flux2 VAE (MLX) decode path aligned with ``Flux2VAE.decode_packed_latents``."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from PIL import Image

from backend.engine.common.vae import vae_output_to_uint8_hwc
from backend.engine.common.attention import scaled_dot_product_attention_bhsd_mx
from backend.engine.runtime._base import RuntimeContext


def _to_nhwc(x: Any) -> Any:
    return mx.transpose(x, (0, 2, 3, 1))


def _to_nchw(x: Any) -> Any:
    return mx.transpose(x, (0, 3, 1, 2))


class _Flux2ResnetBlock2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, eps: float = 1e-6, groups: int = 32):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups=groups, dims=in_channels, eps=eps, pytorch_compatible=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=groups, dims=out_channels, eps=eps, pytorch_compatible=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv_shortcut = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1) if in_channels != out_channels else None
        )

    def __call__(self, hidden_states: mx.array) -> mx.array:
        residual = _to_nhwc(hidden_states)
        h = self.norm1(_to_nhwc(hidden_states).astype(mx.float32)).astype(mx.bfloat16)
        h = nn.silu(h)
        h = self.conv1(h)
        h = self.norm2(h.astype(mx.float32)).astype(mx.bfloat16)
        h = nn.silu(h)
        h = self.conv2(h)
        if self.conv_shortcut is not None:
            residual = self.conv_shortcut(residual)
        return _to_nchw(h + residual)


class _Flux2AttentionBlock(nn.Module):
    def __init__(self, channels: int, groups: int = 32, eps: float = 1e-6):
        super().__init__()
        self.group_norm = nn.GroupNorm(num_groups=groups, dims=channels, eps=eps, pytorch_compatible=True)
        self.to_q = nn.Linear(channels, channels)
        self.to_k = nn.Linear(channels, channels)
        self.to_v = nn.Linear(channels, channels)
        self.to_out = nn.Linear(channels, channels)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        h = _to_nhwc(hidden_states)
        b, hh, ww, c = h.shape
        normed = self.group_norm(h.astype(mx.float32)).astype(mx.bfloat16)
        q = self.to_q(normed).reshape(b, hh * ww, 1, c)
        k = self.to_k(normed).reshape(b, hh * ww, 1, c)
        v = self.to_v(normed).reshape(b, hh * ww, 1, c)
        q = mx.transpose(q, (0, 2, 1, 3))
        k = mx.transpose(k, (0, 2, 1, 3))
        v = mx.transpose(v, (0, 2, 1, 3))
        out = scaled_dot_product_attention_bhsd_mx(
            mx, q, k, v, scale=float(1 / mx.sqrt(q.shape[-1])),
        )
        out = mx.transpose(out, (0, 2, 1, 3)).reshape(b, hh, ww, c)
        out = self.to_out(out)
        return _to_nchw(h + out)


class _Flux2UNetMidBlock2D(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6, groups: int = 32):
        super().__init__()
        self.resnets = [
            _Flux2ResnetBlock2D(channels, channels, eps=eps, groups=groups),
            _Flux2ResnetBlock2D(channels, channels, eps=eps, groups=groups),
        ]
        self.attentions = [_Flux2AttentionBlock(channels, groups=groups, eps=eps)]

    def __call__(self, hidden_states: mx.array) -> mx.array:
        hidden_states = self.resnets[0](hidden_states)
        hidden_states = self.attentions[0](hidden_states)
        hidden_states = self.resnets[1](hidden_states)
        return hidden_states


class _Flux2Upsample2D(nn.Module):
    def __init__(self, channels: int, out_channels: int | None = None):
        super().__init__()
        out_channels = out_channels or channels
        self.conv = nn.Conv2d(channels, out_channels, kernel_size=3, stride=1, padding=1)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        hidden_states = mx.repeat(hidden_states, 2, axis=2)
        hidden_states = mx.repeat(hidden_states, 2, axis=3)
        h = self.conv(_to_nhwc(hidden_states))
        return _to_nchw(h)


class _Flux2UpDecoderBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 3,
        eps: float = 1e-6,
        groups: int = 32,
        add_upsample: bool = True,
    ):
        super().__init__()
        self.resnets = [
            _Flux2ResnetBlock2D(
                in_channels=in_channels if i == 0 else out_channels,
                out_channels=out_channels,
                eps=eps,
                groups=groups,
            )
            for i in range(num_layers)
        ]
        self.upsamplers = [_Flux2Upsample2D(out_channels, out_channels)] if add_upsample else []

    def __call__(self, hidden_states: mx.array) -> mx.array:
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states)
        for upsampler in self.upsamplers:
            hidden_states = upsampler(hidden_states)
        return hidden_states


class _Flux2Decoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 32,
        out_channels: int = 3,
        block_out_channels: tuple[int, ...] = (128, 256, 512, 512),
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, block_out_channels[-1], kernel_size=3, stride=1, padding=1)
        self.mid_block = _Flux2UNetMidBlock2D(
            channels=block_out_channels[-1],
            eps=eps,
            groups=norm_num_groups,
        )
        self.up_blocks = []
        rev = list(reversed(block_out_channels))
        for i, out_ch in enumerate(rev):
            in_ch = out_ch if i == 0 else rev[i - 1]
            is_final = i == len(rev) - 1
            self.up_blocks.append(
                _Flux2UpDecoderBlock2D(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    num_layers=layers_per_block + 1,
                    eps=eps,
                    groups=norm_num_groups,
                    add_upsample=not is_final,
                )
            )
        self.conv_norm_out = nn.GroupNorm(
            num_groups=norm_num_groups,
            dims=block_out_channels[0],
            eps=eps,
            pytorch_compatible=True,
        )
        self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, kernel_size=3, stride=1, padding=1)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        hidden_states = _to_nchw(self.conv_in(_to_nhwc(hidden_states)))
        hidden_states = self.mid_block(hidden_states)
        for up_block in self.up_blocks:
            hidden_states = up_block(hidden_states)
        h = self.conv_norm_out(_to_nhwc(hidden_states).astype(mx.float32)).astype(mx.bfloat16)
        h = nn.silu(h)
        h = self.conv_out(h)
        return _to_nchw(h)


class _Flux2VAE(nn.Module):
    scaling_factor: float = 1.0
    shift_factor: float = 0.0
    latent_channels: int = 32

    def __init__(self):
        super().__init__()
        self.decoder = _Flux2Decoder()
        self.post_quant_conv = nn.Conv2d(
            self.latent_channels,
            self.latent_channels,
            kernel_size=1,
            padding=0,
        )
        self.bn_running_mean = mx.zeros((4 * self.latent_channels,), dtype=mx.float32)
        self.bn_running_var = mx.ones((4 * self.latent_channels,), dtype=mx.float32)
        self.bn_eps = 1e-4

    def decode(self, latents: mx.array) -> mx.array:
        if latents.ndim == 5:
            latents = latents[:, :, 0, :, :]
        latents = (latents / self.scaling_factor) + self.shift_factor
        latents = _to_nhwc(latents)
        latents = self.post_quant_conv(latents)
        latents = _to_nchw(latents)
        return self.decoder(latents)

    def decode_packed_latents(self, packed_latents: mx.array) -> mx.array:
        if packed_latents.ndim == 5:
            packed_latents = packed_latents[:, :, 0, :, :]
        bn_mean = self.bn_running_mean.reshape(1, -1, 1, 1)
        bn_std = mx.sqrt(self.bn_running_var.reshape(1, -1, 1, 1) + self.bn_eps)
        latents = packed_latents * bn_std + bn_mean
        latents = self._unpatchify_latents(latents)
        return self.decode(latents)

    @staticmethod
    def _unpatchify_latents(latents: mx.array) -> mx.array:
        b, c, h, w = latents.shape
        latents = mx.reshape(latents, (b, c // 4, 2, 2, h, w))
        latents = mx.transpose(latents, (0, 1, 4, 2, 5, 3))
        latents = mx.reshape(latents, (b, c // 4, h * 2, w * 2))
        return latents


def _flatten_param_tree(node: Any, prefix: str, out: dict[str, Any]) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            _flatten_param_tree(value, child, out)
        return
    if isinstance(node, list):
        for idx, value in enumerate(node):
            child = f"{prefix}.{idx}" if prefix else str(idx)
            _flatten_param_tree(value, child, out)
        return
    out[prefix] = node


def load_flux2_vae_decoder(
    ctx: RuntimeContext,
    bundle_root: Path,
    *,
    on_log: Any | None = None,
) -> _Flux2VAE:
    """Load Flux2 VAE once; reuse for step previews to avoid reloading every denoise step."""
    vae_dir = bundle_root / "vae"
    if not vae_dir.exists():
        raise RuntimeError(f"Flux2 VAE decode: missing vae dir at {vae_dir}")
    model = _Flux2VAE()
    weights: dict[str, Any] = {}
    for sf in sorted(vae_dir.glob("*.safetensors")):
        weights.update(ctx.load_weights(str(sf)))
    if not weights:
        raise RuntimeError(f"Flux2 VAE decode: no safetensors in {vae_dir}")

    if "bn.running_mean" in weights:
        model.bn_running_mean = weights["bn.running_mean"]
    if "bn.running_var" in weights:
        model.bn_running_var = weights["bn.running_var"]

    with open(vae_dir / "config.json", encoding="utf-8") as f:
        import json

        cfg = json.load(f)
    model.scaling_factor = float(cfg.get("scaling_factor", 1.0))
    model.shift_factor = float(cfg.get("shift_factor", 0.0))
    model.bn_eps = float(cfg.get("batch_norm_eps", 1e-4))

    w = dict(weights)
    if "post_quant_conv.weight" in w:
        w["post_quant_conv.weight"] = mx.transpose(w["post_quant_conv.weight"], (0, 2, 3, 1))
    if "decoder.conv_in.weight" in w:
        w["decoder.conv_in.weight"] = mx.transpose(w["decoder.conv_in.weight"], (0, 2, 3, 1))
    if "decoder.conv_out.weight" in w:
        w["decoder.conv_out.weight"] = mx.transpose(w["decoder.conv_out.weight"], (0, 2, 3, 1))
    for key, value in list(w.items()):
        if ".to_out.0." in key:
            w[key.replace(".to_out.0.", ".to_out.")] = value
            continue
        if ".conv1.weight" in key or ".conv2.weight" in key or ".conv_shortcut.weight" in key:
            w[key] = mx.transpose(value, (0, 2, 3, 1))
        if ".upsamplers.0.conv.weight" in key:
            w[key] = mx.transpose(value, (0, 2, 3, 1))

    flat_params: dict[str, Any] = {}
    _flatten_param_tree(model.parameters(), "", flat_params)
    loaded = 0
    for key, tensor in w.items():
        param = flat_params.get(key)
        if param is None:
            continue
        if getattr(param, "shape", None) != getattr(tensor, "shape", None):
            continue
        param[:] = tensor
        loaded += 1
    ctx.eval(model.parameters())
    if on_log:
        on_log(
            "info",
            f"flux2 preview VAE loaded decoder_tensors={len(w)} loaded_params={loaded}",
        )
    return model


def decode_flux2_latents_with_model(
    ctx: RuntimeContext,
    model: _Flux2VAE,
    latents: Any,
    *,
    on_log: Any | None = None,
) -> Image.Image:
    decoded = model.decode_packed_latents(latents)
    if on_log:
        on_log(
            "info",
            f"vae_decode flux2 preview latent_shape={tuple(latents.shape)} decoded_shape={tuple(decoded.shape)}",
        )
    pixels = vae_output_to_uint8_hwc(decoded, ctx)
    return Image.fromarray(pixels)


def decode_flux2_packed_latents_to_pil(
    ctx: RuntimeContext,
    packed_latents: Any,
    bundle_root: Path,
    *,
    on_log: Any | None = None,
) -> Image.Image:
    model = load_flux2_vae_decoder(ctx, bundle_root, on_log=on_log)
    return decode_flux2_latents_with_model(
        ctx, model, packed_latents, on_log=on_log
    )

