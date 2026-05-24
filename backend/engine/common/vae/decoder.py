"""
AutoencoderKL VAE Decoder — using RuntimeContext.

Standard SD VAE architecture. Shared by all Flux/Z-Image/Qwen/FIBO image models.
"""
from __future__ import annotations

import importlib
from typing import Any

from backend.engine.runtime._base import RuntimeContext


# NCHW ↔ NHWC 转换（MLX Conv2d/GroupNorm 工作在 NHWC；CUDA 使用标准 NCHW）
def _to_nhwc(ctx, x): return ctx.permute(x, (0, 2, 3, 1))
def _to_nchw(ctx, x): return ctx.permute(x, (0, 3, 1, 2))


def _vae_cuda_nchw(ctx: RuntimeContext) -> bool:
    return getattr(ctx, "backend", None) == "cuda"


class ResnetBlock:
    """VAE ResNet: GroupNorm → SiLU → Conv → GroupNorm → SiLU → Conv → +shortcut。"""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        ctx: RuntimeContext,
        use_shortcut: bool = False,
        *,
        cast_after_norm: bool = False,
        norm_input_fp32: bool = False,
    ):
        nn = ctx; self.ctx = ctx
        self.norm1 = nn.GroupNorm(32, in_ch, eps=1e-6, pytorch_compatible=True)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(32, out_ch, eps=1e-6, pytorch_compatible=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1)
        self.use_shortcut = use_shortcut
        self.conv_shortcut = nn.Conv2d(in_ch, out_ch, 1) if use_shortcut else None
        self.cast_after_norm = bool(cast_after_norm)
        self.norm_input_fp32 = bool(norm_input_fp32)

    def forward(self, x):
        ctx = self.ctx
        if _vae_cuda_nchw(ctx):
            if self.norm_input_fp32:
                torch = importlib.import_module("torch")
                x_norm = x.to(dtype=torch.float32)
            else:
                x_norm = x
            h = self.norm1(x_norm)
            h = ctx.silu(h)
            h = self.conv1(h)
            if self.norm_input_fp32:
                torch = importlib.import_module("torch")
                h = h.to(dtype=torch.float32)
            h = self.norm2(h)
            h = ctx.silu(h)
            h = self.conv2(h)
            if self.conv_shortcut is not None:
                return self.conv_shortcut(x) + h
            return x + h
        h = _to_nhwc(ctx, x)
        if self.norm_input_fp32:
            h = h.astype(importlib.import_module("mlx.core").float32)
        h = self.norm1(h)
        if self.cast_after_norm:
            h = h.astype(importlib.import_module("mlx.core").bfloat16)
        h = ctx.silu(h)
        h = self.conv1(h)
        if self.norm_input_fp32:
            h = h.astype(importlib.import_module("mlx.core").float32)
        h = self.norm2(h)
        if self.cast_after_norm:
            h = h.astype(importlib.import_module("mlx.core").bfloat16)
        h = ctx.silu(h)
        h = self.conv2(h)
        h = _to_nchw(ctx, h)
        if self.conv_shortcut is not None:
            s = _to_nhwc(ctx, x); s = self.conv_shortcut(s); x = _to_nchw(ctx, s)
        return x + h


class SpatialAttention:
    """VAE 中间块的空间注意力 (1-head)。"""

    def __init__(
        self,
        dim: int = 512,
        ctx: RuntimeContext = None,
        *,
        cast_after_norm: bool = False,
        norm_input_fp32: bool = False,
    ):
        nn = ctx; self.ctx = ctx
        self.norm = nn.GroupNorm(32, dim, eps=1e-6, pytorch_compatible=True)
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.to_out = nn.Linear(dim, dim)
        self.cast_after_norm = bool(cast_after_norm)
        self.norm_input_fp32 = bool(norm_input_fp32)

    def forward(self, x):
        ctx = self.ctx
        if _vae_cuda_nchw(ctx):
            if self.norm_input_fp32:
                torch = importlib.import_module("torch")
                h = self.norm(x.to(dtype=torch.float32))
            else:
                h = self.norm(x)
            B, C, H, W = h.shape
            h_bhwc = ctx.permute(h, (0, 2, 3, 1))
            scale = C ** -0.5
            q = ctx.reshape(self.to_q(h_bhwc), (B, H * W, 1, C))
            k = ctx.reshape(self.to_k(h_bhwc), (B, H * W, 1, C))
            v = ctx.reshape(self.to_v(h_bhwc), (B, H * W, 1, C))
            q = ctx.permute(q, (0, 2, 1, 3))
            k = ctx.permute(k, (0, 2, 1, 3))
            v = ctx.permute(v, (0, 2, 1, 3))
            out = ctx.attention(q, k, v, scale=scale)
            out = ctx.permute(out, (0, 2, 1, 3))
            out = ctx.reshape(out, (B, H, W, C))
            out = self.to_out(out)
            out = ctx.permute(out, (0, 3, 1, 2))
            return x + out
        # x: NCHW → NHWC for norm and linear (MLX)
        h = _to_nhwc(ctx, x)
        B, H, W, C = h.shape
        scale = C ** -0.5

        if self.norm_input_fp32:
            h = h.astype(importlib.import_module("mlx.core").float32)
        h = self.norm(h)
        if self.cast_after_norm:
            h = h.astype(importlib.import_module("mlx.core").bfloat16)
        q = ctx.reshape(self.to_q(h), (B, H * W, 1, C))
        k = ctx.reshape(self.to_k(h), (B, H * W, 1, C))
        v = ctx.reshape(self.to_v(h), (B, H * W, 1, C))
        q = ctx.permute(q, (0, 2, 1, 3))
        k = ctx.permute(k, (0, 2, 1, 3))
        v = ctx.permute(v, (0, 2, 1, 3))

        out = ctx.attention(q, k, v, scale=scale)
        out = ctx.permute(out, (0, 2, 1, 3))
        out = ctx.reshape(out, (B, H, W, C))
        out = self.to_out(out)
        out = _to_nchw(ctx, out)
        return x + out


class Upsample:
    """最近邻 2x 上采样 → Conv2d。"""

    def __init__(self, in_ch: int, out_ch: int, ctx: RuntimeContext):
        self.ctx = ctx; self.conv = ctx.Conv2d(in_ch, out_ch, 3, padding=1)

    def forward(self, x):
        ctx = self.ctx
        if _vae_cuda_nchw(ctx):
            F = importlib.import_module("torch.nn.functional")
            x_up = F.interpolate(x, scale_factor=2.0, mode="nearest")
            return self.conv(x_up)
        B, C, H, W = x.shape
        h = _to_nhwc(ctx, x)  # [B, H, W, C]
        # 沿 H 重复: [B, H, W, C] → [B, H, 1, W, C] → concat → [B, H*2, W, C]
        h = ctx.reshape(h, (B, H, 1, W, C))
        h = ctx.concat([h, h], axis=2)
        h = ctx.reshape(h, (B, H * 2, W, C))
        # 沿 W 重复: [B, H*2, W, C] → [B, H*2, W, 1, C] → concat → [B, H*2, W*2, C]
        h = ctx.reshape(h, (B, H * 2, W, 1, C))
        h = ctx.concat([h, h], axis=3)
        h = ctx.reshape(h, (B, H * 2, W * 2, C))
        h = self.conv(h)
        return _to_nchw(ctx, h)


class VAEDecoder:
    """AutoencoderKL 解码器 — 适配 RuntimeContext 的通用 VAE 解码路径。

    架构: ConvIn → Mid(resnet+attn+resnet) → Up1 → Up2 → Up3 → Up4 → NormOut → ConvOut
    """

    def __init__(self, latent_channels: int = 16, ctx: RuntimeContext = None,
                 scaling_factor: float = 1.0, shift_factor: float = 0.0):
        self.ctx = ctx
        nn = ctx; C = latent_channels
        self.scaling_factor = scaling_factor
        self.shift_factor = shift_factor

        self.conv_in = nn.Conv2d(C, 512, 3, padding=1)

        # Mid block
        self.mid_resnet1 = ResnetBlock(512, 512, ctx)
        self.mid_attn = SpatialAttention(512, ctx)
        self.mid_resnet2 = ResnetBlock(512, 512, ctx)

        # Up1: 3×512 → up 2x
        self.up1_resnets = [ResnetBlock(512, 512, ctx) for _ in range(3)]
        self.up1_up = Upsample(512, 512, ctx)
        # Up2: 3×512 → up 2x
        self.up2_resnets = [ResnetBlock(512, 512, ctx) for _ in range(3)]
        self.up2_up = Upsample(512, 512, ctx)
        # Up3: 512→256 (first shortcut) → up 2x
        self.up3_resnets = [
            ResnetBlock(512, 256, ctx, use_shortcut=True),
            ResnetBlock(256, 256, ctx),
            ResnetBlock(256, 256, ctx),
        ]
        self.up3_up = Upsample(256, 256, ctx)
        # Up4: 256→128 (first shortcut)
        self.up4_resnets = [
            ResnetBlock(256, 128, ctx, use_shortcut=True),
            ResnetBlock(128, 128, ctx),
            ResnetBlock(128, 128, ctx),
        ]

        self.norm_out = nn.GroupNorm(32, 128, eps=1e-6, pytorch_compatible=True)
        self.conv_out = nn.Conv2d(128, 3, 3, padding=1)

        self._param_map: dict[str, Any] = {}
        self._built = False

    def _build_param_map(self):
        if self._built: return
        _collect_nn_params(self, "", self._param_map)
        self._built = True

    def load_weights(self, weights: list[tuple[str, Any]], strict=False):
        self._build_param_map()
        loaded, skipped = [], []
        for key, tensor in weights:
            if key in self._param_map:
                p = self._param_map[key]
                if p.shape == tensor.shape:
                    p[:] = tensor; loaded.append(key)
                else:
                    skipped.append(f"{key} shape:{p.shape} vs {tensor.shape}")
            else:
                skipped.append(key)
        if strict and skipped: raise ValueError(f"Weight errors: {skipped}")
        return loaded, skipped

    def forward(self, latents):
        ctx = self.ctx
        # 处理 5D 输入 [B, C, F, H, W]：去掉帧维度
        if latents.ndim == 5:
            latents = latents[:, :, 0, :, :]
        # Apply scaling (matching diffusers/reference VAE)
        if self.scaling_factor != 1.0 or self.shift_factor != 0.0:
            latents = (latents / self.scaling_factor) + self.shift_factor
        if _vae_cuda_nchw(ctx):
            x = self.conv_in(latents)
        else:
            x = _to_nhwc(ctx, latents)
            x = self.conv_in(x)
            x = _to_nchw(ctx, x)

        x = self.mid_resnet1.forward(x)
        x = self.mid_attn.forward(x)
        x = self.mid_resnet2.forward(x)

        for r in self.up1_resnets: x = r.forward(x)
        x = self.up1_up.forward(x)
        for r in self.up2_resnets: x = r.forward(x)
        x = self.up2_up.forward(x)
        for r in self.up3_resnets: x = r.forward(x)
        x = self.up3_up.forward(x)
        for r in self.up4_resnets: x = r.forward(x)

        if _vae_cuda_nchw(ctx):
            x = self.norm_out(x)
            x = ctx.silu(x)
            x = self.conv_out(x)
        else:
            x = _to_nhwc(ctx, x)
            x = self.norm_out(x)
            x = ctx.silu(x)
            x = self.conv_out(x)
            x = _to_nchw(ctx, x)
        return x


def vae_output_to_uint8_hwc(image: Any, ctx: Any | None = None) -> Any:
    """VAE tensor/array [-1, 1] → HWC ``uint8`` numpy.

    Runs ``ctx.eval`` before host read when ``ctx.is_tensor`` applies (MLX lazy graphs).
    Uses float32 + ``rint`` for 8-bit conversion to reduce FP16 pepper-noise in flat areas.
    """
    import numpy as np

    if ctx is not None and getattr(ctx, "is_tensor", lambda _x: False)(image):
        ctx.eval(image)
    if hasattr(image, "detach"):
        arr = image.detach().cpu().numpy()
    elif hasattr(image, "numpy"):
        arr = image.numpy()
    else:
        arr = np.asarray(image)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.shape[0] <= 4:
        arr = np.transpose(arr, (1, 2, 0))
    arr = np.asarray(arr, dtype=np.float32)
    arr = (arr + 1.0) / 2.0 * 255.0
    np.clip(arr, 0.0, 255.0, out=arr)
    return np.rint(arr).astype(np.uint8)


def _collect_nn_params(obj: Any, prefix: str, result: dict):
    if hasattr(obj, 'parameters') and callable(obj.parameters):
        try:
            for pname, ptensor in obj.parameters().items():
                result[f"{prefix}.{pname}" if prefix else pname] = ptensor
            return
        except Exception: pass
    for attr_name in sorted(dir(obj)):
        if attr_name.startswith('_') or attr_name in ('ctx', '_param_map', '_built'):
            continue
        try: attr = getattr(obj, attr_name)
        except Exception: continue
        if attr is None or isinstance(attr, (int, float, str, bool, type)): continue
        new = f"{prefix}.{attr_name}" if prefix else attr_name
        if hasattr(attr, 'parameters') and callable(attr.parameters):
            _collect_nn_params(attr, new, result)
        elif isinstance(attr, (list, tuple)):
            for i, item in enumerate(attr):
                _collect_nn_params(item, f"{new}.{i}", result)
        elif hasattr(attr, '__dict__') and not isinstance(attr, type):
            _collect_nn_params(attr, new, result)
