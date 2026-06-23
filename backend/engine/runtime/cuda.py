"""
CUDA Runtime — RuntimeContext implemented via PyTorch.
"""
from __future__ import annotations

import gc
from typing import Any

import torch
import torch.nn as nn

from ._base import RuntimeContext


# torch has no native RMSNorm, implement our own
class _CudaRMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dims))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight.float() * norm).to(dtype)


class _CudaConv3d(nn.Conv3d):
    """Conv3d with optional load-time weight layout transpose (WAN-style checkpoints)."""

    @classmethod
    def from_pretrained(cls, in_channels, out_channels, kernel_size,
                        stride=1, padding=0, bias=True, weight=None):
        """Create Conv3d and optionally load transposed weights.

        Some WAN checkpoints store weight as [C_out, T, H, W, C_in];
        PyTorch expects [C_out, C_in, T, H, W], needs transpose on load.
        """
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding, padding)
        module = cls(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        if weight is not None:
            weight_t = weight.permute(0, 4, 1, 2, 3)
            module.weight.data = weight_t
        return module


class CudaContext(RuntimeContext):
    backend = "cuda"

    def __init__(self, device: str | None = None):
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def device(self):
        return self._device

    def _on_device(self, module: nn.Module) -> nn.Module:
        return module.to(self._device)

    # ------------------------------------------------------------------
    # Module factories
    # ------------------------------------------------------------------

    def Linear(self, in_features: int, out_features: int, bias: bool = True) -> Any:
        return self._on_device(nn.Linear(in_features, out_features, bias=bias))

    def LayerNorm(self, dims: int, eps: float = 1e-5, affine: bool = True, bias: bool = True) -> Any:
        return self._on_device(nn.LayerNorm(dims, eps=eps, elementwise_affine=affine, bias=bias))

    def RMSNorm(self, dims: int, eps: float = 1e-6) -> Any:
        return self._on_device(_CudaRMSNorm(dims, eps=eps))

    def GroupNorm(self, num_groups: int, num_channels: int, eps: float = 1e-5,
                  pytorch_compatible: bool = False, **_kwargs: Any) -> Any:
        del pytorch_compatible  # NCHW path (CUDA VAE) / MLX NHWC handled in ``common.codecs.vae.decoder`` per backend.
        return self._on_device(nn.GroupNorm(num_groups, num_channels, eps=eps))

    def SiLU(self) -> Any:
        return self._on_device(nn.SiLU())

    def GELU(self, approximate: str = "tanh") -> Any:
        return self._on_device(nn.GELU(approximate))

    def Embedding(self, num_embeddings: int, dim: int) -> Any:
        return self._on_device(nn.Embedding(num_embeddings, dim))

    def Conv1d(self, in_channels: int, out_channels: int, kernel_size: int,
               stride: int = 1, padding: int = 0, bias: bool = True) -> Any:
        return self._on_device(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        )

    def Conv2d(self, in_channels: int, out_channels: int, kernel_size: int | tuple,
               stride: int | tuple = 1, padding: int | tuple = 0, bias: bool = True) -> Any:
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)
        return self._on_device(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        )

    def Conv3d(self, in_channels: int, out_channels: int, kernel_size: int | tuple,
               stride: int | tuple = 1, padding: int | tuple = 0, bias: bool = True) -> Any:
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding, padding)
        return self._on_device(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        )

    def Sequential(self, *layers) -> Any:
        return self._on_device(nn.Sequential(*layers))

    def ModuleList(self, layers: list) -> Any:
        return self._on_device(nn.ModuleList(layers))

    def Dropout(self, p: float = 0.0) -> Any:
        return nn.Dropout(p)

    # ------------------------------------------------------------------
    # Tensor creation
    # ------------------------------------------------------------------

    def zeros(self, shape: tuple, dtype: Any = None) -> Any:
        return torch.zeros(shape, dtype=dtype or torch.float32, device=self._device)

    def ones(self, shape: tuple, dtype: Any = None) -> Any:
        return torch.ones(shape, dtype=dtype or torch.float32, device=self._device)

    def full(self, shape: tuple, value: float, dtype: Any = None) -> Any:
        return torch.full(shape, value, dtype=dtype or torch.float32, device=self._device)

    def arange(self, start: int, end: int | None = None, step: int = 1, dtype: Any = None) -> Any:
        if end is None:
            return torch.arange(0, start, step, dtype=dtype or torch.int32, device=self._device)
        return torch.arange(start, end, step, dtype=dtype or torch.int32, device=self._device)

    def randn(self, shape: tuple, dtype: Any = None) -> Any:
        t = torch.randn(shape, dtype=dtype or torch.float32, device=self._device)
        return t

    def seeded_randn(self, shape: tuple, seed: int, dtype: Any = None) -> Any:
        g = torch.Generator(device=self._device)
        g.manual_seed(seed)
        return torch.randn(shape, dtype=dtype or torch.float32, device=self._device, generator=g)

    def seed_random(self, seed: int) -> None:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))

    def conv2d(self, x: Any, weight: Any, stride: int = 1, padding: int = 0) -> Any:
        return torch.nn.functional.conv2d(x, weight, stride=stride, padding=padding)

    def array(self, data: Any, dtype: Any = None) -> Any:
        return torch.tensor(data, dtype=dtype or torch.float32, device=self._device)

    def expand_dims(self, x: Any, axis: int) -> Any:
        return x.unsqueeze(axis)

    def squeeze(self, x: Any, axis: int) -> Any:
        return x.squeeze(axis)

    def zeros_like(self, x: Any) -> Any:
        return torch.zeros_like(x)

    def ones_like(self, x: Any) -> Any:
        return torch.ones_like(x)

    # ------------------------------------------------------------------
    # Tensor operations
    # ------------------------------------------------------------------

    def concat(self, tensors: list, axis: int = 0) -> Any:
        return torch.cat(tensors, dim=axis)

    def stack(self, tensors: list, axis: int = 0) -> Any:
        return torch.stack(tensors, dim=axis)

    def where(self, cond: Any, x: Any, y: Any) -> Any:
        return torch.where(cond, x, y)

    def reshape(self, x: Any, shape: tuple) -> Any:
        return x.reshape(shape)

    def permute(self, x: Any, dims: tuple) -> Any:
        return x.permute(dims)

    def flip(self, x: Any, axis: int = 0) -> Any:
        return torch.flip(x, dims=(axis,))

    def sin(self, x: Any) -> Any:
        return torch.sin(x)

    def cos(self, x: Any) -> Any:
        return torch.cos(x)

    def exp(self, x: Any) -> Any:
        return torch.exp(x)

    def log(self, x: Any) -> Any:
        return torch.log(x)

    def sqrt(self, x: Any) -> Any:
        return torch.sqrt(x)

    def einsum(self, equation: str, *operands) -> Any:
        return torch.einsum(equation, *operands)

    def matmul(self, a: Any, b: Any) -> Any:
        return a @ b

    def mul(self, a: Any, b: Any) -> Any:
        return a * b

    def div(self, a: Any, b: Any) -> Any:
        return a / b

    def softmax(self, x: Any, axis: int = -1) -> Any:
        return torch.softmax(x, dim=axis)

    def silu(self, x: Any) -> Any:
        return torch.nn.functional.silu(x)

    def tanh(self, x: Any) -> Any:
        return torch.tanh(x)

    def split(self, x: Any, indices: list, axis: int = -1) -> list[Any]:
        return list(torch.split(x, indices, dim=axis)) if len(indices) > 1 else [x[:, :indices[0]]] if axis == -1 else list(torch.split(x, indices, dim=axis))

    def broadcast_to(self, x: Any, shape: tuple) -> Any:
        return x.expand(*shape)

    def outer(self, a: Any, b: Any) -> Any:
        return torch.outer(a.flatten(), b.flatten()).reshape(*a.shape, *b.shape)

    def max(self, x: Any) -> Any:
        return torch.max(x)

    def sum(self, x: Any, axis: Any = None) -> Any:
        return torch.sum(x, dim=axis) if axis is not None else torch.sum(x)

    def square(self, x: Any) -> Any:
        return torch.square(x)

    def rsqrt(self, x: Any) -> Any:
        return torch.rsqrt(x)

    def power(self, base: Any, exponent: Any) -> Any:
        return torch.pow(base, exponent)

    def mean(self, x: Any, axis: Any = None, keepdims: bool = False) -> Any:
        return torch.mean(x, dim=axis, keepdim=keepdims)

    def gelu(self, x: Any, approximate: str = "none") -> Any:
        return torch.nn.functional.gelu(x, approximate=approximate)

    def meshgrid(self, *arrays, indexing: str = "ij") -> Any:
        return torch.meshgrid(*arrays, indexing=indexing)

    def repeat(self, x: Any, repeats: int, axis: int = 0) -> Any:
        return x.repeat_interleave(repeats, dim=axis)

    def linspace(self, start: float, end: float, steps: int, dtype: Any = None) -> Any:
        return torch.linspace(start, end, steps, dtype=dtype or torch.float32)

    def dequantize(self, weight: Any, scales: Any, biases: Any, group_size: int, bits: int) -> Any:
        """Unpack MLX **affine** quantized weights (``uint32`` + per-group scales/biases) to float32.

        Matches MLX packing: unsigned integer codes in ``[0, 2**bits - 1]`` per nibble/byte,
        with ``w ≈ scales * q + biases`` along the input axis (see ``mlx.core.quantize`` /
        ``TransformerBase.load_weights``). Supports ``bits`` 4 and 8 only.
        """
        if bits not in (4, 8):
            raise RuntimeError(
                f"CudaContext.dequantize implements MLX affine 4-bit and 8-bit only (got bits={bits})"
            )
        w = weight.to(torch.int64) & 0xFFFFFFFF
        parts: list[Any] = []
        if bits == 8:
            for sh in (0, 8, 16, 24):
                q = (w >> sh) & 255
                parts.append(q.to(torch.float32))
        else:
            for sh in range(0, 32, 4):
                q = (w >> sh) & 15
                parts.append(q.to(torch.float32))
        dq = torch.cat(parts, dim=-1)
        out_features, in_dim = dq.shape
        s = scales.to(device=dq.device, dtype=torch.float32)
        if s.dim() == 1:
            s = s.unsqueeze(0).expand(out_features, -1)
        num_groups = s.shape[-1]
        if in_dim != num_groups * group_size:
            raise RuntimeError(
                f"dequantize shape mismatch: in_dim={in_dim}, num_groups={num_groups}, group_size={group_size}"
            )
        s_rep = s.repeat_interleave(group_size, dim=-1)
        if biases is not None:
            b = biases.to(device=dq.device, dtype=torch.float32)
            if b.dim() == 1:
                b = b.unsqueeze(0).expand(out_features, -1)
            b_rep = b.repeat_interleave(group_size, dim=-1)
            return dq * s_rep + b_rep
        return dq * s_rep

    def is_tensor(self, x: Any) -> bool:
        return isinstance(x, torch.Tensor)

    def is_integer_dtype_tensor(self, x: Any) -> bool:
        if not isinstance(x, torch.Tensor):
            return False
        return x.dtype in (torch.int32, torch.int64)

    def cast(self, x: Any, dtype: Any) -> Any:
        if isinstance(x, torch.Tensor):
            return x.to(dtype=dtype)
        return torch.tensor(x, dtype=dtype, device=self._device)

    def to_numpy(self, x: Any) -> Any:
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return x

    # ------------------------------------------------------------------
    # Differentiation / evaluation / compilation
    # ------------------------------------------------------------------

    def eval(self, *arrays) -> None:
        pass

    def compile(self, fn, *args, **kwargs) -> Any:
        return torch.compile(fn, *args, **kwargs)

    # ------------------------------------------------------------------
    # Advanced operations
    # ------------------------------------------------------------------

    def attention(self, q: Any, k: Any, v: Any, scale: float | None = None,
                  mask: Any | None = None) -> Any:
        return torch.nn.functional.scaled_dot_product_attention(
            q, k, v, scale=scale, attn_mask=mask,
        )

    def interpolate(self, x: Any, scale_factor: float | tuple, mode: str = "bilinear") -> Any:
        return torch.nn.functional.interpolate(x, scale_factor=scale_factor, mode=mode)

    # ------------------------------------------------------------------
    # Memory
    # ------------------------------------------------------------------

    def clear_cache(self) -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def active_memory_gb(self) -> float:
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 ** 3)
        return 0.0

    # ------------------------------------------------------------------
    # Weight I/O
    # ------------------------------------------------------------------

    def load_weights(self, path: str) -> dict:
        if str(path).endswith(".pth"):
            raw = torch.load(path, map_location=self._device, weights_only=True)
            if isinstance(raw, dict) and "state_dict" in raw:
                raw = raw["state_dict"]
            if not isinstance(raw, dict):
                raise RuntimeError(f"Unsupported checkpoint layout in {path!r}")
            return raw
        import safetensors.torch
        return safetensors.torch.load_file(path, device=self._device)

    def save_weights(self, weights: dict, path: str) -> None:
        import safetensors.torch
        safetensors.torch.save_file(weights, path)

    # ------------------------------------------------------------------
    # Data types
    # ------------------------------------------------------------------

    def float32(self) -> Any:
        return torch.float32

    def float64(self) -> Any:
        return torch.float64

    def float16(self) -> Any:
        return torch.float16

    def bfloat16(self) -> Any:
        return torch.bfloat16

    def int32(self) -> Any:
        return torch.int32

    def int64(self) -> Any:
        return torch.int64

    def bool_(self) -> Any:
        return torch.bool
