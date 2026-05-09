"""
CUDA Runtime — 基于 PyTorch 实现 RuntimeContext。
"""
from __future__ import annotations

import gc
from typing import Any

import torch
import torch.nn as nn

from ._base import RuntimeContext


# torch 没有原生 RMSNorm，自行实现
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
    """参考 mlx-video Conv3d 的 weight 转置惯例。"""

    @classmethod
    def from_pretrained(cls, in_channels, out_channels, kernel_size,
                        stride=1, padding=0, bias=True, weight=None):
        """创建 Conv3d 并可选加载转置权重。

        mlx-video 的 WAN Conv3d 权重形状为 [C_out, T, H, W, C_in]，
        PyTorch 期望 [C_out, C_in, T, H, W]，需要转置加载。
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

    # ------------------------------------------------------------------
    # 模块工厂
    # ------------------------------------------------------------------

    def Linear(self, in_features: int, out_features: int, bias: bool = True) -> Any:
        return nn.Linear(in_features, out_features, bias=bias)

    def LayerNorm(self, dims: int, eps: float = 1e-5, affine: bool = True, bias: bool = True) -> Any:
        return nn.LayerNorm(dims, eps=eps, elementwise_affine=affine, bias=bias)

    def RMSNorm(self, dims: int, eps: float = 1e-6) -> Any:
        return _CudaRMSNorm(dims, eps=eps)

    def GroupNorm(self, num_groups: int, num_channels: int, eps: float = 1e-5) -> Any:
        return nn.GroupNorm(num_groups, num_channels, eps=eps)

    def SiLU(self) -> Any:
        return nn.SiLU()

    def GELU(self, approximate: str = "tanh") -> Any:
        return nn.GELU(approximate)

    def Embedding(self, num_embeddings: int, dim: int) -> Any:
        return nn.Embedding(num_embeddings, dim)

    def Conv1d(self, in_channels: int, out_channels: int, kernel_size: int,
               stride: int = 1, padding: int = 0, bias: bool = True) -> Any:
        return nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)

    def Conv2d(self, in_channels: int, out_channels: int, kernel_size: int | tuple,
               stride: int | tuple = 1, padding: int | tuple = 0, bias: bool = True) -> Any:
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)
        return nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)

    def Conv3d(self, in_channels: int, out_channels: int, kernel_size: int | tuple,
               stride: int | tuple = 1, padding: int | tuple = 0, bias: bool = True) -> Any:
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding, padding)
        return nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)

    def Sequential(self, *layers) -> Any:
        return nn.Sequential(*layers)

    def ModuleList(self, layers: list) -> Any:
        return nn.ModuleList(layers)

    # ------------------------------------------------------------------
    # 张量创建
    # ------------------------------------------------------------------

    def zeros(self, shape: tuple, dtype: Any = None) -> Any:
        return torch.zeros(shape, dtype=dtype or torch.float32, device=self._device)

    def ones(self, shape: tuple, dtype: Any = None) -> Any:
        return torch.ones(shape, dtype=dtype or torch.float32, device=self._device)

    def full(self, shape: tuple, value: float, dtype: Any = None) -> Any:
        return torch.full(shape, value, dtype=dtype or torch.float32, device=self._device)

    def arange(self, start: int, end: int, step: int = 1, dtype: Any = None) -> Any:
        return torch.arange(start, end, step, dtype=dtype or torch.int32, device=self._device)

    def randn(self, shape: tuple, dtype: Any = None) -> Any:
        t = torch.randn(shape, dtype=dtype or torch.float32, device=self._device)
        return t

    def seeded_randn(self, shape: tuple, seed: int, dtype: Any = None) -> Any:
        g = torch.Generator(device=self._device)
        g.manual_seed(seed)
        return torch.randn(shape, dtype=dtype or torch.float32, device=self._device, generator=g)

    def conv2d(self, x: Any, weight: Any, stride: int = 1, padding: int = 0) -> Any:
        return torch.nn.functional.conv2d(x, weight, stride=stride, padding=padding)

    def array(self, data: Any, dtype: Any = None) -> Any:
        return torch.tensor(data, dtype=dtype or torch.float32, device=self._device)

    def expand_dims(self, x: Any, axis: int) -> Any:
        return x.unsqueeze(axis)

    def zeros_like(self, x: Any) -> Any:
        return torch.zeros_like(x)

    def ones_like(self, x: Any) -> Any:
        return torch.ones_like(x)

    # ------------------------------------------------------------------
    # 张量操作
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

    def gelu(self, x: Any, approximate: str = "none") -> Any:
        return torch.nn.functional.gelu(x, approximate=approximate)

    def meshgrid(self, *arrays, indexing: str = "ij") -> Any:
        return torch.meshgrid(*arrays, indexing=indexing)

    def repeat(self, x: Any, repeats: int, axis: int = 0) -> Any:
        return x.repeat_interleave(repeats, dim=axis)

    # ------------------------------------------------------------------
    # 微分 / 评估 / 编译
    # ------------------------------------------------------------------

    def eval(self, *arrays) -> None:
        pass

    def compile(self, fn, *args, **kwargs) -> Any:
        return torch.compile(fn, *args, **kwargs)

    # ------------------------------------------------------------------
    # 高级操作
    # ------------------------------------------------------------------

    def attention(self, q: Any, k: Any, v: Any, scale: float | None = None,
                  mask: Any | None = None) -> Any:
        return torch.nn.functional.scaled_dot_product_attention(
            q, k, v, scale=scale, attn_mask=mask,
        )

    def interpolate(self, x: Any, scale_factor: float | tuple, mode: str = "bilinear") -> Any:
        return torch.nn.functional.interpolate(x, scale_factor=scale_factor, mode=mode)

    # ------------------------------------------------------------------
    # 内存
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
    # 权重 I/O
    # ------------------------------------------------------------------

    def load_weights(self, path: str) -> dict:
        import safetensors.torch
        return safetensors.torch.load_file(path, device=self._device)

    def save_weights(self, weights: dict, path: str) -> None:
        import safetensors.torch
        safetensors.torch.save_file(weights, path)

    # ------------------------------------------------------------------
    # 数据类型
    # ------------------------------------------------------------------

    def float32(self) -> Any:
        return torch.float32

    def float16(self) -> Any:
        return torch.float16

    def bfloat16(self) -> Any:
        return torch.bfloat16

    def int32(self) -> Any:
        return torch.int32

    def bool_(self) -> Any:
        return torch.bool
