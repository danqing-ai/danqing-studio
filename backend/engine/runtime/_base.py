"""
RuntimeContext — 后端无关的张量操作上下文。

模型代码只依赖 RuntimeContext，不直接 import mlx/torch。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar


class RuntimeContext(ABC):
    """后端无关的张量操作上下文。"""

    backend: ClassVar[str]  # "mlx" | "cuda"

    # ------------------------------------------------------------------
    # 模块工厂
    # ------------------------------------------------------------------

    @abstractmethod
    def Linear(self, in_features: int, out_features: int, bias: bool = True) -> Any:
        ...

    @abstractmethod
    def LayerNorm(self, dims: int, eps: float = 1e-5, affine: bool = True, bias: bool = True) -> Any:
        ...

    @abstractmethod
    def RMSNorm(self, dims: int, eps: float = 1e-6) -> Any:
        ...

    @abstractmethod
    def GroupNorm(self, num_groups: int, num_channels: int, eps: float = 1e-5,
                  pytorch_compatible: bool = False) -> Any:
        ...

    @abstractmethod
    def SiLU(self) -> Any:
        ...

    @abstractmethod
    def GELU(self, approximate: str = "tanh") -> Any:
        ...

    @abstractmethod
    def Embedding(self, num_embeddings: int, dim: int) -> Any:
        ...

    @abstractmethod
    def Conv1d(self, in_channels: int, out_channels: int, kernel_size: int,
               stride: int = 1, padding: int = 0, bias: bool = True) -> Any:
        ...

    @abstractmethod
    def Conv2d(self, in_channels: int, out_channels: int, kernel_size: int | tuple,
               stride: int | tuple = 1, padding: int | tuple = 0, bias: bool = True) -> Any:
        ...

    @abstractmethod
    def Conv3d(self, in_channels: int, out_channels: int, kernel_size: int | tuple,
               stride: int | tuple = 1, padding: int | tuple = 0, bias: bool = True) -> Any:
        ...

    @abstractmethod
    def Sequential(self, *layers) -> Any:
        ...

    @abstractmethod
    def ModuleList(self, layers: list) -> Any:
        ...

    # ------------------------------------------------------------------
    # 张量创建
    # ------------------------------------------------------------------

    @abstractmethod
    def zeros(self, shape: tuple, dtype: Any = None) -> Any:
        ...

    @abstractmethod
    def ones(self, shape: tuple, dtype: Any = None) -> Any:
        ...

    @abstractmethod
    def full(self, shape: tuple, value: float, dtype: Any = None) -> Any:
        ...

    @abstractmethod
    def arange(self, start: int, end: int, step: int = 1, dtype: Any = None) -> Any:
        ...

    @abstractmethod
    def randn(self, shape: tuple, dtype: Any = None) -> Any:
        ...

    @abstractmethod
    def seeded_randn(self, shape: tuple, seed: int, dtype: Any = None) -> Any:
        ...

    @abstractmethod
    def conv2d(self, x: Any, weight: Any, stride: int = 1, padding: int = 0) -> Any:
        ...

    @abstractmethod
    def array(self, data: Any, dtype: Any = None) -> Any:
        ...

    @abstractmethod
    def expand_dims(self, x: Any, axis: int) -> Any:
        ...

    @abstractmethod
    def zeros_like(self, x: Any) -> Any:
        ...

    @abstractmethod
    def ones_like(self, x: Any) -> Any:
        ...

    # ------------------------------------------------------------------
    # 张量操作
    # ------------------------------------------------------------------

    @abstractmethod
    def concat(self, tensors: list, axis: int = 0) -> Any:
        ...

    @abstractmethod
    def stack(self, tensors: list, axis: int = 0) -> Any:
        ...

    @abstractmethod
    def where(self, cond: Any, x: Any, y: Any) -> Any:
        ...

    @abstractmethod
    def reshape(self, x: Any, shape: tuple) -> Any:
        ...

    @abstractmethod
    def permute(self, x: Any, dims: tuple) -> Any:
        ...

    @abstractmethod
    def sin(self, x: Any) -> Any:
        ...

    @abstractmethod
    def cos(self, x: Any) -> Any:
        ...

    @abstractmethod
    def exp(self, x: Any) -> Any:
        ...

    @abstractmethod
    def log(self, x: Any) -> Any:
        ...

    @abstractmethod
    def sqrt(self, x: Any) -> Any:
        ...

    @abstractmethod
    def einsum(self, equation: str, *operands) -> Any:
        ...

    @abstractmethod
    def matmul(self, a: Any, b: Any) -> Any:
        ...

    @abstractmethod
    def mul(self, a: Any, b: Any) -> Any:
        ...

    @abstractmethod
    def div(self, a: Any, b: Any) -> Any:
        ...

    @abstractmethod
    def softmax(self, x: Any, axis: int = -1) -> Any:
        ...

    @abstractmethod
    def silu(self, x: Any) -> Any:
        ...

    @abstractmethod
    def gelu(self, x: Any, approximate: str = "none") -> Any:
        ...

    @abstractmethod
    def meshgrid(self, *arrays, indexing: str = "ij") -> Any:
        ...

    @abstractmethod
    def repeat(self, x: Any, repeats: int, axis: int = 0) -> Any:
        ...

    # ------------------------------------------------------------------
    # 微分 / 评估 / 编译（MLX 特需）
    # ------------------------------------------------------------------

    @abstractmethod
    def eval(self, *arrays) -> None:
        ...

    @abstractmethod
    def compile(self, fn, *args, **kwargs) -> Any:
        ...

    # ------------------------------------------------------------------
    # 高级操作
    # ------------------------------------------------------------------

    @abstractmethod
    def attention(self, q: Any, k: Any, v: Any, scale: float | None = None,
                  mask: Any | None = None) -> Any:
        ...

    @abstractmethod
    def interpolate(self, x: Any, scale_factor: float | tuple, mode: str = "bilinear") -> Any:
        ...

    # ------------------------------------------------------------------
    # 内存
    # ------------------------------------------------------------------

    @abstractmethod
    def clear_cache(self) -> None:
        ...

    @abstractmethod
    def active_memory_gb(self) -> float:
        ...

    # ------------------------------------------------------------------
    # 权重 I/O
    # ------------------------------------------------------------------

    @abstractmethod
    def load_weights(self, path: str) -> dict:
        ...

    @abstractmethod
    def save_weights(self, weights: dict, path: str) -> None:
        ...

    # ------------------------------------------------------------------
    # 数据类型
    # ------------------------------------------------------------------

    @abstractmethod
    def float32(self) -> Any:
        ...

    @abstractmethod
    def float16(self) -> Any:
        ...

    @abstractmethod
    def bfloat16(self) -> Any:
        ...

    @abstractmethod
    def int32(self) -> Any:
        ...

    @abstractmethod
    def bool_(self) -> Any:
        ...

    # ------------------------------------------------------------------
    # 便利方法
    # ------------------------------------------------------------------

    def platform(self) -> str:
        return self.backend
