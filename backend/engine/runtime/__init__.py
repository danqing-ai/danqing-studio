"""Runtime backends (MLX / CUDA) behind :class:`RuntimeContext`."""

from __future__ import annotations

from ._base import RuntimeContext

__all__ = ["RuntimeContext", "MLXContext", "CudaContext"]


def __getattr__(name: str):
    if name == "MLXContext":
        from .mlx import MLXContext as _MLXContext

        return _MLXContext
    if name == "CudaContext":
        from .cuda import CudaContext as _CudaContext

        return _CudaContext
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
