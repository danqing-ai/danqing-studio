"""
平台检测 — 启动时自动发现可用的 GPU 后端。
"""
from __future__ import annotations

import platform
import sys


class PlatformInfo:
    """GPU 后端自动检测。"""

    @staticmethod
    def detect() -> list[str]:
        """返回可用的后端列表: ["mlx"] / ["cuda"] / ["mlx", "cuda"]。"""
        backends = []

        # MLX: Apple Silicon + Darwin
        if sys.platform == "darwin" and platform.machine() == "arm64":
            try:
                import mlx.core
                backends.append("mlx")
            except ImportError:
                pass

        # CUDA: PyTorch + NVIDIA GPU
        try:
            import torch
            if torch.cuda.is_available():
                backends.append("cuda")
        except ImportError:
            pass

        return backends

    @staticmethod
    def best_available() -> str:
        """返回最佳可用后端名，无可用的则抛异常。"""
        backends = PlatformInfo.detect()
        if not backends:
            raise RuntimeError("No GPU backend available (need MLX on Apple Silicon or CUDA on NVIDIA)")
        return backends[0]

    @staticmethod
    def is_apple_silicon() -> bool:
        return sys.platform == "darwin" and platform.machine() == "arm64"

    @staticmethod
    def is_cuda_available() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
