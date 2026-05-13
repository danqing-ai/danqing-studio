"""
平台检测 — 启动时自动发现可用的 GPU 后端。
"""
from __future__ import annotations

import importlib
import platform
import sys


class PlatformInfo:
    """GPU 后端自动检测。"""

    @staticmethod
    def detect() -> list[str]:
        """返回可用的后端列表: ["mlx"] / ["cuda"] / ["mlx", "cuda"]。"""
        backends: list[str] = []

        if sys.platform == "darwin" and platform.machine() == "arm64":
            try:
                importlib.import_module("mlx.core")
                backends.append("mlx")
            except ImportError:
                pass

        try:
            torch = importlib.import_module("torch")
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
            torch = importlib.import_module("torch")
            return bool(torch.cuda.is_available())
        except ImportError:
            return False

    @staticmethod
    def get_mlx_memory_stats() -> dict:
        """Return MLX GPU memory stats (active/cache/peak in GB). Empty dict if MLX unavailable."""
        try:
            mx = importlib.import_module("mlx.core")
            return {
                "active_gb": round(mx.get_active_memory() / (1024**3), 2),
                "cache_gb": round(mx.get_cache_memory() / (1024**3), 2),
                "peak_gb": round(mx.get_peak_memory() / (1024**3), 2),
            }
        except Exception:
            return {}
