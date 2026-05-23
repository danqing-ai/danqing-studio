"""HeartMuLa CUDA — deferred; use MLX on Apple Silicon."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


class HeartMulaCudaGenerator:
    def __init__(self, ctx: Any, bundle_root: Path):
        self._bundle_root = Path(bundle_root)
        self.last_frame_count = 0

    def load(self) -> None:
        raise RuntimeError(_CUDA_DEFERRED)

    def generate_waveform(self, **kwargs: Any) -> np.ndarray:
        raise RuntimeError(_CUDA_DEFERRED)


_CUDA_DEFERRED = (
    "HeartMuLa CUDA 路径尚未实现；请在 registry backends 中使用 mlx，"
    "或在 Apple Silicon 上通过 danqing-audio 生成。"
)
