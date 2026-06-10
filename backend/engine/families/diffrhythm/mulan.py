"""
DiffRhythm 2 MuQ-MuLan style encoder — backend dispatch (MLX / CUDA).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


class MuQStyleEncoder:
    """Text style encoder for DiffRhythm 2 (MLX preferred on ``mlx`` backend)."""

    def __init__(self, ctx: Any, cache_dir: Path, mulan_repo_id: str):
        self._ctx = ctx
        backend = getattr(ctx, "backend", "mlx")
        if backend == "mlx":
            from .mulan_mlx import MuQStyleEncoderMLX

            self._enc = MuQStyleEncoderMLX(cache_dir, mulan_repo_id, ctx)
        elif backend == "cuda":
            from .mulan_cuda import MuQStyleEncoderTorch

            self._enc = MuQStyleEncoderTorch(cache_dir, mulan_repo_id)
        else:
            raise RuntimeError(f"Unsupported MuQ style encoder backend: {backend!r}")

    def load(self) -> None:
        self._enc.load()

    def encode_text(self, style_prompt: str, *, array_fn: Any) -> Any:
        return self._enc.encode_text(style_prompt, array_fn=array_fn)
