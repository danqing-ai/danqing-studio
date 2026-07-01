"""Boogu-Image CUDA generation — not yet implemented (use MLX backend)."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Callable


def resolve_boogu_output_path(work: Path, model_key: str, seed: int) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return str(work / f"{model_key}_{seed}_{ts}.png")


class BooguImageCudaGenerator:
    def __init__(
        self,
        ctx: Any,
        bundle_root: Path,
        *,
        config: Any | None = None,
        entry: Any | None = None,
        version_key: str | None = None,
    ) -> None:
        self._ctx = ctx
        self._bundle_root = bundle_root
        self._config = config
        self._entry = entry
        self._version_key = version_key

    def load(self) -> None:
        raise RuntimeError(
            "Boogu-Image CUDA native inference is not yet implemented. "
            "Install an MLX model version (mlx-q4 / mlx-q8 / bf16) on Apple Silicon."
        )

    def generate_and_save(self, **_kwargs: Any) -> str:
        raise RuntimeError(
            "Boogu-Image CUDA native inference is not yet implemented. "
            "Install an MLX model version (mlx-q4 / mlx-q8 / bf16) on Apple Silicon."
        )
