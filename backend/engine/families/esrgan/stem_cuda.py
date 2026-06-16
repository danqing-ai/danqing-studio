"""Real-ESRGAN CUDA placeholder — upscale is MLX-only until a torch RRDB path lands."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

_MSG = (
    "Real-ESRGAN (real-esrgan-x4plus) upscale is MLX-only today. "
    "Use Apple Silicon with MLX, or choose SeedVR2 on CUDA."
)


def expected_esrgan_weight_files() -> tuple[str, ...]:
    return ("model.safetensors",)


def validate_esrgan_bundle(bundle_path: Path, model_key: str) -> None:
    del bundle_path, model_key
    raise RuntimeError(_MSG)


def load_esrgan_upscale_pipeline(
    *,
    bundle_path: Path,
    model_key: str,
    tile: int = 0,
    on_log: Callable[[str, str], None] | None = None,
    model_cache: Any | None = None,
    cache_key: str | None = None,
    cache_size_gb: float | None = None,
) -> Any:
    del bundle_path, model_key, tile, on_log, model_cache, cache_key, cache_size_gb
    raise RuntimeError(_MSG)


def run_esrgan_upscale(
    *,
    bundle_path: Path,
    model_key: str,
    source_image: Path,
    scale: int,
    softness: float,
    seed: int | None,
    output_png: Path,
    on_log: Callable[[str, str], None] | None = None,
    pipeline: Any | None = None,
    tile_size: int = 0,
) -> dict[str, Any]:
    del (
        bundle_path,
        model_key,
        source_image,
        scale,
        softness,
        seed,
        output_png,
        on_log,
        pipeline,
        tile_size,
    )
    raise RuntimeError(_MSG)
