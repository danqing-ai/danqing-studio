"""Real-ESRGAN upscaler — public stem (MLX dispatch; CUDA fails loud)."""
from __future__ import annotations

from typing import Any, Callable

from backend.engine.families.esrgan.stem_mlx import ESRGANUpscaleRuntime, expected_esrgan_weight_files


def _use_mlx_esrgan() -> bool:
    from backend.engine.platform import PlatformInfo

    return "mlx" in PlatformInfo.detect()


def validate_esrgan_bundle(bundle_path: Any, model_key: str) -> None:
    if _use_mlx_esrgan():
        from backend.engine.families.esrgan.stem_mlx import validate_esrgan_bundle as fn

        return fn(bundle_path, model_key)
    from backend.engine.families.esrgan.stem_cuda import validate_esrgan_bundle as fn

    return fn(bundle_path, model_key)


def load_esrgan_upscale_pipeline(
    *,
    bundle_path: Any,
    model_key: str,
    tile: int = 0,
    on_log: Callable[[str, str], None] | None = None,
    model_cache: Any | None = None,
    cache_key: str | None = None,
    cache_size_gb: float | None = None,
) -> Any:
    if _use_mlx_esrgan():
        from backend.engine.families.esrgan.stem_mlx import load_esrgan_upscale_pipeline as fn

        return fn(
            bundle_path=bundle_path,
            model_key=model_key,
            tile=tile,
            on_log=on_log,
            model_cache=model_cache,
            cache_key=cache_key,
            cache_size_gb=cache_size_gb,
        )
    from backend.engine.families.esrgan.stem_cuda import load_esrgan_upscale_pipeline as fn

    return fn(
        bundle_path=bundle_path,
        model_key=model_key,
        tile=tile,
        on_log=on_log,
        model_cache=model_cache,
        cache_key=cache_key,
        cache_size_gb=cache_size_gb,
    )


def run_esrgan_upscale(
    *,
    bundle_path: Any,
    model_key: str,
    source_image: Any,
    scale: int,
    softness: float,
    seed: int | None,
    output_png: Any,
    on_log: Callable[[str, str], None] | None = None,
    pipeline: ESRGANUpscaleRuntime | None = None,
    tile_size: int = 0,
) -> dict[str, Any]:
    if _use_mlx_esrgan():
        from backend.engine.families.esrgan.stem_mlx import run_esrgan_upscale as fn

        return fn(
            bundle_path=bundle_path,
            model_key=model_key,
            source_image=source_image,
            scale=scale,
            softness=softness,
            seed=seed,
            output_png=output_png,
            on_log=on_log,
            pipeline=pipeline,
            tile_size=tile_size,
        )
    from backend.engine.families.esrgan.stem_cuda import run_esrgan_upscale as fn

    return fn(
        bundle_path=bundle_path,
        model_key=model_key,
        source_image=source_image,
        scale=scale,
        softness=softness,
        seed=seed,
        output_png=output_png,
        on_log=on_log,
        pipeline=pipeline,
        tile_size=tile_size,
    )


__all__ = [
    "ESRGANUpscaleRuntime",
    "expected_esrgan_weight_files",
    "load_esrgan_upscale_pipeline",
    "run_esrgan_upscale",
    "validate_esrgan_bundle",
]
