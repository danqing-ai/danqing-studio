"""Real-ESRGAN upscaler — MLX job runner (Shape B)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import PIL.Image

from backend.engine.families.real_esrgan.config import (
    RealESRGANVariant,
    expected_weight_files,
    load_model_from_bundle,
    load_variant_config,
)
from backend.engine.families.real_esrgan.utils.upsampler import RealESRGANer


@dataclass
class RealESRGANUpscalePipeline:
    upsampler: RealESRGANer
    variant: RealESRGANVariant

    @classmethod
    def from_bundle(
        cls,
        bundle_path: Path,
        *,
        tile_size: int = 256,
        denoise_strength: float = 1.0,
    ) -> RealESRGANUpscalePipeline:
        model, variant = load_model_from_bundle(
            bundle_path,
            denoise_strength=denoise_strength,
        )
        upsampler = RealESRGANer(
            model,
            variant.netscale,
            tile=tile_size,
            tile_pad=10,
            pre_pad=10,
        )
        return cls(upsampler=upsampler, variant=variant)


def validate_real_esrgan_bundle(bundle_path: Path) -> None:
    missing = [n for n in expected_weight_files() if not (bundle_path / n).is_file()]
    if missing:
        raise RuntimeError(
            f"Real-ESRGAN bundle at {bundle_path} is missing: {missing}. "
            "Install via Models panel (ModelScope mlx-community/*)."
        )


def load_real_esrgan_upscale_pipeline(
    *,
    bundle_path: Path,
    model_key: str,
    model_cache: Any | None = None,
    cache_key: str | None = None,
    cache_size_gb: float | None = None,
    on_log: Callable[[str, str], None] | None = None,
    tile_size: int = 256,
    denoise_strength: float = 1.0,
) -> RealESRGANUpscalePipeline:
    _ = model_key, on_log
    if model_cache is not None and cache_key:
        cached = model_cache.get(cache_key)
        if cached is not None:
            return cached

    pipeline = RealESRGANUpscalePipeline.from_bundle(
        bundle_path,
        tile_size=tile_size,
        denoise_strength=denoise_strength,
    )
    if model_cache is not None and cache_key and cache_size_gb is not None:
        model_cache.put(cache_key, pipeline, cache_size_gb)
    return pipeline


def run_real_esrgan_upscale(
    *,
    bundle_path: Path,
    model_key: str,
    source_image: Path,
    scale: int,
    softness: float,
    seed: int | None,
    output_png: Path,
    on_log: Callable[[str, str], None] | None = None,
    pipeline: RealESRGANUpscalePipeline | None = None,
) -> dict[str, Any]:
    """Run Real-ESRGAN upscale; ``softness`` maps to general-x4v3 denoise blend."""
    _ = seed
    validate_real_esrgan_bundle(bundle_path)

    if scale not in (2, 4):
        raise RuntimeError(f"Real-ESRGAN upscale scale must be 2 or 4, got {scale!r}")
    if not source_image.is_file():
        raise RuntimeError(f"Real-ESRGAN source image not found: {source_image}")

    variant_cfg = load_variant_config(bundle_path)
    denoise_strength = max(0.0, min(1.0, 1.0 - float(softness)))

    if pipeline is None:
        pipeline = RealESRGANUpscalePipeline.from_bundle(
            bundle_path,
            denoise_strength=denoise_strength,
        )
    elif denoise_strength < 1.0 and (bundle_path / "model_wdn.safetensors").is_file():
        pipeline = RealESRGANUpscalePipeline.from_bundle(
            bundle_path,
            denoise_strength=denoise_strength,
        )

    if on_log:
        on_log(
            "info",
            " ".join(
                [
                    "real_esrgan_upscale backend=backend.engine.families.real_esrgan.stem_mlx",
                    f"bundle={bundle_path}",
                    f"model_key={model_key}",
                    f"variant={variant_cfg.name}",
                    f"netscale={variant_cfg.netscale}",
                    f"scale={scale}",
                    f"denoise_strength={denoise_strength:.2f}",
                ]
            ),
        )

    img = np.asarray(PIL.Image.open(source_image))
    outscale = float(scale)
    out, _mode = pipeline.upsampler.enhance(img, outscale=outscale)

    output_png.parent.mkdir(parents=True, exist_ok=True)
    if out.ndim == 2:
        PIL.Image.fromarray(out, mode="L").save(str(output_png))
    elif out.shape[2] == 4:
        PIL.Image.fromarray(out, mode="RGBA").save(str(output_png))
    else:
        PIL.Image.fromarray(out, mode="RGB").save(str(output_png))

    return {
        "upscale_backend": "backend.engine.families.real_esrgan.stem_mlx",
        "variant": variant_cfg.name,
        "scale": int(scale),
        "denoise_strength": denoise_strength,
    }
