"""
SeedVR2 超分 — 与 ``ImagePipeline.run_upscale`` 对接的唯一入口。

超分热路径：``seedvr2.upscale_pipeline.SeedVR2UpscalePipeline``；数值子模块在 ``seedvr2.runtime``。
本模块仅做 bundle 校验与 ``ImagePipeline`` 入口适配。
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Callable


def expected_seedvr2_weight_files(model_key: str) -> tuple[str, ...]:
    if "7b" in model_key.lower():
        return ("seedvr2_ema_7b_fp16.safetensors", "ema_vae_fp16.safetensors")
    return ("seedvr2_ema_3b_fp16.safetensors", "ema_vae_fp16.safetensors")


def validate_seedvr2_bundle(bundle_path: Path, model_key: str) -> None:
    missing = [n for n in expected_seedvr2_weight_files(model_key) if not (bundle_path / n).is_file()]
    if missing:
        raise RuntimeError(
            f"SeedVR2 bundle at {bundle_path} is missing weight file(s): {missing}. "
            "Expected flat directory with `ema_vae_fp16.safetensors` plus "
            "`seedvr2_ema_7b_fp16.safetensors` or `seedvr2_ema_3b_fp16.safetensors` "
            "(see registry `local_path`, e.g. models/Upscaler/seedvr2-7b-fp16)."
        )


def run_seedvr2_upscale(
    *,
    bundle_path: Path,
    model_key: str,
    source_image: Path,
    scale: int,
    softness: float,
    seed: int | None,
    output_png: Path,
    on_log: Callable[[str, str], None] | None = None,
) -> dict[str, Any]:
    """执行 SeedVR2 超分并写出 PNG。由 ``ImagePipeline`` 在 MLX 路径下调用。"""
    validate_seedvr2_bundle(bundle_path, model_key)

    if scale not in (2, 4):
        raise RuntimeError(f"SeedVR2 upscale scale must be 2 or 4, got {scale!r}")
    if not source_image.is_file():
        raise RuntimeError(f"SeedVR2 upscale source image not found: {source_image}")

    from backend.engine.seedvr2.config import ModelConfig
    from backend.engine.seedvr2.runtime.utils.scale_factor import ScaleFactor
    from backend.engine.seedvr2.upscale_pipeline import SeedVR2UpscalePipeline

    if "7b" in model_key.lower():
        model_config = ModelConfig.seedvr2_7b()
    else:
        model_config = ModelConfig.seedvr2_3b()

    pipeline = SeedVR2UpscalePipeline.from_bundle(bundle_path, model_config)
    resolution = ScaleFactor.parse(f"{int(scale)}x")
    soft = max(0.0, min(1.0, float(softness)))
    sd = int(seed) if seed is not None else random.randint(0, 2 ** 31 - 1)

    if on_log:
        on_log(
            "info",
            " ".join(
                [
                    "seedvr2_upscale backend=seedvr2.upscale_pipeline",
                    f"bundle={bundle_path}",
                    f"model_key={model_key}",
                    f"resolution={resolution}",
                    f"softness={soft}",
                    f"seed={sd}",
                ]
            ),
        )

    generated = pipeline.generate_image(
        seed=sd,
        image_path=source_image,
        resolution=resolution,
        softness=soft,
    )
    output_png.parent.mkdir(parents=True, exist_ok=True)
    generated.image.save(str(output_png))

    return {
        "upscale_backend": "seedvr2.upscale_pipeline",
        "seed": sd,
        "softness": soft,
        "scale": int(scale),
        "reference_model_name": getattr(model_config, "model_name", ""),
    }
