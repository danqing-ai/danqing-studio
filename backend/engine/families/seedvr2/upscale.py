"""SeedVR2 图像超分 — 对外入口（MLX 编排见 ``job_mlx``）。"""
from __future__ import annotations

from backend.engine.families.seedvr2.job_mlx import (
    SeedVR2UpscalePipeline,
    SeedVR2UpscaleRuntime,
    expected_seedvr2_weight_files,
    run_seedvr2_upscale,
    validate_seedvr2_bundle,
)
from backend.engine.families.seedvr2.weights_mlx import ModelConfig

__all__ = [
    "ModelConfig",
    "SeedVR2UpscalePipeline",
    "SeedVR2UpscaleRuntime",
    "expected_seedvr2_weight_files",
    "run_seedvr2_upscale",
    "validate_seedvr2_bundle",
]
