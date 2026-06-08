"""Deprecated alias — use ``upscale_pipeline.UpscalePipeline``."""

from __future__ import annotations

from backend.engine.pipelines.upscale_pipeline import UpscalePipeline

ImageUpscalePipeline = UpscalePipeline

__all__ = ["ImageUpscalePipeline", "UpscalePipeline"]
