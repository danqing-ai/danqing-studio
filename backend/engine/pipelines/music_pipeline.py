"""Deprecated alias — use ``audio_pipeline.AudioPipeline``."""

from __future__ import annotations

from backend.engine.pipelines.audio_pipeline import AudioPipeline

MusicPipeline = AudioPipeline

__all__ = ["MusicPipeline", "AudioPipeline"]
