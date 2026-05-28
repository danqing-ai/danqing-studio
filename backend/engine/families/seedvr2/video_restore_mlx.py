"""Deprecated — video restoration lives in ``job_mlx``."""
from __future__ import annotations

from .job_mlx import (
    restore_video_chunk_spatiotemporal,
    run_seedvr2_spatiotemporal_video,
)

__all__ = [
    "restore_video_chunk_spatiotemporal",
    "run_seedvr2_spatiotemporal_video",
]
