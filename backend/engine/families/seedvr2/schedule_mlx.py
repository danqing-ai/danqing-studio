"""Deprecated — scheduler lives in ``job_mlx`` (SeedVR2 upscale hot path)."""
from __future__ import annotations

from .job_mlx import SCHEDULER_REGISTRY, SeedVR2EulerScheduler, try_import_external_scheduler

__all__ = [
    "SCHEDULER_REGISTRY",
    "SeedVR2EulerScheduler",
    "try_import_external_scheduler",
]
