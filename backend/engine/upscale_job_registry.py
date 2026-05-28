"""Upscale job factory registry (Shape B plugins)."""
from __future__ import annotations

from typing import Any, Callable


_UPSCALE_JOBS: dict[str, tuple[str, str]] = {
    "seedvr2": (
        "backend.engine.families.seedvr2.upscale",
        "run_seedvr2_upscale",
    ),
}


def get_upscale_job_runner(family: str) -> Callable[..., Any]:
    entry = _UPSCALE_JOBS.get(family)
    if entry is None:
        supported = ", ".join(sorted(_UPSCALE_JOBS.keys()))
        raise RuntimeError(
            f"Image upscale is not implemented for family {family!r}; supported: {supported}"
        )
    import importlib

    mod = importlib.import_module(entry[0])
    fn = getattr(mod, entry[1])
    if fn is None:
        raise RuntimeError(f"Upscale job runner {entry[0]}.{entry[1]} is missing")
    return fn
