"""Upscale job factory registry (Shape B plugins)."""
from __future__ import annotations

import importlib
from typing import Any, Callable


_UPSCALE_JOBS: dict[str, tuple[str, str]] = {
    "seedvr2": (
        "backend.engine.families.seedvr2.stem",
        "run_seedvr2_upscale",
    ),
}

_UPSCALE_PIPELINE_LOADERS: dict[str, tuple[str, str]] = {
    "seedvr2": (
        "backend.engine.families.seedvr2.stem",
        "load_seedvr2_upscale_pipeline",
    ),
}


def get_upscale_job_runner(family: str) -> Callable[..., Any]:
    entry = _UPSCALE_JOBS.get(family)
    if entry is None:
        supported = ", ".join(sorted(_UPSCALE_JOBS.keys()))
        raise RuntimeError(
            f"Image upscale is not implemented for family {family!r}; supported: {supported}"
        )
    mod = importlib.import_module(entry[0])
    fn = getattr(mod, entry[1])
    if fn is None:
        raise RuntimeError(f"Upscale job runner {entry[0]}.{entry[1]} is missing")
    return fn


def get_upscale_pipeline_loader(family: str) -> Callable[..., Any] | None:
    entry = _UPSCALE_PIPELINE_LOADERS.get(family)
    if entry is None:
        return None
    mod = importlib.import_module(entry[0])
    fn = getattr(mod, entry[1])
    if fn is None:
        raise RuntimeError(f"Upscale pipeline loader {entry[0]}.{entry[1]} is missing")
    return fn
