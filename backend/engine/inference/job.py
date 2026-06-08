"""Single-pass job runner (upscale SR, video SR, etc.)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class JobBundle:
    run_fn: Callable[..., Any]
    kwargs: dict[str, Any]


def run_job(bundle: JobBundle) -> Any:
    return bundle.run_fn(**bundle.kwargs)
