"""LTX / Shape-C video — family-owned two-stage generator."""

from __future__ import annotations

from typing import Any, Callable

from backend.engine.inference._runtime import inference_span
from backend.engine.pipelines.video_run_common import (
    execute_family_video_avatar,
    execute_family_video_generator,
)


def run_family_video_generator(
    pipeline: Any,
    request: Any,
    ctx_exec: Any,
    *,
    is_edit: bool,
    on_progress: Callable | None = None,
    on_log: Callable | None = None,
) -> tuple[str, dict[str, Any]] | None:
    with inference_span(ctx_exec, "two_stage_paradigm"):
        return execute_family_video_generator(
            pipeline,
            request,
            ctx_exec,
            is_edit=is_edit,
            on_progress=on_progress,
            on_log=on_log,
        )


def run_family_video_avatar(
    pipeline: Any,
    request: Any,
    ctx_exec: Any,
    *,
    on_progress: Callable | None = None,
    on_log: Callable | None = None,
) -> tuple[str, dict[str, Any]] | None:
    with inference_span(ctx_exec, "avatar_paradigm"):
        return execute_family_video_avatar(
            pipeline,
            request,
            ctx_exec,
            on_progress=on_progress,
            on_log=on_log,
        )
