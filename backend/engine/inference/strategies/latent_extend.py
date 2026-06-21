"""LTX latent-extend long video strategy (wraps existing video create + run_ltx_long_video)."""
from __future__ import annotations

from typing import Any, Callable

from backend.core.contracts import VideoGenerationRequest, VideoLongGenerationRequest
from backend.engine.sessions.engine_dispatch import dispatch_video_create


def run_latent_extend_strategy(
    *,
    request: VideoLongGenerationRequest,
    ctx_exec: Any,
    video_dispatch: dict[str, Any],
    on_progress: Callable | None,
    on_log: Callable | None,
) -> tuple[str, dict[str, Any]]:
    spec = request.long_video
    gen_req = VideoGenerationRequest(
        model=request.model,
        title=request.title,
        prompt=request.prompt,
        negative_prompt=request.negative_prompt,
        size=request.size,
        fps=request.fps,
        steps=request.steps,
        guidance=request.guidance,
        shift=request.shift,
        seed=request.seed,
        adapters=request.adapters,
        metadata=request.metadata,
        long_video=spec,
    )
    result = dispatch_video_create(
        **video_dispatch,
        request=gen_req,
        exec_ctx=ctx_exec,
        on_progress=on_progress,
        on_log=on_log,
    )
    if result is None:
        raise RuntimeError("long video latent_extend cancelled")
    output_path, metadata = result
    metadata = dict(metadata or {})
    metadata["strategy"] = "latent_extend"
    return str(output_path), metadata
