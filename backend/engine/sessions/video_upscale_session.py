"""VideoUpscaleSession — video SR orchestration."""

from __future__ import annotations

from typing import Any, Callable

from backend.core.contracts import ExecutionContext, VideoUpscaleRequest, parse_model_version
from backend.engine.pipelines.video_upscale_pipeline import VideoUpscalePipeline
from backend.engine.sessions.media_session import MediaSession
from backend.engine.sessions.phased_create import run_video_upscale_phased
from backend.engine.sessions.session_routing import family_has_registered_plugin
from backend.engine.video_upscale_registry import resolve_video_upscale_kind


def routes_to_video_upscale_session(model_field: str, registry: Any) -> bool:
    if not family_has_registered_plugin(model_field, registry, expected_media="video"):
        return False
    model_key, version_key = parse_model_version(model_field)
    entry = registry.get(model_key)
    if entry is None:
        return False
    try:
        resolve_video_upscale_kind(entry, version_key or None)
    except RuntimeError:
        return False
    return True


class VideoUpscaleSession(MediaSession):
    """v3 video upscale — resolve + phased job runner (no DiT backbone preload)."""

    media_label = "video upscale"
    load_plugin = False

    def _make_pipeline(self) -> VideoUpscalePipeline:
        return VideoUpscalePipeline(
            self._runtime_ctx,
            self._registry,
            self._asset_store,
            model_cache=self._cache,
            project_root=self._project_root,
        )

    def run(
        self,
        request: VideoUpscaleRequest,
        exec_ctx: ExecutionContext,
        *,
        on_progress: Callable | None = None,
        on_log: Callable | None = None,
    ) -> Any:
        resolved, pipeline, log_cb = self._prepare(request, exec_ctx, on_log)
        return run_video_upscale_phased(
            pipeline,
            resolved,
            request,
            exec_ctx,
            on_progress=on_progress,
            on_log=log_cb,
        )
