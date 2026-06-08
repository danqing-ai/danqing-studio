"""VideoSession — video create/edit orchestration."""

from __future__ import annotations

from typing import Any, Callable

from backend.core.contracts import (
    ExecutionContext,
    VideoEditRequest,
    VideoGenerationRequest,
)
from backend.engine.pipelines.video_pipeline import VideoPipeline
from backend.engine.sessions.media_session import MediaSession
from backend.engine.sessions.phased_create import run_video_create_phased, run_video_edit_phased
from backend.engine.sessions.session_routing import family_has_registered_plugin


def routes_to_video_session(model_field: str, registry: Any) -> bool:
    return family_has_registered_plugin(model_field, registry, expected_media="video")


class VideoSession(MediaSession):
    """v3 video create/edit — resolve plugin, then phased create/edit helpers."""

    media_label = "video"

    def _make_pipeline(self) -> VideoPipeline:
        return VideoPipeline(
            self._runtime_ctx,
            self._registry,
            self._asset_store,
            model_cache=self._cache,
            project_root=self._project_root,
        )

    def run(
        self,
        request: VideoGenerationRequest,
        exec_ctx: ExecutionContext,
        *,
        on_progress: Callable | None = None,
        on_log: Callable | None = None,
    ) -> Any:
        resolved, pipeline, log_cb = self._prepare(request, exec_ctx, on_log, log_tag="create")
        return run_video_create_phased(
            pipeline,
            resolved,
            request,
            exec_ctx,
            on_progress=on_progress,
            on_log=log_cb,
        )

    def run_edit(
        self,
        request: VideoEditRequest,
        exec_ctx: ExecutionContext,
        *,
        on_progress: Callable | None = None,
        on_log: Callable | None = None,
    ) -> Any:
        resolved, pipeline, log_cb = self._prepare(request, exec_ctx, on_log, log_tag="edit")
        return run_video_edit_phased(
            pipeline,
            resolved,
            request,
            exec_ctx,
            on_progress=on_progress,
            on_log=log_cb,
        )
