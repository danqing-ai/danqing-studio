"""AudioSession — audio create/edit orchestration."""

from __future__ import annotations

from typing import Any, Callable

from backend.core.contracts import AudioEditRequest, AudioGenerationRequest, ExecutionContext
from backend.engine.pipelines.audio_pipeline import AudioPipeline
from backend.engine.sessions.media_session import MediaSession
from backend.engine.sessions.phased_create import (
    run_audio_create_phased,
    run_audio_edit_phased,
)
from backend.engine.sessions.session_routing import routes_with_plugin_and_action


def routes_to_audio_session(model_field: str, registry: Any) -> bool:
    return routes_with_plugin_and_action(
        model_field,
        registry,
        expected_media="audio",
        api_action="create_music",
    )


def routes_to_audio_edit_session(model_field: str, registry: Any) -> bool:
    return routes_with_plugin_and_action(
        model_field,
        registry,
        expected_media="audio",
        api_action="edit",
    )


class AudioSession(MediaSession):
    """v3 audio create/edit — resolve plugin, then phased create/edit helpers."""

    media_label = "audio"

    def _make_pipeline(self) -> AudioPipeline:
        return AudioPipeline(
            ctx=self._runtime_ctx,
            model_registry=self._registry,
            asset_store=self._asset_store,
            model_cache=self._cache,
            project_root=self._project_root,
        )

    def run(
        self,
        request: AudioGenerationRequest,
        exec_ctx: ExecutionContext,
        *,
        on_log: Callable | None = None,
    ) -> Any:
        resolved, pipeline, _log_cb = self._prepare(request, exec_ctx, on_log, log_tag="create")
        return run_audio_create_phased(
            pipeline,
            resolved,
            request,
            exec_ctx,
        )

    def run_edit(
        self,
        request: AudioEditRequest,
        exec_ctx: ExecutionContext,
        *,
        on_log: Callable | None = None,
    ) -> Any:
        resolved, pipeline, _log_cb = self._prepare(request, exec_ctx, on_log, log_tag="edit")
        return run_audio_edit_phased(
            pipeline,
            resolved,
            request,
            exec_ctx,
        )
