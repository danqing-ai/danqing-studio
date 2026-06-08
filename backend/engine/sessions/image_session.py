"""ImageSession — image create/edit orchestration."""

from __future__ import annotations

from typing import Any, Callable

from backend.core.contracts import ExecutionContext, ImageEditRequest, ImageGenerationRequest
from backend.engine.pipelines.image_pipeline import ImagePipeline
from backend.engine.sessions.media_session import MediaSession
from backend.engine.sessions.phased_create import run_image_create_phased, run_image_edit_phased
from backend.engine.sessions.session_routing import routes_with_plugin_and_action


def routes_to_image_session(model_field: str, registry: Any) -> bool:
    return routes_with_plugin_and_action(
        model_field,
        registry,
        expected_media="image",
        api_action="generate",
    )


def routes_to_image_edit_session(model_field: str, registry: Any) -> bool:
    return routes_with_plugin_and_action(
        model_field,
        registry,
        expected_media="image",
        api_action="edit",
    )


class ImageSession(MediaSession):
    """Image create/edit — phased create for registered families."""

    media_label = "image"

    def _make_pipeline(self) -> ImagePipeline:
        return ImagePipeline(
            self._runtime_ctx,
            self._registry,
            self._asset_store,
            model_cache=self._cache,
            project_root=self._project_root,
        )

    def run(
        self,
        request: ImageGenerationRequest,
        exec_ctx: ExecutionContext,
        *,
        on_progress: Callable | None = None,
        on_log: Callable | None = None,
    ) -> Any:
        resolved, pipeline, log_cb = self._prepare(request, exec_ctx, on_log, log_tag="create")
        return run_image_create_phased(
            pipeline,
            resolved,
            request,
            exec_ctx,
            on_progress=on_progress,
            on_log=log_cb,
        )

    def run_edit(
        self,
        request: ImageEditRequest,
        exec_ctx: ExecutionContext,
        *,
        on_progress: Callable | None = None,
        on_log: Callable | None = None,
    ) -> Any:
        resolved, pipeline, log_cb = self._prepare(request, exec_ctx, on_log, log_tag="edit")
        return run_image_edit_phased(
            pipeline,
            resolved,
            request,
            exec_ctx,
            on_progress=on_progress,
            on_log=log_cb,
        )
