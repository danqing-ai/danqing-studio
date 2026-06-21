"""Long-video request capability validation (API, scheduler, engine)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from backend.core.contracts import VideoLongGenerationRequest


@dataclass(frozen=True)
class LongVideoValidationError(Exception):
    code: str
    message: str
    http_status: int = 400

    def __str__(self) -> str:
        return self.message


class _CapabilityEngine(Protocol):
    def supports(self, model_id: str, action: str) -> bool: ...


def validate_long_video_request(
    request: VideoLongGenerationRequest,
    *,
    video_engine: _CapabilityEngine,
    image_engine: _CapabilityEngine | None = None,
) -> str:
    """Validate model capabilities for a long-video task. Returns keyframe_model id (may be empty)."""
    spec = request.long_video
    segment_model = (request.model or spec.segment_video_model or "").strip()
    if not segment_model:
        raise LongVideoValidationError("invalid", "model is required", 400)

    phase = (request.metadata or {}).get("long_video_phase") or ""
    if spec.strategy == "segmented_i2v" and phase == "assemble_only":
        if not spec.shots:
            raise LongVideoValidationError("invalid", "long_video.shots required for assemble", 400)
        return (spec.keyframe_model or "").strip()

    if spec.strategy == "segmented_i2v":
        kf_model = (spec.keyframe_model or "").strip()
        if not kf_model:
            raise LongVideoValidationError("invalid", "keyframe_model required", 400)
        if image_engine is None:
            raise LongVideoValidationError(
                "unsupported", "image engine not available for long video", 503
            )
        if not image_engine.supports(kf_model, "generate"):
            raise LongVideoValidationError(
                "unsupported", "keyframe model does not support create", 409
            )
        if not video_engine.supports(segment_model, "edit"):
            raise LongVideoValidationError(
                "unsupported", "segment model does not support animate", 409
            )
        return kf_model

    if spec.strategy == "latent_extend":
        if not video_engine.supports(segment_model, "generate"):
            raise LongVideoValidationError(
                "unsupported", "model does not support latent_extend create", 409
            )
        return (spec.keyframe_model or "").strip()

    raise LongVideoValidationError(
        "invalid", f"unsupported long_video strategy {spec.strategy!r}", 400
    )


def resolve_long_video_image_engine(request: VideoLongGenerationRequest, engines: Any) -> Any:
    """Resolve image engine for VIDEO_LONG_GENERATION after capability validation."""
    spec = request.long_video
    segment_model = (request.model or spec.segment_video_model or "").strip()
    video_eng = engines.get_video(segment_model)
    phase = (request.metadata or {}).get("long_video_phase") or ""

    if spec.strategy == "segmented_i2v" and phase == "assemble_only":
        validate_long_video_request(request, video_engine=video_eng, image_engine=None)
        kf_model = (spec.keyframe_model or "").strip()
        if kf_model:
            return engines.get_image(kf_model)
        return _fallback_image_engine(engines)

    if spec.strategy == "segmented_i2v":
        kf_model = (spec.keyframe_model or "").strip()
        image_eng = engines.get_image(kf_model)
        validate_long_video_request(
            request, video_engine=video_eng, image_engine=image_eng
        )
        return image_eng

    validate_long_video_request(request, video_engine=video_eng, image_engine=None)
    kf_model = (spec.keyframe_model or "").strip()
    if kf_model:
        return engines.get_image(kf_model)

    return _fallback_image_engine(engines)


def _fallback_image_engine(engines: Any) -> Any:
    from backend.engine.danqing_image_engine import DanQingImageEngine

    image_eng = next(
        (e for e in engines._by_engine_id.values() if isinstance(e, DanQingImageEngine)),
        None,
    )
    if image_eng is None:
        raise LongVideoValidationError(
            "unsupported", "image engine not available for long video", 503
        )
    return image_eng
