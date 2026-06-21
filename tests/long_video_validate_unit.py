"""Unit tests for long-video request validation."""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from backend.core.contracts import VideoLongGenerationRequest, VideoLongVideoSpec
from backend.engine.common.long_video.validate import (
    LongVideoValidationError,
    validate_long_video_request,
)


def _req(**lv_kw) -> VideoLongGenerationRequest:
    spec = VideoLongVideoSpec(strategy="segmented_i2v", **lv_kw)
    return VideoLongGenerationRequest(
        model="wan-2.2-i2v-14b",
        prompt="test",
        long_video=spec,
    )


class LongVideoValidateTests(unittest.TestCase):
    def test_segmented_i2v_requires_keyframe_model(self) -> None:
        req = _req(keyframe_model="")
        video = MagicMock()
        video.supports.return_value = True
        with self.assertRaises(LongVideoValidationError) as ctx:
            validate_long_video_request(req, video_engine=video, image_engine=MagicMock())
        self.assertEqual(ctx.exception.code, "invalid")

    def test_segmented_i2v_checks_capabilities(self) -> None:
        req = _req(keyframe_model="z-image-turbo")
        video = MagicMock()
        video.supports.side_effect = lambda mid, act: act == "edit"
        image = MagicMock()
        image.supports.return_value = True
        kf = validate_long_video_request(req, video_engine=video, image_engine=image)
        self.assertEqual(kf, "z-image-turbo")

    def test_assemble_only_requires_shots(self) -> None:
        req = VideoLongGenerationRequest(
            model="wan-2.2-i2v-14b",
            prompt="test",
            metadata={"long_video_phase": "assemble_only"},
            long_video=VideoLongVideoSpec(strategy="segmented_i2v", shots=[]),
        )
        video = MagicMock()
        with self.assertRaises(LongVideoValidationError) as ctx:
            validate_long_video_request(req, video_engine=video)
        self.assertEqual(ctx.exception.code, "invalid")

    def test_latent_extend_requires_video_create(self) -> None:
        req = VideoLongGenerationRequest(
            model="ltx-video",
            prompt="test",
            long_video=VideoLongVideoSpec(strategy="latent_extend"),
        )
        video = MagicMock()
        video.supports.return_value = False
        with self.assertRaises(LongVideoValidationError) as ctx:
            validate_long_video_request(req, video_engine=video)
        self.assertEqual(ctx.exception.http_status, 409)


if __name__ == "__main__":
    unittest.main()
