"""E2E integration tests for long-video script/chapter analyze (local LLM, no HTTP server)."""
from __future__ import annotations

import asyncio
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from backend.core.contracts import LongVideoChapterAnalyzeRequest
from backend.engine.llm.chapter_analyze import MIN_SCENES

# Shootable prose with named characters and multiple visual beats (~900 CJK chars).
SAMPLE_CHAPTER_NARRATIVE = """
第一章 雨夜访客

林晓住在城郊一栋旧公寓的三楼。这天深夜，窗外下着连绵的秋雨，霓虹灯在积水的路面上拉出长长的倒影。

林晓坐在狭小的工作台前，面前摊着一封泛黄的信。信上没有署名，只写着一行字：「今晚子时，老仓库见。」她揉了揉眉心，决定穿上那件深蓝色的连帽外套出门。

老仓库位于废弃的港区，铁门锈迹斑斑。林晓推开吱呀作响的门，昏黄的应急灯忽明忽暗。角落里站着的老周——她的前导师——正用拐杖轻轻敲击地面。

「你终于来了。」老周沉声说，「他们要在黎明前运走那批证据。」

林晓握紧背包带，跟随老周走向仓库深处。铁皮货架之间，成箱的档案在冷白灯下泛着灰光。忽然，门外传来脚步声。两人迅速躲进阴影，林晓屏住呼吸，从货架缝隙看见两名黑衣护卫手持电筒扫过通道。

对峙持续至拂晓前的最暗时刻。林晓与老周交换眼神，她缓步走出，正面迎向领头的护卫，而老周则从侧翼掀开了隐藏在后墙的通风口——那里通向港区的旧栈道。

当第一缕晨光刺破海平面时，林晓已在栈道上奔跑，怀中紧紧抱着从箱中取出的硬盘。雨停了，远处城市的轮廓在晨曦中渐渐清晰。
""".strip()


def _load_llm_service():
    from backend.core.interfaces import AppSettings
    from backend.core.model_registry import ModelRegistry
    from backend.engine.llm.service import (
        LLMService,
        normalize_app_llm_settings,
        resolve_llm_model_id,
        resolve_vlm_model_id,
    )
    from backend.utils.path_utils import PathResolver

    root = Path(__file__).resolve().parents[1]
    pr = PathResolver(root)
    mr = ModelRegistry.load(pr.get_models_registry_path())
    settings = AppSettings()
    normalize_app_llm_settings(settings, mr)
    svc = LLMService(
        mr,
        pr,
        default_model_id=resolve_llm_model_id(settings, mr),
        vision_model_id=resolve_vlm_model_id(settings, mr),
        llm_think_enabled=settings.default_model_llm_think,
    )
    return svc


def _sample_request(**overrides: object) -> LongVideoChapterAnalyzeRequest:
    base = {
        "chapter_text": SAMPLE_CHAPTER_NARRATIVE,
        "chapter_title": "雨夜访客",
        "locale": "zh",
    }
    base.update(overrides)
    return LongVideoChapterAnalyzeRequest(**base)


class LongVideoChapterAnalyzeIntegrationTests(unittest.TestCase):
    def test_rejects_empty_script_without_llm(self) -> None:
        svc = _load_llm_service()
        with self.assertRaises(RuntimeError):
            svc.analyze_long_video_chapter(
                LongVideoChapterAnalyzeRequest(chapter_text="   ", locale="zh"),
            )

    def test_analyze_long_video_chapter_service_e2e(self) -> None:
        """Direct LLMService path — same logic as POST /api/chat/long-video-chapter-analyze."""
        svc = _load_llm_service()
        if not svc.is_available():
            self.skipTest("no local LLM installed in workspace (models/ + registry default)")

        info = svc.get_model_info()
        self.assertEqual(info["model_id"], "qwen3.5-4b", "expected settings default LLM model")

        resp = svc.analyze_long_video_chapter(_sample_request())

        self.assertEqual(resp.chapter_title, "雨夜访客")
        self.assertGreaterEqual(len(resp.synopsis.strip()), 12)
        self.assertGreaterEqual(len(resp.character_anchor.strip()), 8)
        self.assertGreaterEqual(resp.scene_count, MIN_SCENES)
        self.assertGreaterEqual(len(resp.scene_beats), MIN_SCENES)
        self.assertGreaterEqual(resp.llm_calls, 2)

        orders = [s.order for s in resp.scene_beats]
        self.assertEqual(orders, list(range(1, len(resp.scene_beats) + 1)))
        for scene in resp.scene_beats:
            self.assertTrue((scene.beat or scene.title or "").strip())

        anchor_blob = resp.character_anchor
        synopsis_blob = resp.synopsis
        self.assertNotIn("【角色·", synopsis_blob)
        self.assertTrue(
            "林晓" in anchor_blob or "林晓" in synopsis_blob,
            "expected protagonist name in synopsis or character anchor",
        )

        if resp.shots:
            first = resp.shots[0]
            anchor = resp.character_anchor or ""
            primary = "林晓"
            on_screen = getattr(first, "characters_on_screen", None) or []
            start_vis = getattr(first, "first_frame_visibility", None) or ""
            start_prompt = (getattr(first, "start_visual_prompt", None) or "").strip()
            needs_protagonist = primary in on_screen or primary in start_prompt
            if needs_protagonist and primary in anchor:
                self.assertNotEqual(
                    start_vis,
                    "invisible",
                    "opening shot must not be invisible when protagonist is on screen",
                )
            for shot in resp.shots:
                dur = float(getattr(shot, "duration_sec", 0) or 0)
                self.assertGreaterEqual(dur, 2.0)
                self.assertLessEqual(dur, 10.0)
            for shot in resp.shots:
                self.assertNotEqual(getattr(shot, "flf_mode", None), "first_last")

            from backend.engine.common.long_video.parse_quality import validate_parse_quality

            shot_rows = [s.model_dump() for s in resp.shots]
            beat_lines = [s.beat or s.title or "" for s in resp.scene_beats]
            quality = validate_parse_quality(
                shot_rows,
                beat_sheet=beat_lines,
                character_anchor=resp.character_anchor,
                character_dtos=[c.model_dump() for c in resp.characters],
                style_anchor=resp.style_anchor,
            )
            self.assertFalse(
                quality.critical_issues,
                msg="; ".join(i.message for i in quality.critical_issues[:3]),
            )

    def test_long_video_chapter_analyze_route_e2e(self) -> None:
        """Route handler in-process (no uvicorn) with injected LLMService."""
        svc = _load_llm_service()
        if not svc.is_available():
            self.skipTest("no local LLM installed in workspace (models/ + registry default)")

        from backend.api.routes.llm import long_video_chapter_analyze

        http_request = MagicMock()
        http_request.headers.get.return_value = "zh-CN"

        resp = asyncio.run(
            long_video_chapter_analyze(_sample_request(), http_request, service=svc),
        )

        self.assertGreaterEqual(resp.scene_count, MIN_SCENES)
        self.assertGreaterEqual(resp.llm_calls, 2)
        self.assertGreaterEqual(len(resp.synopsis.strip()), 12)


if __name__ == "__main__":
    unittest.main(verbosity=2)
