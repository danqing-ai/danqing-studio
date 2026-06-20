"""Unit tests for Quick Setup recommendations."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from backend.core.model_registry import ModelRegistry
from backend.services.setup_recommendations import (
    build_setup_recommendations,
    topological_install_order,
)
from backend.utils.size_parse import parse_human_size_to_gb


class SizeParseTests(unittest.TestCase):
    def test_parse_gb(self):
        self.assertAlmostEqual(parse_human_size_to_gb("4GB"), 4.0)
        self.assertAlmostEqual(parse_human_size_to_gb("~2.5GB"), 2.5)

    def test_parse_mb(self):
        self.assertAlmostEqual(parse_human_size_to_gb("512MB"), 512 / 1024)

    def test_parse_invalid(self):
        self.assertIsNone(parse_human_size_to_gb(""))
        self.assertIsNone(parse_human_size_to_gb("n/a"))


class SetupRecommendationsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        registry_path = Path(__file__).resolve().parents[1] / "default_config" / "models_registry.json"
        if not registry_path.is_file():
            raise unittest.SkipTest("models_registry.json missing")
        cls.registry = ModelRegistry.load(registry_path)

    @patch("backend.services.setup_recommendations.PlatformInfo.detect", return_value=["mlx"])
    def test_low_memory_recommends_quantized_image(self, _detect):
        rec = build_setup_recommendations(
            self.registry,
            memory_gb=16.0,
            mlx_memory_limit=16,
        )
        image = next(s for s in rec.slots if s.slot == "image")
        self.assertIsNotNone(image.model_id)
        self.assertIn(image.version_key or "", ("int4", "mlx-4bit", "mlx-8bit", "int8"))

    @patch("backend.services.setup_recommendations.PlatformInfo.detect", return_value=["mlx"])
    def test_high_memory_recommends_image_model(self, _detect):
        rec = build_setup_recommendations(
            self.registry,
            memory_gb=48.0,
            mlx_memory_limit=120,
        )
        image = next(s for s in rec.slots if s.slot == "image")
        self.assertIn(image.model_id, ("z-image-turbo", "flux2-klein-4b"))

    @patch("backend.services.setup_recommendations.PlatformInfo.detect", return_value=["cuda"])
    def test_llm_unavailable_without_mlx(self, _detect):
        rec = build_setup_recommendations(
            self.registry,
            memory_gb=32.0,
            mlx_memory_limit=120,
        )
        llm = next(s for s in rec.slots if s.slot == "llm")
        vlm = next(s for s in rec.slots if s.slot == "vlm")
        self.assertEqual(llm.status, "unavailable")
        self.assertEqual(vlm.status, "unavailable")


class TopologicalInstallTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        registry_path = Path(__file__).resolve().parents[1] / "default_config" / "models_registry.json"
        if not registry_path.is_file():
            raise unittest.SkipTest("models_registry.json missing")
        cls.registry = ModelRegistry.load(registry_path)

    def test_dependency_order(self):
        items = [
            {"model_id": "flux-canny-controlnet", "version_key": "fp16"},
            {"model_id": "flux1-dev", "version_key": "fp16"},
        ]
        ordered = topological_install_order(items, self.registry)
        ids = [row["model_id"] for row in ordered]
        self.assertLess(ids.index("flux1-dev"), ids.index("flux-canny-controlnet"))


if __name__ == "__main__":
    unittest.main()
