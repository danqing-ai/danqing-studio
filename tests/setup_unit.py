"""Unit tests for Quick Setup recommendations."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from backend.core.model_registry import ModelRegistry
from backend.core.version_keys import (
    canonical_version_key,
    resolve_full_bundle_version_key,
    resolve_registry_version_key,
)
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
        self.assertIn(image.version_key or "", ("int4", "mlx-q4", "mlx-q8", "int8"))

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


class VersionKeyTests(unittest.TestCase):
    def test_legacy_prequantized_aliases(self) -> None:
        self.assertEqual(canonical_version_key("mlx-4bit"), "mlx-q4")
        self.assertEqual(canonical_version_key("mlx-3bit"), "mlx-q4")
        self.assertEqual(canonical_version_key("community-8bit"), "mlx-q8")
        self.assertEqual(canonical_version_key("mlx-q6"), "mlx-q4")
        self.assertEqual(canonical_version_key("mlx"), "mlx-q8")

    def test_derived_mlx_int_aliases(self) -> None:
        derived = {"source_type": "derived"}
        self.assertEqual(canonical_version_key("mlx-int4", version_entry=derived), "int4")
        self.assertEqual(canonical_version_key("mlx-int8", version_entry=derived), "int8")

    def test_prequantized_mlx_int_aliases(self) -> None:
        pre = {"source_type": "prequantized", "quantization": {"bits": 4}}
        self.assertEqual(canonical_version_key("mlx-int4", version_entry=pre), "mlx-q4")

    def test_resolve_registry_version_key(self) -> None:
        versions = {
            "mlx-q4": {"source_type": "prequantized", "default": True},
            "int4": {"source_type": "derived"},
        }
        self.assertEqual(resolve_registry_version_key(versions, "mlx-4bit"), "mlx-q4")
        self.assertEqual(resolve_registry_version_key(versions, None), "mlx-q4")

    def test_legacy_vague_version_aliases(self) -> None:
        versions = {
            "fp16": {"source_type": "full", "default": True},
            "int4": {"source_type": "derived", "from_version": "fp16"},
        }
        self.assertEqual(canonical_version_key("original"), "fp16")
        self.assertEqual(canonical_version_key("quant"), "int8")
        self.assertEqual(resolve_registry_version_key(versions, "original"), "fp16")
        self.assertEqual(
            resolve_full_bundle_version_key({"bf16": {"source_type": "full"}}),
            "bf16",
        )

    def test_canonical_local_path_legacy_only(self) -> None:
        from backend.core.version_keys import canonical_local_path

        self.assertEqual(
            canonical_local_path("models/Image/flux2-klein-4b-mlx-community-4bit", "mlx-q4"),
            "models/Image/flux2-klein-4b-mlx-q4",
        )
        self.assertEqual(
            canonical_local_path("models/Image/flux2-klein-4b-int4", "int4"),
            "models/Image/flux2-klein-4b-int4",
        )
        self.assertEqual(
            canonical_local_path("models/Video/longcat-video-mlx-q4", "mlx-q4"),
            "models/Video/longcat-video-mlx-q4",
        )

    def test_is_quantized_registry_version(self) -> None:
        from backend.core.version_keys import is_quantized_registry_version

        self.assertTrue(is_quantized_registry_version("int4", {"source_type": "derived"}))
        self.assertTrue(is_quantized_registry_version("mlx-q4", {"source_type": "prequantized"}))
        self.assertTrue(
            is_quantized_registry_version(
                "mlx-q4",
                {"source_type": "full", "quantization": {"bits": 4}},
            )
        )
        self.assertFalse(is_quantized_registry_version("fp16", {"source_type": "full"}))
        self.assertFalse(is_quantized_registry_version("mlx-bf16", {"source_type": "full"}))


if __name__ == "__main__":
    unittest.main()
