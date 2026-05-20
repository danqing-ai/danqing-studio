"""Backend engine unit tests (no weights, no GPU)."""
from __future__ import annotations

import unittest
from types import SimpleNamespace

from backend.engine.common._base import _mlx_affine_infer_bits_and_group_size
from backend.engine.families.ltx.weights import remap_ltx_weights
from tests.benchmark.cases import BENCHMARK_EXIT_EXEMPT_MISMATCH_VS_MFLUX


def _t(shape: tuple[int, ...]) -> SimpleNamespace:
    return SimpleNamespace(shape=shape)


class MlxAffineQuantInferenceTests(unittest.TestCase):
    def test_infer_8bit_from_dense_shape(self) -> None:
        qw, qs = _t((16, 32)), _t((16, 2))
        bits, gs = _mlx_affine_infer_bits_and_group_size(
            qw, qs, dense_weight_shape=(16, 128), weight_key="layers.0.weight"
        )
        self.assertEqual(bits, 8)
        self.assertEqual(gs, 64)

    def test_infer_4bit_from_dense_shape(self) -> None:
        qw, qs = _t((16, 32)), _t((16, 4))
        bits, gs = _mlx_affine_infer_bits_and_group_size(
            qw, qs, dense_weight_shape=(16, 256), weight_key="layers.0.weight"
        )
        self.assertEqual(bits, 4)
        self.assertEqual(gs, 64)

    def test_ambiguous_without_metadata_raises(self) -> None:
        qw, qs = _t((2, 1)), _t((2, 1))
        with self.assertRaises(RuntimeError):
            _mlx_affine_infer_bits_and_group_size(
                qw, qs, dense_weight_shape=None, weight_key="k", bundle_affine_bits=None
            )


class LtxWeightTests(unittest.TestCase):
    def test_remap_top_level_proj_in_bias(self) -> None:
        out = remap_ltx_weights({"proj_in.bias": object()})
        self.assertIn("patch_embed.proj.bias", out)

    def test_import_ltx_transformer(self) -> None:
        from backend.engine.families.ltx.transformer import LTXTransformer

        self.assertEqual(LTXTransformer.__name__, "LTXTransformer")


class ZImageCudaTests(unittest.TestCase):
    def test_transformer_param_map_on_cuda_context(self) -> None:
        from backend.engine.config.model_configs import ZImageConfig
        from backend.engine.families.z_image.transformer import ZImageTransformer
        from backend.engine.runtime.cuda import CudaContext

        ctx = CudaContext("cpu")
        model = ZImageTransformer(ZImageConfig(), ctx)
        self.assertGreater(len(model._param_map), 100)
        self.assertIn("x_embedder.weight", model._param_map)
        import torch

        w = model._param_map["x_embedder.weight"]
        self.assertIsInstance(w, torch.Tensor)

    def test_combine_cfg_noise_matches_mflux(self) -> None:
        from backend.engine.config.model_configs import ZImageConfig
        from backend.engine.families.z_image.transformer import ZImageTransformer
        from backend.engine.runtime.cuda import CudaContext

        model = ZImageTransformer(ZImageConfig(), CudaContext("cpu"))
        cond, uncond = 2.0, 1.0
        g = 4.0
        out = model.combine_cfg_noise(cond, uncond, g)
        self.assertEqual(out, cond + g * (cond - uncond))


class TaskKindMappingTests(unittest.TestCase):
    def test_registry_action_maps_to_audio_generation(self) -> None:
        from backend.core.task_kinds import AUDIO_GENERATION, task_kind_for_registry_action

        self.assertEqual(task_kind_for_registry_action("audio", "create"), AUDIO_GENERATION)

    def test_registry_action_maps_to_image_generation(self) -> None:
        from backend.core.task_kinds import IMAGE_GENERATION, task_kind_for_registry_action

        self.assertEqual(task_kind_for_registry_action("image", "create"), IMAGE_GENERATION)


class AceStepGenerationTests(unittest.TestCase):
    def test_prepare_music_request_auto_zh_and_turbo_steps(self) -> None:
        import json
        import tempfile
        from pathlib import Path

        from backend.core.contracts import AudioGenerationRequest
        from backend.engine.config.model_configs import AceStepConfig
        from backend.engine.families.ace_step.generation import (
            prepare_music_request,
            resolve_bundle_is_turbo,
        )

        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp)
            dit = bundle / "acestep-v15-turbo"
            dit.mkdir()
            (dit / "config.json").write_text(
                json.dumps({"is_turbo": True}),
                encoding="utf-8",
            )
            (dit / "model.safetensors").write_bytes(b"")

            self.assertTrue(resolve_bundle_is_turbo(bundle))

            req = AudioGenerationRequest(
                model="ace-step-xl-sft",
                prompt="piano",
                lyrics="[verse]\n月光洒在窗前",
                duration=10,
                steps=24,
            )
            prepared = prepare_music_request(req, AceStepConfig(), bundle)
            self.assertEqual(prepared.vocal_language, "zh")
            self.assertEqual(prepared.steps, 8)
            self.assertEqual(prepared.shift, 1.0)
            self.assertTrue(prepared.is_turbo)

    def test_import_public_generation_entry(self) -> None:
        from backend.engine.families.ace_step import generation

        self.assertTrue(callable(generation.create_ace_step_generator))
        self.assertTrue(callable(generation.prepare_music_request))


class BenchmarkMetadataTests(unittest.TestCase):
    def test_exit_exempt_nonempty(self) -> None:
        self.assertIn("z-image-create", BENCHMARK_EXIT_EXEMPT_MISMATCH_VS_MFLUX)
        self.assertIn("qwen-image-rewrite", BENCHMARK_EXIT_EXEMPT_MISMATCH_VS_MFLUX)


if __name__ == "__main__":
    unittest.main()
