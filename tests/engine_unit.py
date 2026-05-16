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


class BenchmarkMetadataTests(unittest.TestCase):
    def test_exit_exempt_nonempty(self) -> None:
        self.assertIn("z-image-create", BENCHMARK_EXIT_EXEMPT_MISMATCH_VS_MFLUX)
        self.assertIn("qwen-image-rewrite", BENCHMARK_EXIT_EXEMPT_MISMATCH_VS_MFLUX)


if __name__ == "__main__":
    unittest.main()
