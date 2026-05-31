"""Backend engine unit tests (no weights, no GPU)."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from backend.engine.common._base import _mlx_affine_infer_bits_and_group_size
from backend.engine.families.ltx.weights import remap_ltx_weights
from tests.benchmark.cases import BENCHMARK_EXIT_EXEMPT_MISMATCH_VS_MFLUX


def _t(shape: tuple[int, ...]) -> SimpleNamespace:
    return SimpleNamespace(shape=shape)


class RuntimeContractTests(unittest.TestCase):
    def test_family_runtime_guidance_semantics(self) -> None:
        from backend.engine.common.runtime_contracts import FamilyRuntimeContract

        flux1 = FamilyRuntimeContract(
            family="flux1",
            config=SimpleNamespace(
                supports_guidance=False,
                structured_prompt=False,
                preserve_guidance_when_disabled=True,
                cfg_negative_eligible=False,
            ),
        )
        self.assertEqual(flux1.resolve_guidance_scalar(3.5), 3.5)
        self.assertFalse(flux1.should_encode_negative_prompt(3.5))

        zimg = FamilyRuntimeContract(
            family="z_image",
            config=SimpleNamespace(supports_guidance=True, structured_prompt=False),
        )
        self.assertTrue(zimg.should_encode_negative_prompt(2.0))
        structured = FamilyRuntimeContract(
            family="fibo",
            config=SimpleNamespace(
                supports_guidance=True,
                structured_prompt=True,
                skip_negative_when_structured_prompt=True,
                cfg_negative_eligible=True,
            ),
        )
        self.assertFalse(structured.should_encode_negative_prompt(2.0))

    def test_family_runtime_zimage_noise_layout(self) -> None:
        import numpy as np

        from backend.engine.common.runtime_contracts import FamilyRuntimeContract

        class _FakeCtx:
            @staticmethod
            def seeded_randn(shape, _seed, dtype=None):
                return np.zeros(shape, dtype=np.float32)

            @staticmethod
            def randn(shape, dtype=None):
                return np.zeros(shape, dtype=np.float32)

            @staticmethod
            def expand_dims(x, axis):
                return np.expand_dims(x, axis=axis)

            @staticmethod
            def squeeze(x, axis):
                return np.squeeze(x, axis=axis)

        contract = FamilyRuntimeContract(
            family="z_image",
            config=SimpleNamespace(
                supports_guidance=True,
                structured_prompt=False,
                z_image_noise_layout=True,
                noise_sample_fp32=True,
            ),
        )
        out = contract.sample_txt2img_noise(
            _FakeCtx(),
            latent_shape=(1, 16, 64, 64),
            seed=42,
            sample_dtype="float32",
            target_dtype="float32",
        )
        self.assertEqual(tuple(out.shape), (1, 16, 64, 64))
        class _DTypeCtx:
            @staticmethod
            def float32():
                return "float32"
        self.assertEqual(contract.noise_sample_dtype(_DTypeCtx(), "bfloat16"), "float32")
        self.assertEqual(contract.noise_sample_dtype(_DTypeCtx(), "mlx.core.bfloat16"), "float32")

    def test_scheduler_resolver_falls_back_to_config_flags(self) -> None:
        from backend.engine.common.runtime_contracts import SchedulerSemanticsResolver

        resolver = SchedulerSemanticsResolver()
        entry = SimpleNamespace(parameters={})
        config = SimpleNamespace(requires_sigma_shift=True)
        sem = resolver.resolve(
            entry=entry,
            config=config,
            request_scheduler=None,
            request_metadata={},
            steps=8,
            width=1024,
            height=1024,
        )
        self.assertEqual(sem.scheduler_name, "flow_match_euler")
        self.assertTrue(sem.requires_sigma_shift)
        self.assertTrue(sem.use_empirical_mu)
        self.assertEqual(sem.set_timesteps_kwargs["num_inference_steps"], 8)
        self.assertEqual(sem.set_timesteps_kwargs["image_seq_len"], (1024 // 16) * (1024 // 16))

    def test_scheduler_resolver_registry_overrides_and_sigma_schedule(self) -> None:
        from backend.engine.common.runtime_contracts import SchedulerSemanticsResolver

        resolver = SchedulerSemanticsResolver()
        entry = SimpleNamespace(
            parameters={
                "scheduler": {"default": "flow_match_euler_flux_dynamic"},
                "requires_sigma_shift": {"default": False},
                "use_empirical_mu": {"default": True},
                "scheduler_sigma_schedule": {"default": "linspace_1_to_inv_steps"},
                "scheduler_mu": {"default": 0.42},
                "enable_cfg_renorm": {"default": True},
                "cfg_renorm_min": {"default": 0.12},
            }
        )
        sem = resolver.resolve(
            entry=entry,
            config=SimpleNamespace(requires_sigma_shift=True),
            request_scheduler=None,
            request_metadata={},
            steps=4,
            width=512,
            height=512,
            init_timestep=2,
        )
        self.assertEqual(sem.scheduler_name, "flow_match_euler_flux_dynamic")
        self.assertFalse(sem.requires_sigma_shift)
        self.assertTrue(sem.use_empirical_mu)
        self.assertTrue(sem.cfg_renorm)
        self.assertEqual(sem.cfg_renorm_min, 0.12)
        self.assertEqual(sem.set_timesteps_kwargs["init_timestep"], 2)
        self.assertEqual(sem.set_timesteps_kwargs["mu"], 0.42)
        sigmas = sem.set_timesteps_kwargs["sigmas"]
        self.assertEqual(len(sigmas), 4)
        self.assertAlmostEqual(sigmas[0], 1.0)
        self.assertAlmostEqual(sigmas[-1], 0.25)


class VideoRuntimeContractTests(unittest.TestCase):
    def test_video_contract_config_flags(self) -> None:
        from backend.engine.common.video_runtime_contracts import video_encoder_type
        from backend.engine.config.model_configs import CogVideoXConfig, WanConfig
        from backend.engine.video_codec_registry import get_video_decode_handler

        wan = WanConfig()
        self.assertEqual(str(getattr(wan, "video_vae_backend", "")), "wan")
        self.assertTrue(bool(getattr(wan, "uses_wan_shift", False)))
        self.assertTrue(bool(getattr(wan, "release_t5_after_encode", False)))
        self.assertIsNotNone(get_video_decode_handler("wan"))

        cog = CogVideoXConfig()
        self.assertEqual(str(getattr(cog, "video_vae_backend", "")), "cogvideox")
        self.assertTrue(bool(getattr(cog, "uses_prediction_type", False)))
        self.assertTrue(bool(getattr(cog, "release_t5_after_encode", False)))
        self.assertEqual(video_encoder_type(cog), "t5")


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
    def test_transformer_dispatch_mlx(self) -> None:
        from backend.engine.config.model_configs import ZImageConfig
        from backend.engine.families.z_image.transformer import ZImageTransformer
        from backend.engine.families.z_image.transformer_mlx import ZImageTransformer as ZImageMLX
        from backend.engine.runtime.mlx import MLXContext

        model = ZImageTransformer(ZImageConfig(), MLXContext())
        self.assertIsInstance(model._inner, ZImageMLX)

    def test_transformer_call_delegates_to_inner(self) -> None:
        from backend.engine.config.model_configs import ZImageConfig
        from backend.engine.families.z_image.transformer import ZImageTransformer
        from backend.engine.runtime.mlx import MLXContext

        model = ZImageTransformer(ZImageConfig(), MLXContext())
        self.assertIs(model.__class__.forward, ZImageTransformer.forward)

    def test_transformer_dispatch_cuda(self) -> None:
        from backend.engine.config.model_configs import ZImageConfig
        from backend.engine.families.z_image.transformer import ZImageTransformer
        from backend.engine.families.z_image.transformer_mlx import ZImageTransformer as ZImageMLX
        from backend.engine.runtime.cuda import CudaContext

        model = ZImageTransformer(ZImageConfig(), CudaContext("cpu"))
        self.assertIs(model._inner.__class__, ZImageMLX)

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


class QwenImageTransformerTests(unittest.TestCase):
    def test_transformer_dispatch_mlx(self) -> None:
        from backend.engine.config.model_configs import QwenImageConfig
        from backend.engine.families.qwen.transformer import QwenImageTransformer
        from backend.engine.families.qwen.transformer_mlx import QwenImageTransformer as QwenMLX
        from backend.engine.runtime.mlx import MLXContext

        model = QwenImageTransformer(QwenImageConfig(), MLXContext())
        self.assertIsInstance(model._inner, QwenMLX)

    def test_transformer_dispatch_cuda(self) -> None:
        from backend.engine.config.model_configs import QwenImageConfig
        from backend.engine.families.qwen.transformer import QwenImageTransformer
        from backend.engine.families.qwen.transformer_cuda import QwenImageTransformerCuda
        from backend.engine.runtime.cuda import CudaContext

        model = QwenImageTransformer(QwenImageConfig(), CudaContext("cpu"))
        self.assertIsInstance(model._inner, QwenImageTransformerCuda)

    def test_transformer_unsupported_backend(self) -> None:
        from backend.engine.config.model_configs import QwenImageConfig
        from backend.engine.families.qwen.transformer import QwenImageTransformer

        ctx = SimpleNamespace(backend="unknown")
        with self.assertRaises(RuntimeError):
            QwenImageTransformer(QwenImageConfig(), ctx)

    def test_qwen_image_registry_declares_cuda_backend(self) -> None:
        import json
        from pathlib import Path

        reg = json.loads(
            (Path(__file__).resolve().parents[1] / "default_config/models_registry.json").read_text(
                encoding="utf-8"
            )
        )
        entry = reg["models"]["qwen-image"]
        self.assertIn("cuda", entry.get("backends", []))
        self.assertIn("mlx", entry.get("backends", []))


class TransformerStemDispatchTests(unittest.TestCase):
    _FAMILY_PKG = {"qwen_image": "qwen"}

    def test_registry_transformer_stems_have_backend_dispatch(self) -> None:
        from pathlib import Path

        from backend.engine import _transformer_registry as reg

        root = Path(__file__).resolve().parents[1]
        families = list(reg._TRANSFORMER.keys()) + list(reg._VIDEO_TRANSFORMER.keys())
        for family in families:
            pkg = self._FAMILY_PKG.get(family, family)
            stem = root / "backend" / "engine" / "families" / pkg / "transformer.py"
            self.assertTrue(stem.is_file(), f"missing stem: {stem}")
            text = stem.read_text(encoding="utf-8")
            self.assertIn("backend", text, family)
            has_dispatch = (
                "_inner" in text
                or 'backend == "cuda"' in text
                or "DelegatingDiTStem" in text
                or "dispatch_dit_implementation" in text
            )
            self.assertTrue(has_dispatch, f"{family} transformer.py lacks backend dispatch")

    def test_get_transformer_class_resolves_stems(self) -> None:
        from backend.engine._transformer_registry import get_transformer_class, get_video_transformer_class

        self.assertTrue(get_transformer_class("flux1"))
        self.assertTrue(get_video_transformer_class("ltx"))


class DiTBackendDispatchTests(unittest.TestCase):
    def test_flux1_dispatch_mlx(self) -> None:
        from backend.engine.config.model_configs import Flux1Config
        from backend.engine.families.flux1.transformer import Flux1Transformer
        from backend.engine.families.flux1.transformer_mlx import Flux1Transformer as Flux1MLX
        from backend.engine.runtime.mlx import MLXContext

        model = Flux1Transformer(Flux1Config(), MLXContext())
        self.assertIsInstance(model._inner, Flux1MLX)

    def test_flux1_dispatch_cuda_fail_loud(self) -> None:
        from backend.engine.config.model_configs import Flux1Config
        from backend.engine.families.flux1.transformer import Flux1Transformer
        from backend.engine.runtime.cuda import CudaContext

        with self.assertRaises(RuntimeError):
            Flux1Transformer(Flux1Config(), CudaContext("cpu"))

    def test_ltx_dispatch_mlx(self) -> None:
        from backend.engine.config.model_configs import LTXConfig
        from backend.engine.families.ltx.transformer import LTXTransformer
        from backend.engine.families.ltx.transformer_mlx import LTXTransformer as LTXMLX
        from backend.engine.runtime.mlx import MLXContext

        model = LTXTransformer(LTXConfig(), MLXContext())
        self.assertIsInstance(model._inner, LTXMLX)

    def test_ltx_dispatch_cuda_fail_loud(self) -> None:
        from backend.engine.config.model_configs import LTXConfig
        from backend.engine.families.ltx.transformer import LTXTransformer
        from backend.engine.runtime.cuda import CudaContext

        with self.assertRaises(RuntimeError):
            LTXTransformer(LTXConfig(), CudaContext("cpu"))

    def test_wan_dispatch_cuda_fail_loud(self) -> None:
        from backend.engine.config.model_configs import WanConfig
        from backend.engine.families.wan.transformer import WanTransformer
        from backend.engine.runtime.cuda import CudaContext

        with self.assertRaises(RuntimeError):
            WanTransformer(WanConfig(), CudaContext("cpu"))

    def test_flux2_and_fibo_cuda_fail_loud(self) -> None:
        from backend.engine.config.model_configs import FIBOConfig, Flux2Config
        from backend.engine.families.fibo.transformer import FIBOTransformer
        from backend.engine.families.flux2.transformer import Flux2Transformer
        from backend.engine.runtime.cuda import CudaContext

        ctx = CudaContext("cpu")
        with self.assertRaises(RuntimeError):
            Flux2Transformer(Flux2Config(), ctx)
        with self.assertRaises(RuntimeError):
            FIBOTransformer(FIBOConfig(), ctx)


class TextEncoderStemTests(unittest.TestCase):
    def test_flux1_wan_fibo_stems_reexport_impl(self) -> None:
        from backend.engine.families.fibo.text_encoder import FiboTextEncoder
        from backend.engine.families.fibo.text_encoder_mlx import FiboTextEncoder as FiboImpl
        from backend.engine.families.flux1.flux1_dual import Flux1TextEncoder as Flux1Impl
        from backend.engine.families.flux1.text_encoder import Flux1TextEncoder
        from backend.engine.families.wan.text_encoder import WanUMT5EncoderMLX
        from backend.engine.families.wan.text_encoder_mlx import WanUMT5EncoderMLX as WanImpl

        self.assertIs(Flux1TextEncoder, Flux1Impl)
        self.assertIs(WanUMT5EncoderMLX, WanImpl)
        self.assertIs(FiboTextEncoder, FiboImpl)


class SeedVR2StemTests(unittest.TestCase):
    def test_upscale_and_job_stems(self) -> None:
        from backend.engine.families.seedvr2.stem_mlx import (
            GeneratedImage,
            SeedVR2EulerScheduler,
            SCHEDULER_REGISTRY,
        )
        from backend.engine.families.seedvr2.stem import (
            ModelConfig,
            SeedVR2UpscalePipeline,
            restore_video_chunk_spatiotemporal,
            run_seedvr2_spatiotemporal_video,
        )
        from backend.engine.families.seedvr2.weights import ModelConfig as MC2
        from backend.engine.families.seedvr2 import stem_mlx

        self.assertIs(MC2, ModelConfig)
        self.assertIn("seedvr2_euler", SCHEDULER_REGISTRY)
        self.assertTrue(SeedVR2UpscalePipeline)
        self.assertTrue(GeneratedImage)
        self.assertIs(restore_video_chunk_spatiotemporal, stem_mlx.restore_video_chunk_spatiotemporal)
        self.assertIs(run_seedvr2_spatiotemporal_video, stem_mlx.run_seedvr2_spatiotemporal_video)


class TaskKindMappingTests(unittest.TestCase):
    def test_registry_action_maps_to_audio_generation(self) -> None:
        from backend.core.task_kinds import AUDIO_GENERATION, task_kind_for_registry_action

        self.assertEqual(task_kind_for_registry_action("audio", "create"), AUDIO_GENERATION)

    def test_registry_action_maps_to_image_generation(self) -> None:
        from backend.core.task_kinds import IMAGE_GENERATION, task_kind_for_registry_action

        self.assertEqual(task_kind_for_registry_action("image", "create"), IMAGE_GENERATION)


class TaskSchedulerCancellationTests(unittest.TestCase):
    def test_queued_cancel_removes_from_execution_queue(self) -> None:
        import asyncio
        import tempfile

        import backend.core.task_kinds as TK
        from backend.persistence.v3_task_store import V3TaskStore
        from backend.scheduler.task_scheduler import TaskScheduler

        class _PathResolver:
            def __init__(self, root: Path):
                self._root = root

            def get_outputs_dir(self) -> Path:
                out = self._root / "outputs"
                out.mkdir(parents=True, exist_ok=True)
                return out

        class _EngineRegistry:
            pass

        async def _run() -> None:
            with tempfile.TemporaryDirectory() as td:
                root = Path(td)
                db = root / "studio.db"
                sched = TaskScheduler(
                    path_resolver=_PathResolver(root),
                    task_store=V3TaskStore(db),
                    asset_store=SimpleNamespace(),
                    engine_registry=_EngineRegistry(),
                    config_store=None,
                )
                task = await sched.submit(
                    kind=TK.AUDIO_GENERATION,
                    model_id="ace-step-xl-sft",
                    params={"model": "ace-step-xl-sft", "prompt": "test", "n": 1},
                )
                out = await sched.cancel(task["id"])
                self.assertEqual(out, "ok")
                row = sched.get_task(task["id"])
                self.assertIsNotNone(row)
                self.assertEqual(row["status"], "cancelled")
                queued_ids = {r["id"] for r in sched.queue_snapshot()["queued"]}
                self.assertNotIn(task["id"], queued_ids)

        asyncio.run(_run())

    def test_running_cancel_not_overwritten_by_completed(self) -> None:
        import asyncio
        import tempfile

        import backend.core.task_kinds as TK
        from backend.core.contracts import EngineResult
        from backend.persistence.v3_task_store import V3TaskStore
        from backend.scheduler.task_scheduler import TaskScheduler

        class _PathResolver:
            def __init__(self, root: Path):
                self._root = root

            def get_outputs_dir(self) -> Path:
                out = self._root / "outputs"
                out.mkdir(parents=True, exist_ok=True)
                return out

        class _AudioEngine:
            async def generate(self, _request, ctx):
                # Simulate user cancellation observed during model execution.
                ctx.cancel_token.cancel()
                return EngineResult(
                    primary_asset_id="ast_test",
                    asset_ids=["ast_test"],
                    output_paths=["/tmp/fake.wav"],
                    metadata={},
                )

        class _EngineRegistry:
            def get_audio(self, _model_id):
                return _AudioEngine()

        async def _run() -> None:
            with tempfile.TemporaryDirectory() as td:
                root = Path(td)
                db = root / "studio.db"
                sched = TaskScheduler(
                    path_resolver=_PathResolver(root),
                    task_store=V3TaskStore(db),
                    asset_store=SimpleNamespace(),
                    engine_registry=_EngineRegistry(),
                    config_store=None,
                )
                task = await sched.submit(
                    kind=TK.AUDIO_GENERATION,
                    model_id="ace-step-xl-sft",
                    params={"model": "ace-step-xl-sft", "prompt": "test", "n": 1},
                )
                await sched._execute(task["id"])
                row = sched.get_task(task["id"])
                self.assertIsNotNone(row)
                self.assertEqual(row["status"], "cancelled")

        asyncio.run(_run())

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
            prepared = prepare_music_request(req, AceStepConfig(), bundle, backend="mlx")
            self.assertEqual(prepared.vocal_language, "zh")
            self.assertEqual(prepared.steps, 8)
            self.assertEqual(prepared.shift, 1.0)
            self.assertTrue(prepared.is_turbo)
            self.assertEqual(prepared.duration, 10.0)

    def test_resolve_lm_inspiration_auto_vocal(self) -> None:
        from backend.core.contracts import AudioGenerationRequest
        from backend.engine.families.ace_step.generation import (
            has_substantial_user_lyrics,
            prepare_music_request,
            resolve_lm_inspiration_mode,
        )
        from backend.engine.config.model_configs import AceStepConfig

        req = AudioGenerationRequest(
            model="ace-step-xl-sft",
            prompt="upbeat pop song about summer",
            lyrics="",
            instrumental=False,
        )
        use_inspiration, reason = resolve_lm_inspiration_mode(
            req, raw_lyrics="", lm_enabled=True
        )
        self.assertTrue(use_inspiration)
        self.assertEqual(reason, "auto_vocal_inspiration")
        self.assertFalse(has_substantial_user_lyrics("hi there"))

        with __import__("tempfile").TemporaryDirectory() as tmp:
            import json
            from pathlib import Path

            bundle = Path(tmp)
            dit = bundle / "acestep-v15-turbo"
            dit.mkdir()
            (dit / "config.json").write_text("{}", encoding="utf-8")
            (dit / "model.safetensors").write_bytes(b"")
            prepared = prepare_music_request(req, AceStepConfig(), bundle, backend="mlx")
            self.assertTrue(prepared.simple_mode)
            self.assertEqual(prepared.lyrics, "")

    def test_resolve_lm_inspiration_substantial_lyrics(self) -> None:
        from backend.core.contracts import AudioGenerationRequest
        from backend.engine.families.ace_step.generation import (
            has_substantial_user_lyrics,
            prepare_music_request,
            resolve_lm_inspiration_mode,
        )
        from backend.engine.config.model_configs import AceStepConfig

        lyrics = "[verse]\n" + ("la la la " * 10)
        self.assertTrue(has_substantial_user_lyrics(lyrics))
        req = AudioGenerationRequest(
            model="ace-step-xl-sft",
            prompt="pop",
            lyrics=lyrics,
        )
        use_inspiration, reason = resolve_lm_inspiration_mode(
            req, raw_lyrics=lyrics, lm_enabled=True
        )
        self.assertFalse(use_inspiration)
        self.assertEqual(reason, "auto_substantial_lyrics")

        with __import__("tempfile").TemporaryDirectory() as tmp:
            import json
            from pathlib import Path

            bundle = Path(tmp)
            dit = bundle / "acestep-v15-turbo"
            dit.mkdir()
            (dit / "config.json").write_text("{}", encoding="utf-8")
            (dit / "model.safetensors").write_bytes(b"")
            prepared = prepare_music_request(req, AceStepConfig(), bundle, backend="mlx")
            self.assertFalse(prepared.simple_mode)
            self.assertEqual(prepared.lyrics, lyrics.strip())

    def test_resource_policy_clamps_duration(self) -> None:
        from backend.engine.families.ace_step.resource_policy import (
            AceStepResourcePolicy,
            clamp_duration,
        )

        policy = AceStepResourcePolicy(
            memory_gb=8.0,
            tier="low",
            max_duration_with_lm=120,
            max_duration_without_lm=180,
            available_lm_models=("acestep-5Hz-lm-0.6B",),
            lm_quantize_bits=8,
        )
        dur, msg = clamp_duration(300.0, lm_enabled=True, policy=policy)
        self.assertEqual(dur, 120.0)
        self.assertIsNotNone(msg)

    def test_quality_assessment_flags_hum(self) -> None:
        from backend.engine.families.ace_step.quality_score import assess_generation_quality

        q = assess_generation_quality(hum_ratio=0.4, mains_acf=0.5, latent_cos=0.5, latent_diff=0.2)
        self.assertLess(q.score, 60.0)
        self.assertEqual(q.grade, "poor")

    def test_audio_codec_fsq_roundtrip(self) -> None:
        import mlx.core as mx

        from backend.engine.families.ace_step.audio_codec_mlx import _MlxResidualFSQ

        levels = [8, 8, 8, 5, 5, 5]
        rf = _MlxResidualFSQ(dim=64, levels=levels, num_quantizers=1)
        x = mx.random.normal((1, 4, 64))
        quantized, indices = rf(x)
        restored = rf.get_output_from_indices(indices)
        diff = float(mx.max(mx.abs(quantized - restored)))
        self.assertLess(diff, 1e-5)

    def test_audio_codec_indices_cast_float_to_int(self) -> None:
        import mlx.core as mx

        from backend.engine.families.ace_step.audio_codec_mlx import _MlxResidualFSQ

        levels = [8, 8, 8, 5, 5, 5]
        rf = _MlxResidualFSQ(dim=64, levels=levels, num_quantizers=1)
        float_idx = mx.array([[[3.0]]], dtype=mx.float32)
        out = rf.get_output_from_indices(float_idx)
        mx.eval(out)
        self.assertEqual(out.shape, (1, 1, 64))

    def test_parse_audio_code_indices(self) -> None:
        from backend.engine.families.ace_step.lm_format import parse_audio_code_indices

        text = "<|audio_code_12|><|audio_code_345|>"
        self.assertEqual(parse_audio_code_indices(text), (12, 345))

    def test_instrumental_lyrics_normalization(self) -> None:
        from backend.engine.families.ace_step.generation import finalize_lyrics_for_inference
        from backend.engine.families.ace_step.lm_format import (
            build_inspiration_user_content,
            extract_lm_generated_lyrics,
            is_instrumental_lyrics,
            normalize_lyrics_body,
            synthesize_fallback_vocal_lyrics,
        )

        self.assertTrue(is_instrumental_lyrics("# Lyric [Instrumental]"))
        self.assertEqual(normalize_lyrics_body("# Lyric [Instrumental]"), "[Instrumental]")
        self.assertIn("不要使用", build_inspiration_user_content("pop", instrumental=False, vocal_language="zh"))
        self.assertEqual(
            extract_lm_generated_lyrics("[Verse 1]\nhello", prefilled_think_end=True),
            "[Verse 1]\nhello",
        )
        fb = synthesize_fallback_vocal_lyrics("夏天", vocal_language="zh")
        self.assertIn("夏天", fb)
        self.assertFalse(is_instrumental_lyrics(fb))
        with self.assertRaises(RuntimeError):
            finalize_lyrics_for_inference(
                "# Lyric [Instrumental]",
                instrumental=False,
                lm_expanded=True,
            )

    def test_format_sample_preserves_user_lyrics(self) -> None:
        from backend.engine.families.ace_step.lm_format import build_lm_format_result

        user = "[Verse 1]\n夏天的风轻轻吹\n[Chorus]\n一起唱"
        meta = {
            "caption": "Expanded pop caption",
            "lyrics_tail": "[Instrumental]",
        }
        result = build_lm_format_result(
            meta,
            caption_in="pop",
            lyrics_in=user,
            duration=60,
            bpm=120,
            keyscale="C major",
            timesignature="4",
            language="zh",
            preserve_user_lyrics=True,
        )
        self.assertIn("夏天的风", result.lyrics)
        self.assertNotEqual(result.lyrics.strip().lower(), "[instrumental]")

    def test_format_metadata_as_cot_and_codes_prompt(self) -> None:
        from backend.engine.families.ace_step.lm_format import (
            build_codes_phase_prompt,
            format_metadata_as_cot,
            lm_planner_codes_enabled,
        )

        self.assertTrue(lm_planner_codes_enabled())
        cot = format_metadata_as_cot(
            {
                "bpm": 120,
                "caption": "upbeat pop",
                "duration": 30,
                "keyscale": "C Major",
                "language": "en",
                "timesignature": "4/4",
            }
        )
        self.assertIn("<think>", cot)
        self.assertIn("bpm: 120", cot)
        self.assertIn("timesignature: 4", cot)

        class _Tok:
            def apply_chat_template(self, messages, *, tokenize=False, add_generation_prompt=False):
                del tokenize, add_generation_prompt
                return f"SYS:{messages[0]['content']}|USER:{messages[1]['content']}|ASSIST:"

        prompt = build_codes_phase_prompt(
            _Tok(),
            caption="pop song",
            lyrics="[Verse]\nhello",
            cot_text=cot,
        )
        self.assertIn("pop song", prompt)
        self.assertIn(cot, prompt)
        self.assertTrue(prompt.endswith("\n\n"))

    def test_lm_codes_cfg_defaults(self) -> None:
        from backend.engine.families.ace_step.constrained_generate import (
            ConstrainedGenerationConfig,
            create_constrained_processor,
            resolve_hf_tokenizer,
        )
        from backend.engine.families.ace_step.lm_format import (
            build_codes_uncond_prompt,
            default_lm_codes_cfg_scale,
        )

        self.assertGreaterEqual(default_lm_codes_cfg_scale(), 1.0)
        uncond = build_codes_uncond_prompt(
            type("_T", (), {"apply_chat_template": lambda self, m, **k: "PROMPT"})()
        )
        self.assertIn("<think>", uncond)
        cfg = ConstrainedGenerationConfig(cfg_scale=2.0, uncond_prompt=uncond)
        self.assertEqual(cfg.cfg_scale, 2.0)

    def test_mlx_lm_tokenizer_wrapper_compat(self) -> None:
        import numpy as np
        import mlx.core as mx
        import torch
        from transformers import AutoTokenizer

        from backend.engine.families.ace_step.constrained_generate import (
            create_constrained_processor,
            mlx_logits_to_torch,
            resolve_hf_tokenizer,
        )
        from mlx_lm.tokenizer_utils import TokenizerWrapper

        inner = AutoTokenizer.from_pretrained("gpt2")
        wrapped = TokenizerWrapper(inner)
        self.assertIs(resolve_hf_tokenizer(wrapped), inner)
        processor = create_constrained_processor(wrapped)
        self.assertEqual(processor.vocab_size, inner.vocab_size)
        bf = mx.array([[[1.0, 2.0]]], dtype=mx.bfloat16)
        mx.eval(bf)
        t = mlx_logits_to_torch(bf)
        self.assertEqual(t.dtype, torch.float32)
        self.assertEqual(float(t[0, 0, 0].item()), 1.0)

    def test_lyrics_alignment_structure_estimate(self) -> None:
        from backend.engine.families.ace_step.lyrics_alignment import (
            estimate_lyrics_alignment,
            format_lrc,
        )

        lyrics = "[Verse 1]\nHello world\nSecond line\n[Chorus]\nSing it loud"
        align = estimate_lyrics_alignment(lyrics, duration_sec=60.0)
        self.assertEqual(align.mode, "structure_estimate")
        self.assertGreaterEqual(len(align.segments), 3)
        lrc = format_lrc(align)
        self.assertIsNotNone(lrc)
        self.assertIn("[00:", lrc or "")

    def test_write_lrc_sidecar(self) -> None:
        import tempfile
        from pathlib import Path

        from backend.engine.families.ace_step.generation import write_lrc_sidecar

        with tempfile.TemporaryDirectory() as td:
            audio = Path(td) / "track.wav"
            audio.write_bytes(b"RIFF")
            lrc = write_lrc_sidecar(
                audio,
                "[Verse 1]\nHello\nWorld",
                duration_sec=30.0,
            )
            self.assertIsNotNone(lrc)
            assert lrc is not None
            self.assertTrue(lrc.is_file())
            self.assertIn("[00:", lrc.read_text(encoding="utf-8"))

    def test_import_public_generation_entry(self) -> None:
        from backend.engine.families.ace_step import generation

        self.assertTrue(callable(generation.create_ace_step_generator))
        self.assertTrue(callable(generation.prepare_music_request))


class RegistrySeedTests(unittest.TestCase):
    def test_seed_workspace_config_copies_registry_once(self) -> None:
        import json
        import tempfile
        from pathlib import Path

        from backend.utils.config_paths import seed_workspace_config_from_defaults

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            default = root / "default_config"
            default.mkdir()
            (default / "models_registry.json").write_text(
                json.dumps({"schema_version": 2, "models": {"z-image": {}}}, ensure_ascii=False),
                encoding="utf-8",
            )
            (default / "presets.json").write_text("{}", encoding="utf-8")
            workspace = root / "workspace"
            seed_workspace_config_from_defaults(default, workspace)
            reg = json.loads((workspace / "config" / "models_registry.json").read_text(encoding="utf-8"))
            self.assertIn("z-image", reg["models"])
            # Second seed must not overwrite an edited workspace copy
            reg_path = workspace / "config" / "models_registry.json"
            reg_path.write_text(
                json.dumps({"schema_version": 2, "models": {"user-only": {}}}, ensure_ascii=False),
                encoding="utf-8",
            )
            seed_workspace_config_from_defaults(default, workspace)
            reg2 = json.loads(reg_path.read_text(encoding="utf-8"))
            self.assertIn("user-only", reg2["models"])
            self.assertNotIn("z-image", reg2["models"])


class BundleReposTests(unittest.TestCase):
    def test_bundle_repos_from_version(self) -> None:
        from backend.core.bundle_repos import (
            bundle_local_paths,
            bundle_repos_from_version,
            primary_and_follow_ups,
            version_primary_local_path,
        )

        ver = {
            "bundle_repos": [
                {"repo_id": "org/a", "local_path": "models/a"},
                {"repo_id": "org/b", "local_path": "models/a/b", "source": "huggingface"},
            ]
        }
        entries = bundle_repos_from_version(ver)
        self.assertEqual(len(entries), 2)
        primary, follow = primary_and_follow_ups(ver)
        self.assertEqual(primary["repo_id"], "org/a")
        self.assertEqual(len(follow), 1)
        self.assertEqual(version_primary_local_path(ver), "models/a")
        self.assertEqual(bundle_local_paths(ver), ["models/a", "models/a/b"])

    def test_local_bundle_root_from_bundle_repos(self) -> None:
        from backend.engine.common.pipeline_registry import local_bundle_root

        class _Entry:
            raw = {
                "versions": {
                    "default": {
                        "bundle_repos": [
                            {
                                "repo_id": "HeartMuLa/HeartMuLaGen",
                                "local_path": "models/Audio/heartmula-oss-3b-happy-new-year",
                            },
                        ],
                    },
                },
            }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle = root / "models/Audio/heartmula-oss-3b-happy-new-year"
            bundle.mkdir(parents=True)
            resolved = local_bundle_root(root, _Entry(), "default")
            self.assertEqual(resolved, bundle.resolve())
            missing = local_bundle_root(root, _Entry(), "missing")
            self.assertIsNone(missing)


class HeartMulaGenerationTests(unittest.TestCase):
    def test_prepare_heartmula_request_tags_and_cfg(self) -> None:
        from backend.core.contracts import AudioGenerationRequest
        from backend.engine.config.model_configs import HeartMulaConfig
        from backend.engine.families.heartmula.generation import (
            create_heartmula_generator,
            prepare_heartmula_request,
            prompt_to_tags,
        )

        req = AudioGenerationRequest(
            model="heartmula-oss-3b-happy-new-year",
            prompt="pop, female vocal, acoustic",
            lyrics="[verse]\nHello world",
            duration=45,
            guidance=2.0,
            temperature=1.2,
            top_k=80,
            codec_steps=12,
            codec_guidance=1.4,
            long_form_temperature=1.1,
            long_form_topk=70,
        )
        prepared = prepare_heartmula_request(req, HeartMulaConfig())
        self.assertIn("pop", prepared.tags)
        self.assertEqual(prepared.cfg_scale, 2.0)
        self.assertEqual(prepared.duration, 45.0)
        self.assertEqual(prepared.temperature, 1.2)
        self.assertEqual(prepared.topk, 80)
        self.assertEqual(prepared.codec_steps, 12)
        self.assertEqual(prepared.codec_guidance, 1.4)
        self.assertEqual(prepared.long_form_temperature, 1.1)
        self.assertEqual(prepared.long_form_topk, 70)
        self.assertEqual(prepared.lyrics, "[verse]\nHello world")
        self.assertTrue(callable(create_heartmula_generator))
        self.assertEqual(prompt_to_tags("  jazz  "), "jazz")

    def test_family_config_registered(self) -> None:
        from backend.engine.config.model_configs import HeartMulaConfig, get_config_class

        self.assertIs(get_config_class("heartmula"), HeartMulaConfig)

    def test_audio_engine_supports_model_with_version_suffix(self) -> None:
        import json
        import tempfile
        from pathlib import Path
        from unittest.mock import MagicMock

        from backend.core.model_registry import ModelRegistry
        from backend.engine.danqing_audio_engine import DanQingAudioEngine

        payload = {
            "schema_version": 2,
            "models": {
                "heartmula-oss-3b-happy-new-year": {
                    "media": "audio",
                    "engine": "danqing-audio",
                    "family": "heartmula",
                    "actions": {"create": {}},
                }
            },
        }
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump(payload, f)
            reg_path = Path(f.name)
        reg = ModelRegistry.load(reg_path)
        reg_path.unlink(missing_ok=True)
        engine = DanQingAudioEngine(
            path_resolver=MagicMock(),
            registry=reg,
            runtimes={"mlx": MagicMock()},
        )
        self.assertTrue(engine.supports("heartmula-oss-3b-happy-new-year:default", "create_music"))

    def test_registry_bundle_repos(self) -> None:
        import json
        from pathlib import Path

        from backend.core.bundle_repos import bundle_repos_from_version

        reg = json.loads(
            (Path(__file__).resolve().parents[1] / "default_config" / "models_registry.json").read_text(
                encoding="utf-8",
            )
        )
        ver = reg["models"]["heartmula-oss-3b-happy-new-year"]["versions"]["default"]
        repos = bundle_repos_from_version(ver)
        self.assertEqual(len(repos), 3)
        self.assertEqual(repos[0]["repo_id"], "HeartMuLa/HeartMuLaGen")
        self.assertNotIn("companion_repo_id", ver)
        self.assertNotIn("extra_companions", ver)
        hm = reg["models"]["heartmula-oss-3b-happy-new-year"]
        self.assertFalse(hm["parameters"]["negative_prompt_support"])
        for key in ("temperature", "top_k", "codec_steps", "codec_guidance", "long_form_temperature", "long_form_topk"):
            self.assertIn(key, hm["parameters"])
        hooks = ver.get("install_hooks")
        self.assertEqual(len(hooks), 1)
        self.assertEqual(hooks[0]["type"], "heartmula_mlx_weights")

    def test_mlx_stack_imports(self) -> None:
        from backend.engine.families.heartmula.generation_mlx import HeartMulaMlxGenerator
        from backend.engine.families.heartmula.mula_mlx import HeartMuLa
        from backend.engine.families.heartmula.weights_mlx import (
            convert_heartmula_weights,
            load_pytorch_weights,
        )

        self.assertTrue(callable(load_pytorch_weights))
        self.assertTrue(callable(convert_heartmula_weights))
        self.assertIsNotNone(HeartMuLa)
        self.assertIsNotNone(HeartMulaMlxGenerator)

    def test_codec_normalize_codes_layout(self) -> None:
        import mlx.core as mx

        from backend.engine.families.heartmula.codec_mlx import HeartCodec, HeartCodecConfig

        codec = HeartCodec(HeartCodecConfig())
        heartlib = mx.zeros((1, 8, 125), dtype=mx.int32)
        normalized = codec._normalize_codes_layout(heartlib)
        self.assertEqual(tuple(normalized.shape), (1, 125, 8))
        already = mx.zeros((1, 125, 8), dtype=mx.int32)
        self.assertEqual(tuple(codec._normalize_codes_layout(already).shape), (1, 125, 8))

    def test_codec_chunk_code_frames_and_routing(self) -> None:
        from backend.engine.families.heartmula.codec_mlx import (
            HeartCodec,
            HeartCodecConfig,
            chunk_code_frames,
            single_pass_frame_limit,
        )

        cfg = HeartCodecConfig()
        codec = HeartCodec(cfg)
        chunk_frames = chunk_code_frames(cfg.frame_rate)
        single_frames = single_pass_frame_limit(cfg.frame_rate)
        self.assertGreater(chunk_frames, int(25 * cfg.frame_rate))
        self.assertLess(chunk_frames, int(35 * cfg.frame_rate))
        self.assertGreaterEqual(single_frames, int(120 * cfg.frame_rate))
        # 300s → thousands of code frames → must exceed single-pass threshold
        self.assertGreater(int(300 * cfg.frame_rate), single_frames)


class HeartMulaCodecParityTests(unittest.TestCase):
    def test_compare_audio_waveforms_identical(self) -> None:
        import numpy as np

        from tests.benchmark.metrics import compare_audio_waveforms, si_sdr_db

        sr = 48000
        t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float64)
        sig = 0.3 * np.sin(2 * np.pi * 440.0 * t)
        res = compare_audio_waveforms(sig, sig, sample_rate=sr)
        self.assertTrue(res.product_ok)
        self.assertIsNotNone(res.si_sdr_db)
        assert res.si_sdr_db is not None
        self.assertGreater(res.si_sdr_db, 80.0)
        self.assertAlmostEqual(si_sdr_db(sig, sig), res.si_sdr_db, places=3)
        assert res.correlation is not None
        self.assertGreater(res.correlation, 0.999)

    def test_codec_parity_skips_without_fixtures(self) -> None:
        from tests.benchmark.heartmula_codec_parity import run_codec_parity_check

        res = run_codec_parity_check(
            Path("tests/benchmark/fixtures/heartmula/does_not_exist.json"),
            output_dir=Path("tests/benchmark/outputs/_codec_parity_unit"),
            case_id="test-missing-fixtures",
        )
        self.assertTrue(res.skipped)
        self.assertIn("codec_parity_skip", res.reason)

    def test_load_codes_layout(self) -> None:
        import tempfile

        import numpy as np

        from tests.benchmark.heartmula_codec_parity import load_codes_npy

        codes = np.random.randint(0, 8192, size=(125, 8), dtype=np.int32)
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tmp:
            path = Path(tmp.name)
            np.save(path, codes)
        try:
            loaded = load_codes_npy(path)
            self.assertEqual(loaded.shape, (125, 8))
            batched = codes[None, ...]
            np.save(path, batched)
            loaded_b = load_codes_npy(path)
            self.assertEqual(loaded_b.shape, (125, 8))
        finally:
            path.unlink(missing_ok=True)


class HunyuanWeightTests(unittest.TestCase):
    def test_remap_strips_transformer_prefix(self) -> None:
        from backend.engine.families.hunyuan.weights import remap_hunyuan_weights

        out = remap_hunyuan_weights({"transformer.x_embedder.proj.bias": object()})
        self.assertIn("x_embedder.proj.bias", out)

    def test_import_hunyuan_transformer(self) -> None:
        from backend.engine.families.hunyuan.transformer import HunyuanVideoTransformer

        self.assertEqual(HunyuanVideoTransformer.__name__, "HunyuanVideoTransformer")

    def test_hunyuan_cfg_batch_helpers(self) -> None:
        import mlx.core as mx

        from backend.engine.common.cfg_batch import broadcast_batch, merge_cfg_forward_kwargs

        class _Ctx:
            @staticmethod
            def concat(parts, axis=0):
                return mx.concatenate(parts, axis=axis)

        ctx = _Ctx()
        one = mx.ones((1, 2, 3))
        two = broadcast_batch(ctx, one, 2)
        self.assertEqual(two.shape, (2, 2, 3))

        pos = {
            "txt_embeds": mx.ones((1, 4, 8)),
            "txt_attn_mask": mx.ones((1, 4)),
            "sigmas": mx.array([0.5]),
        }
        neg = {
            "txt_embeds": mx.zeros((1, 4, 8)),
            "txt_attn_mask": mx.zeros((1, 4)),
            "sigmas": mx.array([0.5]),
        }
        merged = merge_cfg_forward_kwargs(
            ctx, pos, neg, text_keys=frozenset({"txt_embeds", "txt_attn_mask"}),
        )
        self.assertEqual(merged["txt_embeds"].shape[0], 2)
        self.assertEqual(merged["txt_attn_mask"].shape[0], 2)
        self.assertEqual(merged["sigmas"].shape, (1,))

    def test_cogvideox_wan_predict_noise_cfg(self) -> None:
        from backend.engine.families.cogvideox.transformer_mlx import CogVideoXTransformer3D
        from backend.engine.families.wan.transformer import WanTransformer
        from backend.engine.families.wan.transformer_mlx import WanModelMLX

        self.assertTrue(hasattr(CogVideoXTransformer3D, "predict_noise_cfg"))
        self.assertTrue(hasattr(WanModelMLX, "predict_noise_cfg"))
        self.assertTrue(hasattr(WanTransformer, "predict_noise_cfg"))

    def test_cogvideox_param_map_flattens_mlx_modules(self) -> None:
        from backend.engine.config.model_configs import CogVideoXConfig
        from backend.engine.families.cogvideox.transformer_mlx import CogVideoXTransformer3D
        from backend.engine.runtime.mlx import MLXContext

        model = CogVideoXTransformer3D(CogVideoXConfig(), MLXContext(), num_frames=13)
        model._build_param_map()
        self.assertGreater(len(model._param_map), 600)
        self.assertIn("time_embedding.linear_1.weight", model._param_map)
        self.assertIn("transformer_blocks.0.attn1.to_out.0.weight", model._param_map)
        for key, param in model._param_map.items():
            self.assertFalse(
                isinstance(param, dict),
                msg=f"param_map leaf {key!r} must be a tensor, not a nested dict",
            )

    def test_wan_mlx_perf_hooks(self) -> None:
        from backend.engine.config.model_configs import WanConfig
        from backend.engine.families.wan.transformer_mlx import WanModelMLX

        cfg = WanConfig()
        self.assertTrue(cfg.use_mlx_compile)
        self.assertFalse(cfg.vae_spatial_tiling)
        self.assertTrue(hasattr(WanModelMLX, "after_load_weights"))
        self.assertTrue(hasattr(WanModelMLX, "invalidate_text_cache"))

    def test_wan_umt5_weights_load(self) -> None:
        from pathlib import Path

        import mlx.core as mx

        from backend.engine.families.wan.text_encoder_mlx import (
            WanUMT5EncoderMLX,
            _apply_umt5_weights,
            _build_umt5_param_map,
            _load_umt5_state_dict,
            resolve_wan_umt5_pth,
        )
        from tests.benchmark.cases import WAN_VIDEO_BUNDLE, wan_video_bundle_installed

        if not wan_video_bundle_installed():
            self.skipTest("Wan bundle not installed")

        bundle = Path(WAN_VIDEO_BUNDLE)
        resolved = resolve_wan_umt5_pth(bundle)
        self.assertIsNotNone(resolved)
        pth, tok = resolved

        sd = _load_umt5_state_dict(pth)
        from backend.engine.families.wan.text_encoder_mlx import _UMT5Encoder

        model = _UMT5Encoder()
        _apply_umt5_weights(model, sd)
        pmap = _build_umt5_param_map(model)
        ref = sd["blocks.0.attn.q.weight"]
        loaded = pmap["blocks.0.attn.q.weight"]
        mx.eval(loaded)
        self.assertLess(float(mx.max(mx.abs(loaded - ref))), 1e-6)

        from backend.engine.runtime.mlx import MLXContext

        enc = WanUMT5EncoderMLX(MLXContext(), pth, tok, text_len=512)
        out = enc.encode(["a red ball on a table"])
        mx.eval(out)
        self.assertGreater(float(mx.max(mx.abs(out[0, :20]))), 0.05)

    def test_wan_flow_unipc_order2_predictor(self) -> None:
        import mlx.core as mx

        from backend.engine.common.schedulers import WanFlowUniPCScheduler
        from backend.engine.runtime.mlx import MLXContext

        ctx = MLXContext()
        sched = WanFlowUniPCScheduler(1000, ctx=ctx, solver_order=2)
        sched.set_timesteps(4, shift=5.0)
        sample = mx.ones((1, 2, 2, 2))
        m0 = mx.full(sample.shape, 0.5)
        m1 = mx.full(sample.shape, 0.25)
        sched._model_outputs = [m1, m0]
        sched._step_index = 1
        order1 = sched._predictor_step(sample, order=1)
        sched._model_outputs = [m1, m0]
        sched._step_index = 1
        order2 = sched._predictor_step(sample, order=2)
        mx.eval(order1, order2)
        self.assertGreater(float(mx.max(mx.abs(order1 - order2))), 1e-6)

    def test_wan_attention_padding_mask(self) -> None:
        import mlx.core as mx

        from backend.engine.common.attention import wan_attention
        from backend.engine.runtime.mlx import MLXContext

        ctx = MLXContext()
        b, l, h, d = 1, 8, 2, 4
        q = mx.random.normal((b, l, h, d))
        k = mx.random.normal((b, l, h, d))
        v = mx.zeros((b, l, h, d))
        v = v.at[:, 4:, :, :].add(100.0)
        seq_lens = mx.array([4], dtype=mx.int32)
        out_masked = wan_attention(ctx, q, k, v, k_lens=seq_lens)
        out_full = wan_attention(ctx, q, k, v)
        mx.eval(out_masked, out_full)
        self.assertGreater(float(mx.max(mx.abs(out_masked[0, :4] - out_full[0, :4]))), 1.0)

    def test_wan_vae_spatial_tiling_threshold(self) -> None:
        from backend.engine.families.wan.vae_mlx import (
            _needs_wan_spatial_tiling,
            _wan_vae_tile_params,
            patchify,
            unpatchify,
        )

        params = _wan_vae_tile_params({}, spatial_scale=16)
        self.assertFalse(
            _needs_wan_spatial_tiling(10, 10, params, enabled=True),
        )
        self.assertTrue(
            _needs_wan_spatial_tiling(80, 80, params, enabled=True),
        )
        self.assertFalse(
            _needs_wan_spatial_tiling(80, 80, params, enabled=False),
        )

        import mlx.core as mx

        x = mx.random.normal((1, 3, 5, 8, 8))
        y = unpatchify(patchify(x, 2), 2)
        mx.eval(y)
        self.assertLess(float(mx.max(mx.abs(x - y))), 1e-5)

        # Must match diffusers AutoencoderKLWan permutations (not an arbitrary inverse).
        b, c, f, h, w = 1, 3, 5, 8, 8
        q = 2
        t = mx.reshape(x, (b, c, f, h, w))
        p = mx.reshape(t, (b, c, f, h // q, q, w // q, q))
        p = mx.reshape(mx.transpose(p, (0, 1, 6, 4, 2, 3, 5)), (b, c * q * q, f, h // q, w // q))
        c0 = c
        u = mx.reshape(p, (b, c0, q, q, f, h // q, w // q))
        u = mx.reshape(mx.transpose(u, (0, 1, 4, 5, 3, 6, 2)), (b, c0, f, h, w))
        mx.eval(u)
        self.assertLess(float(mx.max(mx.abs(patchify(x, 2) - p))), 1e-5)
        self.assertLess(float(mx.max(mx.abs(x - u))), 1e-5)

    def test_extract_glyph_texts(self) -> None:
        from backend.engine.families.hunyuan.text_encoder import extract_glyph_texts

        self.assertEqual(extract_glyph_texts('hello "WORLD" test'), "WORLD")
        self.assertIsNone(extract_glyph_texts("no quoted text"))

    def test_hunyuan_config_defaults(self) -> None:
        from backend.engine.config.model_configs import HunyuanVideoConfig

        cfg = HunyuanVideoConfig()
        self.assertEqual(cfg.dim_in, 32)
        self.assertEqual(cfg.vae_scale, 16)
        self.assertEqual(cfg.encoder_type, "hunyuan_video_dual")

    def test_hunyuan_registry_entries(self) -> None:
        import json
        from pathlib import Path

        reg = json.loads(
            (Path(__file__).resolve().parents[1] / "default_config" / "models_registry.json").read_text(
                encoding="utf-8",
            )
        )
        models = reg.get("models") or {}
        self.assertIn("hunyuan-video-1.5-480p-t2v", models)
        self.assertEqual(models["hunyuan-video-1.5-480p-t2v"]["family"], "hunyuan")
        self.assertIn("hunyuan-video-1.5-i2v-step-distill", models)
        distill = models["hunyuan-video-1.5-i2v-step-distill"]["parameters"]
        self.assertFalse(distill.get("supports_guidance"))
        self.assertFalse(distill.get("negative_prompt_support"))
        self.assertNotIn("guide_scale", distill)

    def test_hunyuan_sr_scheduler_sigmas(self) -> None:
        import mlx.core as mx

        from backend.engine.common.schedulers import FlowMatchEulerScheduler
        from backend.engine.families.hunyuan.sr_mlx import _configure_step_distill_scheduler

        class _Ctx:
            @staticmethod
            def float32():
                return mx.float32

            @staticmethod
            def array(x, dtype=None):
                return mx.array(x, dtype=dtype)

            @staticmethod
            def concat(parts, axis=0):
                return mx.concatenate(parts, axis=axis)

            @staticmethod
            def zeros(shape, dtype=None):
                return mx.zeros(shape, dtype=dtype)

        ctx = _Ctx()
        sched = FlowMatchEulerScheduler(ctx=ctx)
        timesteps = _configure_step_distill_scheduler(ctx, sched, 6)
        self.assertEqual(int(timesteps.shape[0]), 6)

    def test_hunyuan_registry_modelscope_repos(self) -> None:
        import json
        from pathlib import Path

        from backend.core.bundle_repos import bundle_repos_from_version

        reg = json.loads(
            (Path(__file__).resolve().parents[1] / "default_config" / "models_registry.json").read_text(
                encoding="utf-8",
            )
        )
        models = reg.get("models") or {}
        for mid in (
            "hunyuan-video-1.5-480p-t2v",
            "hunyuan-video-1.5-480p-i2v",
            "hunyuan-video-1.5-i2v-step-distill",
            "hunyuan-video-1.5-1080p-sr",
        ):
            m = models[mid]
            self.assertEqual(m.get("source"), "modelscope")
            ver = m["versions"]["original"]
            repos = bundle_repos_from_version(ver)
            self.assertGreaterEqual(len(repos), 1)
            self.assertEqual(repos[0]["repo_id"], "Tencent-Hunyuan/HunyuanVideo-1.5")
            self.assertIn("hunyuan_ms_variant", ver)
        t2v = bundle_repos_from_version(models["hunyuan-video-1.5-480p-t2v"]["versions"]["original"])
        self.assertEqual(len(t2v), 3)
        self.assertEqual(t2v[1]["repo_id"], "Qwen/Qwen2.5-VL-7B-Instruct")
        self.assertEqual(t2v[2]["repo_id"], "google/byt5-small")
        params = models["hunyuan-video-1.5-480p-t2v"]["parameters"]
        self.assertEqual(
            params["text_encoder_qwen_local"],
            "models/Text/qwen2.5-vl-7b-instruct",
        )
        self.assertTrue(params.get("text_encoder_release_after_encode"))
        self.assertNotIn("companion_repo_id", models["hunyuan-video-1.5-480p-t2v"]["versions"]["original"])
        self.assertNotIn("shared_te_local_path", models["hunyuan-video-1.5-480p-i2v"]["versions"]["original"])

    def test_hunyuan_te_path_resolution(self) -> None:
        import tempfile
        from pathlib import Path

        from backend.engine.families.hunyuan.text_encoder import _resolve_byt5_dirs, _resolve_qwen_dirs

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            native = root / "qwen"
            native.mkdir()
            (native / "config.json").write_text("{}", encoding="utf-8")
            enc, tok = _resolve_qwen_dirs(native)
            self.assertEqual(enc, native)
            self.assertEqual(tok, native)

            legacy = root / "legacy"
            (legacy / "text_encoder").mkdir(parents=True)
            (legacy / "tokenizer").mkdir()
            enc, tok = _resolve_qwen_dirs(legacy)
            self.assertEqual(enc, legacy / "text_encoder")

            byt5 = root / "byt5"
            byt5.mkdir()
            (byt5 / "config.json").write_text("{}", encoding="utf-8")
            enc2, tok2 = _resolve_byt5_dirs(byt5)
            self.assertEqual(enc2, byt5)

    def test_hunyuan_ms_bundle_assemble(self) -> None:
        import tempfile
        from pathlib import Path

        from backend.services.hunyuan_ms_bundle import assemble_hunyuan_modelscope_bundle

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            native = root / "transformer" / "480p_t2v"
            native.mkdir(parents=True)
            (native / "config.json").write_text("{}", encoding="utf-8")
            (native / "diffusion_pytorch_model.safetensors").write_text("x", encoding="utf-8")
            (root / "vae").mkdir()
            (root / "vae" / "config.json").write_text("{}", encoding="utf-8")

            assemble_hunyuan_modelscope_bundle(root, "480p_t2v")
            self.assertTrue((root / "transformer" / "config.json").is_file())
            self.assertFalse((root / "transformer" / "480p_t2v").exists())

    def test_torch_device_preferences(self) -> None:
        from backend.engine.common.text_encoders.torch_device import resolve_torch_inference_device

        self.assertEqual(resolve_torch_inference_device("cpu"), "cpu")
        with self.assertRaises(RuntimeError):
            resolve_torch_inference_device("bad-device")

    def test_hunyuan_text_encoder_cache(self) -> None:
        from pathlib import Path
        from unittest.mock import MagicMock, patch

        from backend.engine.families.hunyuan import text_encoder as te

        te._ENCODER_CACHE.clear()
        with patch.object(te, "HunyuanVideoTextEncoder") as cls:
            inst = MagicMock()
            cls.return_value = inst
            ctx = MagicMock()
            ctx.backend = "mlx"
            cfg = MagicMock()
            root = Path("/tmp/hunyuan-cache-test-bundle")
            a = te.get_hunyuan_text_encoder(ctx, root, cfg)
            b = te.get_hunyuan_text_encoder(ctx, root, cfg)
            self.assertIs(a, b)
            cls.assert_called_once()
        te._ENCODER_CACHE.clear()

    def test_hunyuan_text_encoder_requires_mlx(self) -> None:
        from pathlib import Path
        from unittest.mock import MagicMock

        from backend.engine.families.hunyuan import text_encoder as te

        ctx = MagicMock()
        ctx.backend = "cuda"
        with self.assertRaises(RuntimeError):
            te.get_hunyuan_text_encoder(ctx, Path("/tmp"), MagicMock())

    def test_hf_tokenizer_json_bpe(self) -> None:
        import json
        import tempfile
        from pathlib import Path

        from backend.engine.common.hf_tokenizer_json import HFTokenizerJson, render_qwen_chat_messages

        data = {
            "model": {
                "type": "BPE",
                "vocab": {"a": 0, "b": 1, "ab": 2, "<unk>": 3},
                "merges": [["a", "b"]],
            },
            "added_tokens": [{"id": 4, "content": "<|im_start|>", "special": True}],
            "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": False},
        }
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "tokenizer.json").write_text(json.dumps(data), encoding="utf-8")
            tok = HFTokenizerJson.from_directory(root)
            self.assertEqual(tok.encode("ab"), [2])
            self.assertEqual(tok.encode("<|im_start|>"), [4])

        chat = render_qwen_chat_messages(
            [
                {"role": "system", "content": [{"type": "text", "text": "hi"}]},
                {"role": "user", "content": [{"type": "text", "text": "go"}]},
            ]
        )
        self.assertIn("<|im_start|>system", chat)
        self.assertIn("<|im_start|>assistant", chat)

    def test_hf_tokenizer_json_bpe_earliest_merge(self) -> None:
        import json
        import tempfile
        from pathlib import Path

        from backend.engine.common.hf_tokenizer_json import HFTokenizerJson

        data = {
            "model": {
                "type": "BPE",
                "vocab": {">": 29, "pop": 8539, ">p": 100, "op": 101, "<unk>": 0},
                "merges": [["p", "o"], ["po", "p"], [">", "p"]],
            },
            "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": False},
        }
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "tokenizer.json").write_text(json.dumps(data), encoding="utf-8")
            tok = HFTokenizerJson.from_directory(root)
            self.assertEqual(tok.encode(">pop"), [29, 8539])

    def test_heartmula_hf_tokenizer_encode(self) -> None:
        from pathlib import Path

        from backend.engine.common.hf_tokenizer_json import HFTokenizerJson
        from backend.engine.families.heartmula.bundle import bundle_is_ready

        bundle = Path("models/Audio/heartmula-oss-3b-happy-new-year")
        if not bundle_is_ready(bundle):
            self.skipTest("HeartMuLa bundle not installed")

        tok = HFTokenizerJson.from_directory(bundle)
        sample = "<tag>pop, happy</tag>"
        ids = tok.encode(sample)
        self.assertEqual(ids, [17224, 29, 8539, 11, 6380, 524, 4681, 29])

    def test_byt5_tokenize_batch(self) -> None:
        import tempfile
        from pathlib import Path
        from unittest.mock import MagicMock, patch

        with tempfile.TemporaryDirectory() as td:
            tok_dir = Path(td)
            with patch("backend.engine.families.hunyuan.text_encoder.load_hf_tokenizer") as load:
                mock_tok = MagicMock()
                import numpy as np

                mock_tok.encode_batch.return_value = (
                    np.zeros((1, 8), dtype=np.int32),
                    np.zeros((1, 8), dtype=np.int32),
                )
                load.return_value = mock_tok
                from backend.engine.families.hunyuan.text_encoder import _byt5_tokenize_batch

                _byt5_tokenize_batch(["x"], tok_dir, 8)
                load.assert_called_once_with(str(tok_dir))
                mock_tok.encode_batch.assert_called_once()

    def test_cast_floating_mx_tree(self) -> None:
        import mlx.core as mx

        from backend.engine.common.mlx_dtype import cast_floating_mx_tree

        tree = {"a": mx.array([1.0], dtype=mx.float32), "b": mx.array([1], dtype=mx.int32)}
        out = cast_floating_mx_tree(tree, mx.bfloat16)
        self.assertEqual(out["a"].dtype, mx.bfloat16)
        self.assertEqual(out["b"].dtype, mx.int32)

    def test_hunyuan_release_weights(self) -> None:
        from unittest.mock import MagicMock

        from backend.engine.families.hunyuan.text_encoder import HunyuanVideoTextEncoder

        enc = HunyuanVideoTextEncoder.__new__(HunyuanVideoTextEncoder)
        qwen = MagicMock()
        byt5 = MagicMock()
        enc._qwen = qwen
        enc._byt5_mlx = byt5
        enc.release_weights()
        self.assertIsNone(enc._qwen)
        self.assertIsNone(enc._byt5_mlx)
        qwen.release_weights.assert_called_once()
        byt5.release_weights.assert_called_once()

    def test_hunyuan_encode_empty_fails(self) -> None:
        from unittest.mock import MagicMock
        from pathlib import Path

        from backend.engine.families.hunyuan.text_encoder import HunyuanVideoTextEncoder

        enc = HunyuanVideoTextEncoder.__new__(HunyuanVideoTextEncoder)
        enc.ctx = MagicMock()
        enc.bundle_root = Path("/tmp")
        enc.mllm_max_length = 1000
        enc.byt5_max_length = 256
        with self.assertRaises(RuntimeError):
            enc.encode([])

    def test_hunyuan_vae_spatial_tile_params(self) -> None:
        from backend.engine.families.hunyuan.vae_mlx import (
            _hunyuan_vae_tile_params,
            _needs_hunyuan_spatial_tiling,
        )

        cfg = {"spatial_compression_ratio": 16}
        params = _hunyuan_vae_tile_params(cfg)
        self.assertEqual(params.tile_sample_min_height, 256)
        self.assertEqual(params.tile_latent_min_height, 16)
        self.assertEqual(params.overlap_factor, 0.25)
        # 1080p-class latent grid (~120×68)
        self.assertTrue(_needs_hunyuan_spatial_tiling(True, 68, 120, params))
        self.assertFalse(_needs_hunyuan_spatial_tiling(False, 68, 120, params))
        # Small latent tile fits in one pass
        self.assertFalse(_needs_hunyuan_spatial_tiling(True, 16, 16, params))

    def test_hunyuan_sr_registry_spatial_tiling(self) -> None:
        import json
        from pathlib import Path

        reg = json.loads(
            (Path(__file__).resolve().parents[1] / "default_config" / "models_registry.json").read_text(
                encoding="utf-8",
            )
        )
        sr = reg["models"]["hunyuan-video-1.5-1080p-sr"]
        self.assertTrue(sr["parameters"].get("vae_spatial_tiling"))


class InstallHooksTests(unittest.TestCase):
    def test_install_hooks_from_version_parses_objects(self) -> None:
        from backend.core.install_hooks import install_hooks_from_version

        ver = {
            "install_hooks": [
                {"type": "heartmula_mlx_weights", "dtype": "bfloat16"},
                "other_hook",
            ]
        }
        hooks = install_hooks_from_version(ver)
        self.assertEqual(len(hooks), 2)
        self.assertEqual(hooks[0]["type"], "heartmula_mlx_weights")
        self.assertEqual(hooks[0]["dtype"], "bfloat16")
        self.assertEqual(hooks[1]["type"], "other_hook")

    def test_install_hooks_from_version_empty(self) -> None:
        from backend.core.install_hooks import install_hooks_from_version

        self.assertEqual(install_hooks_from_version(None), [])
        self.assertEqual(install_hooks_from_version({}), [])

    def test_heartmula_hook_runner_resolves(self) -> None:
        from backend.core.install_hooks import _resolve_runner

        fn = _resolve_runner("heartmula_mlx_weights")
        self.assertTrue(callable(fn))
        self.assertEqual(fn.__name__, "run_heartmula_mlx_weights")

    def test_unknown_hook_type_fails_loud(self) -> None:
        from backend.core.install_hooks import run_install_hooks

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(RuntimeError) as ctx:
                run_install_hooks(
                    model_name="test",
                    version_key="default",
                    ver_config={"install_hooks": [{"type": "nonexistent"}]},
                    bundle_root=root,
                )
            self.assertIn("nonexistent", str(ctx.exception))

    def test_prune_pytorch_weights_keeps_mlx_and_config(self) -> None:
        from backend.engine.families.heartmula.install_hook import prune_pytorch_weights

        with tempfile.TemporaryDirectory() as tmp:
            comp = Path(tmp) / "HeartMuLa-oss-3B"
            comp.mkdir()
            (comp / "config.json").write_text("{}", encoding="utf-8")
            (comp / "model.safetensors").write_bytes(b"pt")
            (comp / "mlx").mkdir()
            (comp / "mlx" / "model.safetensors").write_bytes(b"mlx")
            removed = prune_pytorch_weights(comp)
            self.assertEqual(removed, ["model.safetensors"])
            self.assertFalse((comp / "model.safetensors").exists())
            self.assertTrue((comp / "mlx" / "model.safetensors").is_file())
            self.assertTrue((comp / "config.json").is_file())

    def test_mlx_weights_ready_requires_both_components(self) -> None:
        from backend.engine.families.heartmula.bundle import mlx_weights_ready

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "tokenizer.json").write_text("{}", encoding="utf-8")
            (root / "gen_config.json").write_text("{}", encoding="utf-8")
            mula = root / "HeartMuLa-oss-3B"
            codec = root / "HeartCodec-oss"
            mula.mkdir()
            codec.mkdir()
            self.assertFalse(mlx_weights_ready(root))
            (mula / "mlx").mkdir(parents=True)
            (mula / "mlx" / "model.safetensors").write_bytes(b"x")
            self.assertFalse(mlx_weights_ready(root))
            (codec / "mlx").mkdir(parents=True)
            (codec / "mlx" / "model.safetensors").write_bytes(b"y")
            self.assertTrue(mlx_weights_ready(root))


class PipelineProgressBridgeTests(unittest.TestCase):
    def test_denoise_emit_matches_execution_context(self) -> None:
        from backend.core.contracts import ProgressEvent
        from backend.engine.pipelines.pipeline_progress import emit_denoise_progress
        from backend.engine.progress_bridge import make_pipeline_progress_callback

        events: list[ProgressEvent] = []

        class _Ctx:
            def on_progress(self, ev: ProgressEvent) -> None:
                events.append(ev)

        on_progress = make_pipeline_progress_callback(_Ctx())  # type: ignore[arg-type]
        emit_denoise_progress(on_progress, 1, 40)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].step, 1)
        self.assertEqual(events[0].total, 40)
        self.assertEqual(events[0].phase, "denoising")


class BenchmarkMetadataTests(unittest.TestCase):
    def test_exit_exempt_nonempty(self) -> None:
        self.assertIn("z-image-create", BENCHMARK_EXIT_EXEMPT_MISMATCH_VS_MFLUX)
        self.assertIn("qwen-image-rewrite", BENCHMARK_EXIT_EXEMPT_MISMATCH_VS_MFLUX)

    def test_z_image_turbo_enable_thinking_aligned_with_mflux(self) -> None:
        import json
        from pathlib import Path

        reg = json.loads(
            (Path(__file__).resolve().parents[1] / "default_config" / "models_registry.json").read_text(
                encoding="utf-8",
            )
        )
        turbo = reg["models"]["z-image-turbo"]
        self.assertTrue(
            turbo["parameters"].get("enable_thinking"),
            "z-image-turbo must keep enable_thinking=true to align tokenizer chat-template semantics with mflux",
        )


class MemoryPolicyTests(unittest.TestCase):
    def test_clamp_mlx_memory_limit_gb(self) -> None:
        from backend.engine.memory_policy import clamp_mlx_memory_limit_gb

        self.assertEqual(clamp_mlx_memory_limit_gb(120), 120)
        self.assertEqual(clamp_mlx_memory_limit_gb(8), 16)
        self.assertEqual(clamp_mlx_memory_limit_gb(999), 512)
        self.assertEqual(clamp_mlx_memory_limit_gb("bad", default=64), 64)

    def test_model_cache_single_slot(self) -> None:
        from backend.engine.common.cache import ModelCache

        cache = ModelCache(lambda: 120.0, max_entries=1, ttl_minutes=60)
        cache.put("a", object(), 5.0)
        self.assertEqual(cache.stats["cached_models"], 1)
        cache.put("b", object(), 3.0)
        stats = cache.stats
        self.assertEqual(stats["cached_models"], 1)
        self.assertEqual(stats["models"][0]["key"], "b")
        self.assertIsNone(cache.get("a"))
        self.assertIsNotNone(cache.get("b"))

    def test_model_cache_set_ttl_and_evict(self) -> None:
        from backend.engine.common.cache import ModelCache

        cache = ModelCache(lambda: 120.0, max_entries=2, ttl_minutes=10)
        cache.set_ttl_minutes(45)
        self.assertEqual(cache.stats["ttl_minutes"], 45)
        cache.put("x", {"v": 1}, 1.0)
        cache.evict("x")
        self.assertIsNone(cache.get("x"))

    def test_model_cache_purge_idle_after_ttl(self) -> None:
        from datetime import datetime, timedelta

        from backend.engine.common.cache import ModelCache

        cache = ModelCache(lambda: 120.0, ttl_minutes=30)
        cache.put("idle-model", object(), 2.0)
        with cache._lock:
            entry = cache._cache["idle-model"]
            entry.last_used = datetime.now() - timedelta(minutes=31)
        evicted = cache.purge_idle()
        self.assertEqual(evicted, ["idle-model"])
        self.assertEqual(cache.stats["cached_models"], 0)
        self.assertIsNone(cache.get("idle-model"))

    def test_mlx_context_apply_memory_limit(self) -> None:
        import os

        from backend.engine.runtime.mlx import MLXContext

        ctx = MLXContext(memory_limit_gb=80)
        self.assertEqual(ctx.memory_limit_gb, 80)
        self.assertEqual(os.environ.get("MLX_METAL_MEMORY_LIMIT"), "80")
        ctx.apply_memory_limit_gb(96)
        self.assertEqual(ctx.memory_limit_gb, 96)
        self.assertEqual(os.environ.get("MLX_METAL_MEMORY_LIMIT"), "96")


class RegistryProfilesTests(unittest.TestCase):
    def test_expand_profile_merges_fields(self) -> None:
        from backend.core.registry_profiles import expand_registry_document

        doc = {
            "profiles": {
                "flux2-base": {
                    "engine": "danqing-image",
                    "media": "image",
                    "parameters": {"steps": {"type": "int", "default": 4}},
                }
            },
            "models": {
                "flux2-klein-9b": {
                    "profile": "flux2-base",
                    "family": "flux2",
                    "name": {"en": "Flux2 Klein"},
                }
            },
        }
        expanded = expand_registry_document(doc)
        model = expanded["models"]["flux2-klein-9b"]
        self.assertEqual(model["engine"], "danqing-image")
        self.assertEqual(model["family"], "flux2")
        self.assertEqual(model["parameters"]["steps"]["default"], 4)

    def test_unknown_profile_fails_loud(self) -> None:
        from backend.core.registry_profiles import expand_registry_document

        with self.assertRaises(ValueError):
            expand_registry_document(
                {"models": {"x": {"profile": "missing", "family": "flux2"}}}
            )

    def test_validate_registry_document_catches_bad_profile(self) -> None:
        from backend.core.registry_profiles import validate_registry_document

        errors = validate_registry_document(
            {"models": {"x": {"profile": "nope", "family": "flux2"}}}
        )
        self.assertTrue(any("nope" in e for e in errors))

    def test_apply_standard_profile_preserves_expansion(self) -> None:
        import copy

        from backend.core.registry_profiles import apply_standard_profile, expand_registry_document

        doc = {
            "profiles": {
                "image_dit_standard": {
                    "engine": "danqing-image",
                    "type": "diffusion",
                    "category": "base_models",
                    "parameters": {
                        "lora_support": True,
                        "seed_support": True,
                        "preview_mode": {"type": "enum", "default": "stream", "options": ["stream", "none"]},
                        "preview_interval_steps": {"type": "int", "default": 2, "min": 1, "max": 8},
                        "preview_max_edge": {"type": "int", "default": 512, "min": 128, "max": 1024},
                    },
                },
                "video_dit_standard": {
                    "engine": "danqing-video",
                    "type": "video",
                    "category": "video_models",
                    "parameters": {"seed_support": True},
                },
            },
            "models": {
                "demo": {
                    "engine": "danqing-image",
                    "category": "base_models",
                    "type": "diffusion",
                    "family": "flux2",
                    "parameters": {
                        "steps": {"type": "int", "default": 4},
                        "preview_mode": {"type": "enum", "default": "stream", "options": ["stream", "none"]},
                        "preview_interval_steps": {"type": "int", "default": 2, "min": 1, "max": 8},
                        "preview_max_edge": {"type": "int", "default": 512, "min": 128, "max": 1024},
                        "lora_support": True,
                        "seed_support": True,
                    },
                }
            },
        }
        before = expand_registry_document(copy.deepcopy(doc))
        apply_standard_profile(doc)
        after = expand_registry_document(doc)
        self.assertEqual(before, after)
        self.assertEqual(doc["models"]["demo"]["profile"], "image_dit_standard")

    def test_audit_registry_document_flags_duplicate_params(self) -> None:
        from backend.core.registry_profiles import audit_registry_document

        doc = {
            "profiles": {
                "p1": {
                    "engine": "danqing-image",
                    "parameters": {"steps": {"type": "int", "default": 4}},
                }
            },
            "models": {
                "demo": {
                    "profile": "p1",
                    "family": "flux2",
                    "engine": "danqing-image",
                    "parameters": {"steps": {"type": "int", "default": 4}},
                }
            },
        }
        hints = audit_registry_document(doc)
        self.assertTrue(any("steps" in h and "duplicates profile" in h for h in hints))


class BundleManifestTests(unittest.TestCase):
    def test_scan_components_classifies_files(self) -> None:
        from backend.core.bundle_manifest import scan_components

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "transformer.safetensors").write_bytes(b"x")
            (root / "vae").mkdir(parents=True)
            (root / "vae" / "diffusion_pytorch_model.safetensors").write_bytes(b"y")
            (root / "text_encoder").mkdir(parents=True)
            (root / "text_encoder" / "model.safetensors").write_bytes(b"z")

            components = scan_components(root)
            self.assertIn("transformer", components)
            self.assertIn("vae", components)
            self.assertIn("text_encoder", components)

    def test_t5_encoder_bundle_paths_flux_layout(self) -> None:
        from backend.engine.common.bundle_layout import t5_encoder_bundle_paths

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "text_encoder_2").mkdir()
            (root / "text_encoder_2" / "model.safetensors").write_bytes(b"x")
            (root / "tokenizer_2").mkdir()
            (root / "tokenizer_2" / "tokenizer.json").write_text("{}", encoding="utf-8")
            enc, tok = t5_encoder_bundle_paths(root)
            self.assertTrue(enc.endswith("text_encoder_2"))
            self.assertTrue(tok.endswith("tokenizer_2"))

    def test_assert_media_bundle_ready_none_raises(self) -> None:
        from backend.engine.common.bundle_layout import assert_media_bundle_ready

        with self.assertRaises(RuntimeError) as ctx:
            assert_media_bundle_ready(None, family="flux2", model_id="demo")
        self.assertIn("no installed bundle", str(ctx.exception))

    def test_assert_bundle_ready_missing_component(self) -> None:
        from backend.core.bundle_manifest import assert_bundle_ready_for_family

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "transformer.safetensors").write_bytes(b"x")
            with self.assertRaises(RuntimeError) as ctx:
                assert_bundle_ready_for_family(root, family="flux2", model_id="test-model")
            self.assertIn("text_encoder", str(ctx.exception))

    def test_bundle_component_status_flags_missing(self) -> None:
        from backend.core.bundle_manifest import bundle_component_status

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "transformer.safetensors").write_bytes(b"x")
            status = bundle_component_status(root, family="flux2")
            self.assertIsNotNone(status)
            assert status is not None
            self.assertFalse(status["complete"])
            self.assertIn("text_encoder", status["missing"])


if __name__ == "__main__":
    unittest.main()
