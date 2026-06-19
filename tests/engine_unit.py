"""Backend engine unit tests (no weights, no GPU)."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from backend.engine.common.model.base import _mlx_affine_infer_bits_and_group_size
from backend.engine.families.ltx.weights import remap_ltx_weights


def _load_default_registry_expanded() -> dict:
    import json

    from backend.catalog.loader import expand_catalog_document

    path = Path(__file__).resolve().parents[1] / "default_config" / "models_registry.json"
    return expand_catalog_document(json.loads(path.read_text(encoding="utf-8")))


def _t(shape: tuple[int, ...]) -> SimpleNamespace:
    return SimpleNamespace(shape=shape)


class StructuralGuideTests(unittest.TestCase):
    def test_infer_guide_type_and_lora_map(self) -> None:
        from backend.engine.families.flux1.structural import (
            CONTROLNET_LORA_MAP,
            companion_lora_id,
            infer_guide_type,
        )

        self.assertEqual(infer_guide_type("flux-canny-controlnet"), "canny")
        self.assertEqual(infer_guide_type("flux-depth-controlnet"), "depth")
        self.assertEqual(infer_guide_type("flux-redux"), "redux")
        self.assertEqual(companion_lora_id("flux-canny-controlnet"), "flux1-canny-dev-lora")
        self.assertIn("flux-depth-controlnet", CONTROLNET_LORA_MAP)
        from backend.engine.families.flux1.structural import is_fill_controlnet, is_redux_controlnet

        self.assertTrue(is_fill_controlnet("flux-fill-controlnet"))
        self.assertTrue(is_redux_controlnet("flux-redux"))
        self.assertIsNone(companion_lora_id("flux-redux"))


class ZImageEnhancementTests(unittest.TestCase):
    def test_lemica_schedules(self) -> None:
        from backend.engine.families.z_image.transformer_mlx import lemica_compute_steps

        medium = lemica_compute_steps("medium", 8)
        self.assertIsNotNone(medium)
        assert medium is not None
        self.assertEqual(len(medium), 8)
        self.assertEqual(sum(1 for x in medium if x), 6)
        self.assertIsNone(lemica_compute_steps("none", 8))

    def test_z_image_structural_infer_and_augment(self) -> None:
        from backend.engine.families.z_image.structural import infer_guide_type

        self.assertEqual(
            infer_guide_type("z-image-turbo-fun-controlnet-union"),
            "auto",
        )

    def test_remap_zimage_control_keys(self) -> None:
        from backend.engine.families.z_image.weights import remap_zimage_control_weights

        out = remap_zimage_control_weights(
            {"control_all_x_embedder.2-1.weight": "t", "layers.0.weight": "skip"}
        )
        self.assertIn("control_x_embedder.weight", out)
        self.assertNotIn("layers.0.weight", out)

    def test_structural_guide_inpaint_pair_validation(self) -> None:
        from backend.core.contracts import StructuralGuide
        from pydantic import ValidationError

        with self.assertRaises(ValidationError):
            StructuralGuide(asset_id="a1", inpaint_source_asset_id="s1")
        ok = StructuralGuide(
            asset_id="a1",
            inpaint_source_asset_id="s1",
            inpaint_mask_asset_id="m1",
        )
        self.assertEqual(ok.inpaint_source_asset_id, "s1")

    def test_z_image_weighted_sum_merge_numpy(self) -> None:
        import numpy as np

        from backend.engine.tools.z_image_merge import weighted_sum_merge

        class _Ctx:
            pass

        wa = {"w": np.array([1.0, 0.0], dtype=np.float32)}
        wb = {"w": np.array([0.0, 1.0], dtype=np.float32)}
        out = weighted_sum_merge(wa, wb, alpha=0.5, ctx=_Ctx())
        np.testing.assert_allclose(out["w"], [0.5, 0.5], rtol=1e-5)

    def test_esrgan_weight_remap(self) -> None:
        from backend.engine.families.esrgan.weights import remap_esrgan_weights

        out = remap_esrgan_weights({"body.0.rdb1.conv1.weight": "t", "conv_first.weight": "x"})
        self.assertIn("body_0.rdb1.conv1.weight", out)
        self.assertEqual(out["conv_first.weight"], "x")

    def test_esrgan_find_weight_nested(self) -> None:
        import tempfile
        from pathlib import Path

        import mlx.core as mx

        from backend.engine.families.esrgan.stem_mlx import _find_esrgan_weight_file, validate_esrgan_bundle

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            nested = root / "mlx-community" / "Real-ESRGAN-x4plus"
            nested.mkdir(parents=True)
            sf = nested / "model.safetensors"
            mx.save_safetensors(str(sf), {"conv_first.weight": mx.zeros((1,))})
            found = _find_esrgan_weight_file(root)
            self.assertEqual(found.name, "model.safetensors")
            validate_esrgan_bundle(root)

    def test_validate_esrgan_bundle_missing(self) -> None:
        import tempfile
        from pathlib import Path

        from backend.engine.families.esrgan.stem_mlx import validate_esrgan_bundle

        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(RuntimeError):
                validate_esrgan_bundle(Path(td))

    def test_image_edit_request_z_image_enhancements(self) -> None:
        from backend.core.contracts import ImageEditRequest, LatentRefineSpec, StructuralGuide

        req = ImageEditRequest(
            model="z-image-turbo:fp16",
            operation="rewrite",
            source_asset_id="ast_src",
            prompt="test",
            structural_guide=StructuralGuide(
                asset_id="ast_ctrl",
                model_id="z-image-turbo-fun-controlnet-union",
                type="auto",
            ),
            lemica_mode="medium",
            latent_refine=LatentRefineSpec(scale=1.5, denoise_strength=0.35),
        )
        self.assertEqual(req.lemica_mode, "medium")
        self.assertEqual(req.latent_refine.scale, 1.5)

    def test_latent_refine_reads_entry_family_not_runtime(self) -> None:
        from unittest.mock import patch

        from backend.core.contracts import LatentRefineSpec
        from backend.engine.families.z_image.latent_refine import apply_latent_refine_if_requested

        entry = SimpleNamespace(family="z_image")
        request = SimpleNamespace(
            latent_refine=LatentRefineSpec(scale=1.5, denoise_strength=0.35),
            seed=0,
        )
        latents = SimpleNamespace(shape=(16, 1, 96, 60), ndim=4)

        with patch(
            "backend.engine.common.mlx_only.require_mlx_backend",
            side_effect=RuntimeError("family_check_passed"),
        ):
            with self.assertRaises(RuntimeError) as ctx:
                apply_latent_refine_if_requested(
                    pipeline=SimpleNamespace(ctx=SimpleNamespace(backend="mlx")),
                    latents=latents,
                    request=request,
                    entry=entry,
                    version_key="fp16",
                    model=None,
                    timesteps=[],
                    sigmas=None,
                    txt_embeds=None,
                    neg_embeds=None,
                    guidance=0.0,
                    extra_cond={},
                )
        self.assertEqual(str(ctx.exception), "family_check_passed")

    def test_latent_refine_resize_uses_mlx_nn_upsample(self) -> None:
        import mlx.core as mx

        from backend.engine.families.z_image.latent_refine import _resize_latent_nchw
        from backend.engine.runtime.mlx import MLXContext

        ctx = MLXContext()
        latents = mx.random.normal((16, 1, 96, 60))
        out = _resize_latent_nchw(ctx, latents, 144, 90, mode="linear")
        self.assertEqual(tuple(out.shape), (16, 1, 144, 90))

    def test_merged_model_id_slug(self) -> None:
        from backend.engine.tools.user_merged_model_registry import merged_model_id_from_output_name

        self.assertEqual(merged_model_id_from_output_name("MyBlend_v2"), "z-image-merged-myblend-v2")

    def test_mlx_only_guard_cuda_raises(self) -> None:
        from backend.engine.common.mlx_only import require_mlx_backend, require_mlx_if_option_active

        class _CudaCtx:
            backend = "cuda"

        with self.assertRaises(RuntimeError) as ctx:
            require_mlx_backend(_CudaCtx(), feature="lemica_mode")
        self.assertIn("MLX-only", str(ctx.exception))

        with self.assertRaises(RuntimeError):
            require_mlx_if_option_active(_CudaCtx(), feature="lemica_mode", option="fast")

        require_mlx_if_option_active(_CudaCtx(), feature="lemica_mode", option="none")

    def test_esrgan_stem_cuda_placeholder(self) -> None:
        import tempfile
        from pathlib import Path

        from backend.engine.families.esrgan.stem_cuda import run_esrgan_upscale

        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(RuntimeError) as ctx:
                run_esrgan_upscale(
                    bundle_path=Path(td),
                    model_key="real-esrgan-x4plus",
                    source_image=Path(td) / "x.png",
                    scale=4,
                    softness=0.0,
                    seed=0,
                    output_png=Path(td) / "out.png",
                )
            self.assertIn("MLX-only", str(ctx.exception))

    def test_register_merged_z_image_model(self) -> None:
        import json
        import tempfile
        from pathlib import Path

        from backend.core.model_registry import ModelRegistry
        from backend.engine.tools.user_merged_model_registry import (
            list_user_merged_models,
            register_merged_z_image_model,
        )

        template = {
            "catalog": {
                "media": "image",
                "name": {"zh": "Z-Image-Turbo", "en": "Z-Image-Turbo"},
                "engine": "danqing-image",
                "type": "diffusion",
                "category": "base_models",
            },
            "runtime": {
                "family": "z_image",
                "backends": ["mlx", "cuda"],
                "overrides": {"supports_guidance": False},
            },
            "actions": {"create": {}},
            "ui": {"extends": "image_dit_standard"},
            "distribution": {
                "versions": {
                    "fp16": {"local_path": "models/Image/z-image-turbo-fp16", "default": True},
                }
            },
        }
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cfg = root / "config"
            cfg.mkdir()
            reg_path = cfg / "models_registry.json"
            reg_path.write_text(
                json.dumps({"schema_version": 3, "models": {"z-image-turbo": template}}),
                encoding="utf-8",
            )
            manifest = {"merge_method": "weighted_sum", "alpha": 0.5}
            row = register_merged_z_image_model(
                registry_path=reg_path,
                config_dir=cfg,
                output_name="demo-blend",
                local_path="models/Image/demo-blend-fp16",
                template_model_id="z-image-turbo",
                merge_manifest=manifest,
                task_id="tsk_test",
            )
            self.assertEqual(row["id"], "z-image-merged-demo-blend")
            self.assertEqual(len(list_user_merged_models(cfg)), 1)
            reg = ModelRegistry.load(reg_path)
            entry = reg.require("z-image-merged-demo-blend")
            self.assertEqual(entry.family, "z_image")
            ver = entry.raw.get("versions") or {}
            self.assertEqual(ver["fp16"]["local_path"], "models/Image/demo-blend-fp16")
            reg.reload()
            self.assertIsNotNone(reg.get("z-image-merged-demo-blend"))


class ImageCliGenerateTests(unittest.TestCase):
    def test_structural_guide_requires_control_asset(self) -> None:
        from backend.cli.image_cli import generate

        with self.assertRaises(ValueError) as ctx:
            generate(
                "flux1-dev",
                "test",
                controlnet="flux-canny-controlnet",
                project_root=Path(__file__).resolve().parents[1],
            )
        self.assertIn("control-asset-id", str(ctx.exception).lower())

    def test_fill_rejected_on_generate(self) -> None:
        from backend.cli.image_cli import generate

        with self.assertRaises(ValueError) as ctx:
            generate(
                "flux1-dev",
                "test",
                controlnet="flux-fill-controlnet",
                control_asset_id="ast_fake",
                project_root=Path(__file__).resolve().parents[1],
            )
        self.assertIn("retouch", str(ctx.exception).lower())


class ImageCliEditTests(unittest.TestCase):
    def test_retouch_requires_mask(self) -> None:
        from backend.cli.image_cli import edit

        with self.assertRaises(ValueError) as ctx:
            edit(
                "flux-fill-controlnet",
                "retouch",
                "test",
                source_asset_id="ast_fake",
                project_root=Path(__file__).resolve().parents[1],
            )
        self.assertIn("mask", str(ctx.exception).lower())

    def test_extend_requires_directions(self) -> None:
        from backend.cli.image_cli import edit

        with self.assertRaises(ValueError) as ctx:
            edit(
                "flux-fill-controlnet",
                "extend",
                "test",
                source_asset_id="ast_fake",
                project_root=Path(__file__).resolve().parents[1],
            )
        self.assertIn("extend-directions", str(ctx.exception).lower())


class RegistryActionTests(unittest.TestCase):
    def test_fill_api_actions_collapse_to_edit(self) -> None:
        from backend.core.registry_format import api_action_frozenset, registry_declares_action

        acts = {"retouch": {}, "extend": {}}
        self.assertEqual(api_action_frozenset(acts, media="image"), frozenset({"edit"}))
        self.assertTrue(registry_declares_action(acts, "extend"))
        self.assertTrue(registry_declares_action(acts, "retouch"))
        self.assertFalse(registry_declares_action(acts, "create"))


class FluxFillPatchEmbedTests(unittest.TestCase):
    def test_fill_patch_token_dim_matches_x_embedder(self) -> None:
        from backend.engine.config.model_configs import Flux1Config
        from backend.engine.families.flux1.fill_edit import FILL_PATCH_TOKEN_DIM
        from backend.engine.families.flux1.transformer_mlx import Flux1DiTMLX
        from backend.engine.runtime.mlx import MLXContext

        ctx = MLXContext()
        config = Flux1Config(patch_token_dim=FILL_PATCH_TOKEN_DIM, supports_guidance=True)
        model = Flux1DiTMLX(config, ctx)
        w = model._param_map["patch_embed.proj.weight"]
        self.assertEqual(tuple(w.shape), (3072, 1, 1, FILL_PATCH_TOKEN_DIM))


class FluxFillMaskTests(unittest.TestCase):
    def test_outpaint_mask_layout(self) -> None:
        from PIL import Image

        from backend.engine.families.flux1.fill_edit import (
            build_outpaint_image_and_mask,
            mask_pil_to_weight,
            reshape_mask_latent_channels,
        )

        src = Image.new("RGB", (64, 64), color=(128, 64, 32))
        canvas, mask = build_outpaint_image_and_mask(src, ["right"], 64)
        self.assertEqual(canvas.size, (128, 64))
        m = mask_pil_to_weight(mask)
        self.assertEqual(float(m[:64, :64].max()), 0.0)
        self.assertGreater(float(m[:, 64:].max()), 0.5)
        packed = reshape_mask_latent_channels(m, 64, 128)
        self.assertEqual(packed.shape, (1, 64, 8, 16))


class ControlNetRuntimeTests(unittest.TestCase):
    def test_declared_backends_mlx_placeholder(self) -> None:
        from backend.engine.families.flux1.structural import (
            CONTROLNET_CUDA_BATCH_PLANNED,
            CONTROLNET_DECLARED_BACKENDS,
        )

        self.assertEqual(CONTROLNET_DECLARED_BACKENDS, ("mlx",))
        self.assertTrue(CONTROLNET_CUDA_BATCH_PLANNED)

    def test_require_fails_on_non_mlx_context(self) -> None:
        from types import SimpleNamespace
        from unittest.mock import patch

        from backend.engine.families.flux1.structural import require_controlnet_runtime

        fake_ctx = SimpleNamespace()
        with patch(
            "backend.engine.families.flux1.structural.controlnet_runtime_available",
            return_value=True,
        ):
            with self.assertRaises(RuntimeError) as ctx:
                require_controlnet_runtime(fake_ctx, feature="structural_guide")
        msg = str(ctx.exception).lower()
        self.assertIn("mlx-only", msg)
        self.assertIn("cuda", msg)


class DepthProMlxTests(unittest.TestCase):
    def test_depth_encode_mlx_when_bundle_present(self) -> None:
        import json

        import numpy as np
        from PIL import Image

        pointer = (
            Path(__file__).resolve().parents[1]
            / "default_config"
            / "workspace.pointer.json"
        )
        if not pointer.is_file():
            self.skipTest("workspace.pointer.json missing")
        ws = json.loads(pointer.read_text(encoding="utf-8")).get("custom_workspace_dir")
        bundle = Path(ws) / "models" / "Tools" / "depth-pro-fp16"
        if not (bundle / "model.safetensors").is_file():
            self.skipTest("depth-pro bundle not installed")

        from backend.engine.families.flux1.depth_encode_mlx import estimate_depth_rgb01_mlx

        img = Image.new("RGB", (256, 256), color=(40, 120, 200))
        out = estimate_depth_rgb01_mlx(
            img, width=256, height=256, depth_bundle_root=bundle,
        )
        self.assertEqual(out.shape, (256, 256, 3))
        self.assertEqual(out.dtype, np.float32)
        self.assertGreaterEqual(float(out.min()), 0.0)
        self.assertLessEqual(float(out.max()), 1.0)


class FluxReduxParityTests(unittest.TestCase):
    def test_redux_mlx_matches_cuda_when_bundle_present(self) -> None:
        import json

        import numpy as np
        from PIL import Image

        pointer = (
            Path(__file__).resolve().parents[1]
            / "default_config"
            / "workspace.pointer.json"
        )
        if not pointer.is_file():
            self.skipTest("workspace.pointer.json missing")
        ws = json.loads(pointer.read_text(encoding="utf-8")).get("custom_workspace_dir")
        bundle = Path(ws) / "models" / "ControlNet" / "flux-redux-fp16"
        if not bundle.is_dir():
            self.skipTest("flux-redux bundle not installed")

        from backend.engine.families.flux1.redux_encode_cuda import encode_redux_context_tokens_cuda
        from backend.engine.families.flux1.redux_encode_mlx import encode_redux_context_tokens_mlx

        img = Image.new("RGB", (512, 512), color=(40, 120, 200))
        out_mlx = encode_redux_context_tokens_mlx(img, redux_bundle_root=bundle)
        out_cuda = encode_redux_context_tokens_cuda(img, redux_bundle_root=bundle)
        self.assertEqual(out_mlx.shape, out_cuda.shape)
        self.assertEqual(out_mlx.shape[1:], (729, 4096))
        cos = float(
            np.sum(out_mlx.flatten() * out_cuda.flatten())
            / (np.linalg.norm(out_mlx) * np.linalg.norm(out_cuda) + 1e-8)
        )
        # Native MLX SigLIP + redux MLP — not bit-identical to HF torch path.
        self.assertGreater(cos, 0.85)


class FluxDepthParityTests(unittest.TestCase):
    def test_depth_mlx_matches_cuda_when_bundle_present(self) -> None:
        import importlib.util
        import json

        import numpy as np
        from PIL import Image

        if importlib.util.find_spec("torchvision") is None:
            self.skipTest("torchvision required for depth-pro CUDA reference path")

        pointer = (
            Path(__file__).resolve().parents[1]
            / "default_config"
            / "workspace.pointer.json"
        )
        if not pointer.is_file():
            self.skipTest("workspace.pointer.json missing")
        ws = json.loads(pointer.read_text(encoding="utf-8")).get("custom_workspace_dir")
        bundle = Path(ws) / "models" / "Tools" / "depth-pro-fp16"
        if not (bundle / "model.safetensors").is_file():
            self.skipTest("depth-pro bundle not installed")

        from backend.engine.families.flux1.depth_encode_cuda import estimate_depth_rgb01_cuda
        from backend.engine.families.flux1.depth_encode_mlx import estimate_depth_rgb01_mlx

        img = Image.new("RGB", (512, 512), color=(40, 120, 200))
        out_mlx = estimate_depth_rgb01_mlx(
            img, width=256, height=256, depth_bundle_root=bundle,
        )
        out_cuda = estimate_depth_rgb01_cuda(
            img, width=256, height=256, depth_bundle_root=bundle,
        )
        self.assertEqual(out_mlx.shape, out_cuda.shape)
        cos = float(
            np.sum(out_mlx.flatten() * out_cuda.flatten())
            / (np.linalg.norm(out_mlx) * np.linalg.norm(out_cuda) + 1e-8)
        )
        # Native Depth Pro MLX reimplementation — min-max RGB postprocess aligns; internals differ.
        self.assertGreater(cos, 0.90)


class CogView4GlmEncoderMlxTests(unittest.TestCase):
    def test_glm_encode_mlx_matches_torch_when_bundle_present(self) -> None:
        import json

        import mlx.core as mx
        import numpy as np

        pointer = (
            Path(__file__).resolve().parents[1]
            / "default_config"
            / "workspace.pointer.json"
        )
        if not pointer.is_file():
            self.skipTest("workspace.pointer.json missing")
        ws = json.loads(pointer.read_text(encoding="utf-8")).get("custom_workspace_dir")
        bundle = Path(ws) / "models" / "Image" / "cogview4-6b-bf16"
        te_dir = bundle / "text_encoder"
        tok_dir = bundle / "tokenizer"
        if not (te_dir / "config.json").is_file():
            self.skipTest("cogview4-6b text encoder bundle not installed")

        from backend.engine.families.cogview4.text_encoder import CogView4TextEncoder
        from backend.engine.families.cogview4.text_encoder_cuda import build_glm4_text_encoder_torch
        from backend.engine.runtime.mlx import MLXContext

        ctx = MLXContext()
        enc = CogView4TextEncoder(
            ctx,
            str(te_dir),
            tokenizer_path=str(tok_dir),
            max_seq_len=256,
        )
        prompt = "a studio photo of a red backpack"
        out_mlx = np.array(enc.encode([prompt]).astype(mx.float32))
        torch_enc = build_glm4_text_encoder_torch(str(te_dir))
        np_ids = enc._tokenize_glm_np([prompt]).astype(np.int64)
        out_torch = torch_enc.encode_numpy(np_ids)
        self.assertEqual(out_mlx.shape, out_torch.shape)
        cos = float(
            np.sum(out_mlx.flatten() * out_torch.flatten())
            / (np.linalg.norm(out_mlx) * np.linalg.norm(out_torch) + 1e-8)
        )
        self.assertGreater(cos, 0.999)


class CogView4DiTForwardSmokeTests(unittest.TestCase):
    def test_forward_smoke(self) -> None:
        import mlx.core as mx

        from backend.engine.config.model_configs import CogView4Config
        from backend.engine.families.cogview4.transformer_mlx import CogView4DiTMLX
        from backend.engine.runtime.mlx import MLXContext

        cfg = CogView4Config(num_layers=2)
        model = CogView4DiTMLX(cfg, MLXContext())
        latents = mx.zeros((1, 16, 32, 32), dtype=mx.bfloat16)
        txt = mx.zeros((1, 16, 4096), dtype=mx.bfloat16)
        sigmas = mx.array([1.0, 0.75, 0.5, 0.25], dtype=mx.float32)
        out = model.forward(
            latents,
            0,
            txt_embeds=txt,
            sigmas=sigmas,
            original_size=(512, 512),
            target_size=(512, 512),
            crop_coords=(0, 0),
        )
        self.assertEqual(tuple(out.shape), (1, 16, 32, 32))

    def test_encode_output_fits_dit_when_bundle_present(self) -> None:
        import json

        import mlx.core as mx

        pointer = (
            Path(__file__).resolve().parents[1]
            / "default_config"
            / "workspace.pointer.json"
        )
        if not pointer.is_file():
            self.skipTest("workspace.pointer.json missing")
        ws = json.loads(pointer.read_text(encoding="utf-8")).get("custom_workspace_dir")
        bundle = Path(ws) / "models" / "Image" / "cogview4-6b-bf16"
        te_dir = bundle / "text_encoder"
        tok_dir = bundle / "tokenizer"
        if not (te_dir / "config.json").is_file():
            self.skipTest("cogview4-6b text encoder bundle not installed")

        from backend.engine.config.model_configs import CogView4Config
        from backend.engine.families.cogview4.text_encoder import CogView4TextEncoder
        from backend.engine.families.cogview4.transformer_mlx import CogView4DiTMLX
        from backend.engine.runtime.mlx import MLXContext

        ctx = MLXContext()
        enc = CogView4TextEncoder(
            ctx,
            str(te_dir),
            tokenizer_path=str(tok_dir),
            max_seq_len=256,
        )
        txt = enc.encode(["一只橘猫"])
        self.assertEqual(int(txt.shape[-1]), 4096)
        self.assertGreater(int(txt.shape[1]), 0)

        model = CogView4DiTMLX(CogView4Config(num_layers=2), ctx)
        latents = mx.zeros((1, 16, 32, 32), dtype=mx.bfloat16)
        sigmas = mx.array([1.0, 0.5], dtype=mx.float32)
        out = model.forward(
            latents,
            0,
            txt_embeds=txt,
            sigmas=sigmas,
            original_size=(512, 512),
            target_size=(512, 512),
            crop_coords=(0, 0),
        )
        self.assertEqual(tuple(out.shape), (1, 16, 32, 32))


class CogView4SchedulerTests(unittest.TestCase):
    def test_calculate_shift_mu_512_latent(self) -> None:
        from backend.engine.common.ops.schedulers import cogview4_calculate_shift_mu

        mu = cogview4_calculate_shift_mu(1024, base_seq_len=256, base_shift=0.25, max_shift=0.75)
        self.assertAlmostEqual(mu, 1.75, places=4)

    def test_scheduler_sigmas_endpoints(self) -> None:
        from backend.engine.common.ops.schedulers import FlowMatchEulerCogView4Scheduler
        from backend.engine.runtime.mlx import MLXContext

        sched = FlowMatchEulerCogView4Scheduler(ctx=MLXContext())
        sched.set_timesteps(
            4,
            image_seq_len=1024,
            scheduler_base_image_seq_len=256,
            scheduler_base_shift=0.25,
            scheduler_max_shift=0.75,
        )
        self.assertAlmostEqual(float(sched.sigmas[0]), 1.0, places=3)
        self.assertAlmostEqual(float(sched.timesteps[0]), 1000.0, places=3)


class CogView4RegistryTests(unittest.TestCase):
    def test_registry_declares_mlx_only(self) -> None:
        entry = _load_default_registry_expanded()["models"]["cogview4-6b"]
        self.assertEqual(entry.get("backends"), ["mlx"])
        self.assertEqual(entry.get("family"), "cogview4")


class ControlNetScopeTests(unittest.TestCase):
    def test_scope_filter(self) -> None:
        from backend.api.routes.settings import _controlnet_matches_scope

        self.assertTrue(_controlnet_matches_scope({}, "create"))
        self.assertTrue(_controlnet_matches_scope({}, None))
        self.assertFalse(_controlnet_matches_scope({"retouch": {}}, "create"))
        self.assertFalse(_controlnet_matches_scope({"extend": {}}, "create"))
        self.assertTrue(_controlnet_matches_scope({"retouch": {}}, "retouch"))
        self.assertTrue(_controlnet_matches_scope({"extend": {}}, "extend"))
        self.assertFalse(_controlnet_matches_scope({}, "retouch"))


class RuntimeContractTests(unittest.TestCase):
    def test_family_runtime_guidance_semantics(self) -> None:
        from backend.engine.contracts.runtime_contracts import FamilyRuntimeContract

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

        from backend.engine.contracts.runtime_contracts import FamilyRuntimeContract

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
        from backend.engine.contracts.runtime_contracts import SchedulerSemanticsResolver

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
        from backend.engine.contracts.runtime_contracts import SchedulerSemanticsResolver

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

    def test_z_image_scheduler_matches_diffusers_static_shift(self) -> None:
        import json
        from pathlib import Path

        import numpy as np
        from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

        from backend.engine.contracts.runtime_contracts import SchedulerSemanticsResolver
        from backend.engine.common.ops.schedulers import FlowMatchEulerScheduler, get_scheduler
        from backend.engine.runtime.mlx import MLXContext

        reg = _load_default_registry_expanded()
        entry = SimpleNamespace(parameters=reg["models"]["z-image"]["parameters"])
        resolver = SchedulerSemanticsResolver()
        sem = resolver.resolve(
            entry=entry,
            config=SimpleNamespace(requires_sigma_shift=True),
            request_scheduler=None,
            request_metadata={},
            steps=50,
            width=768,
            height=1024,
        )
        self.assertFalse(sem.use_empirical_mu)
        self.assertEqual(sem.set_timesteps_kwargs.get("scheduler_shift"), 6.0)

        ctx = MLXContext()
        sched = get_scheduler(sem.scheduler_name, ctx=ctx)
        self.assertIsInstance(sched, FlowMatchEulerScheduler)
        sched.set_timesteps(**sem.set_timesteps_kwargs)
        danqing_sigmas = np.array(sched.sigmas)

        official = FlowMatchEulerDiscreteScheduler.from_config(
            {"num_train_timesteps": 1000, "use_dynamic_shifting": False, "shift": 6.0}
        )
        official.sigma_min = 0.0
        official.set_timesteps(50)
        ref_sigmas = official.sigmas.cpu().numpy()

        self.assertLess(
            float(np.max(np.abs(danqing_sigmas[:51] - ref_sigmas[:51]))),
            1e-5,
            "z-image scheduler sigmas should match diffusers static shift=6.0",
        )


class VideoRuntimeContractTests(unittest.TestCase):
    def test_video_contract_config_flags(self) -> None:
        from backend.engine.contracts import video_encoder_type
        from backend.engine.config.model_configs import WanConfig
        from backend.engine.video_codec_registry import get_video_decode_handler

        wan = WanConfig()
        self.assertEqual(str(getattr(wan, "video_vae_backend", "")), "wan")
        self.assertTrue(bool(getattr(wan, "uses_wan_shift", False)))
        self.assertTrue(bool(getattr(wan, "release_t5_after_encode", False)))
        self.assertIsNotNone(get_video_decode_handler("wan"))
        self.assertEqual(video_encoder_type(wan), "t5")


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

    def test_ltx_timestep_embedder_callable(self) -> None:
        import mlx.core as mx

        from backend.engine.families.ltx.transformer_mlx import LTX23AdaLayerNormSingle

        adaln = LTX23AdaLayerNormSingle(4096, num_params=9, timestep_dim=256)
        t_emb = mx.zeros((1, 256))
        params, embedded = adaln(t_emb)
        self.assertEqual(tuple(params.shape), (1, 9 * 4096))
        self.assertEqual(tuple(embedded.shape), (1, 4096))

    def test_ltx_upsampler_weight_path_dgrauet_bundle(self) -> None:
        from tests.benchmark.registry_utils import resolve_benchmark_data_root

        from backend.engine.families.ltx.vae_mlx import _resolve_upsampler_weight_path

        bundle = resolve_benchmark_data_root() / "models/Video/ltx-2.3-distilled-mlx-q4"
        if not bundle.is_dir():
            return
        path = _resolve_upsampler_weight_path(bundle, "spatial_x2")
        self.assertEqual(path.name, "spatial_upscaler_x2_v1_1.safetensors")

        from backend.engine.families.ltx.vae_mlx import load_ltx23_latent_upsampler
        from backend.engine.runtime.mlx import MLXContext

        ctx = MLXContext()
        up = load_ltx23_latent_upsampler(bundle, load_fn=ctx.load_weights)
        self.assertEqual(up.mid_channels, 1024)

    def test_ltx_video_decoder_per_channel_stats(self) -> None:
        import mlx.nn as nn

        from backend.engine.families.ltx.vae_mlx import LTX23VideoDecoder

        dec = LTX23VideoDecoder(causal=False, spatial_padding_mode="zeros")
        stats = dec.per_channel_statistics.parameters()
        self.assertIn("mean", stats)
        self.assertIn("std", stats)
        self.assertNotIn("mean_of_means", stats)


class ZImageCudaTests(unittest.TestCase):
    def test_transformer_dispatch_mlx(self) -> None:
        from backend.engine.config.model_configs import ZImageConfig
        from backend.engine.families.z_image.transformer import ZImageTransformer
        from backend.engine.families.z_image.transformer_mlx import ZImageDiTMLX as ZImageMLX
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
        from backend.engine.families.z_image.transformer_cuda import ZImageDiTCuda
        from backend.engine.runtime.cuda import CudaContext

        model = ZImageTransformer(ZImageConfig(), CudaContext("cpu"))
        self.assertIsInstance(model._inner, ZImageDiTCuda)

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

    def test_combine_cfg_noise_z_image_convention(self) -> None:
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
        from backend.engine.families.qwen.transformer_mlx import QwenImageDiTMLX as QwenMLX
        from backend.engine.runtime.mlx import MLXContext

        model = QwenImageTransformer(QwenImageConfig(), MLXContext())
        self.assertIsInstance(model._inner, QwenMLX)

    def test_transformer_dispatch_cuda(self) -> None:
        from backend.engine.config.model_configs import QwenImageConfig
        from backend.engine.families.qwen.transformer import QwenImageTransformer
        from backend.engine.families.qwen.transformer_cuda import QwenImageDiTCuda
        from backend.engine.runtime.cuda import CudaContext

        model = QwenImageTransformer(QwenImageConfig(), CudaContext("cpu"))
        self.assertIsInstance(model._inner, QwenImageDiTCuda)

    def test_transformer_unsupported_backend(self) -> None:
        from backend.engine.config.model_configs import QwenImageConfig
        from backend.engine.families.qwen.transformer import QwenImageTransformer

        ctx = SimpleNamespace(backend="unknown")
        with self.assertRaises(RuntimeError):
            QwenImageTransformer(QwenImageConfig(), ctx)

    def test_qwen_image_registry_declares_cuda_backend(self) -> None:
        entry = _load_default_registry_expanded()["models"]["qwen-image"]
        self.assertIn("cuda", entry.get("backends", []))
        self.assertIn("mlx", entry.get("backends", []))

    def test_qwen_text_encoder_weights_accepts_pre_remapped_encoder_keys(self) -> None:
        import mlx.core as mx

        from backend.engine.families.qwen.weights_mlx import apply_qwen_text_encoder_weights

        flat = {
            "encoder.embed_tokens.weight": mx.zeros((16, 8)),
            "encoder.layers.0.self_attn.q_proj.weight": mx.zeros((12, 8)),
        }
        nested = apply_qwen_text_encoder_weights(flat)
        enc = nested.get("encoder")
        self.assertIsInstance(enc, dict)
        self.assertIn("embed_tokens", enc)
        self.assertEqual(len(enc["layers"]), 1)
        self.assertIn("self_attn", enc["layers"][0])

    def test_qwen_edit_unpack_portrait_latent_grid(self) -> None:
        import mlx.core as mx

        from backend.engine.families.qwen.edit_util import (
            pack_qwen_latents_to_sequence,
            unpack_qwen_sequence_to_nchw,
        )
        from backend.engine.runtime.mlx import MLXContext

        ctx = MLXContext()
        width_px, height_px = 832, 1248
        h_lat, w_lat = height_px // 16, width_px // 16
        latents = ctx.seeded_randn((1, 64, h_lat, w_lat), 0)
        seq = pack_qwen_latents_to_sequence(ctx, latents)
        self.assertEqual(tuple(seq.shape), (1, h_lat * w_lat, 64))
        out = unpack_qwen_sequence_to_nchw(ctx, seq, height_px, width_px)
        self.assertEqual(tuple(out.shape), (1, 64, h_lat, w_lat))


class ErnieImageTransformerTests(unittest.TestCase):
    def test_transformer_dispatch_mlx(self) -> None:
        from backend.engine.config.model_configs import ErnieImageConfig
        from backend.engine.families.ernie_image.transformer import ErnieImageTransformer
        from backend.engine.families.ernie_image.transformer_mlx import ErnieImageDiTMLX as ErnieMLX
        from backend.engine.runtime.mlx import MLXContext

        model = ErnieImageTransformer(ErnieImageConfig(), MLXContext())
        self.assertIsInstance(model._inner, ErnieMLX)

    def test_transformer_dispatch_cuda_fail_loud(self) -> None:
        from types import SimpleNamespace

        from backend.engine.config.model_configs import ErnieImageConfig
        from backend.engine.families.ernie_image.transformer import ErnieImageTransformer

        with self.assertRaises(RuntimeError):
            ErnieImageTransformer(ErnieImageConfig(), SimpleNamespace(backend="cuda"))

    def test_remap_ernie_weights(self) -> None:
        from backend.engine.families.ernie_image.weights import remap_ernie_image_weights

        out = remap_ernie_image_weights(
            {
                "transformer.layers.0.self_attention.to_out.0.weight": object(),
                "model.adaLN_modulation.1.bias": object(),
                "time_proj.weight": object(),
            }
        )
        self.assertIn("layers.0.self_attention.to_out_0.weight", out)
        self.assertIn("adaLN_modulation.linear.bias", out)
        self.assertNotIn("time_proj.weight", out)

    def test_merge_config_from_bundle(self) -> None:
        import json
        import tempfile
        from pathlib import Path

        from backend.engine.config.model_configs import ErnieImageConfig, merge_ernie_image_config_from_bundle

        cfg = ErnieImageConfig()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "transformer").mkdir()
            (root / "transformer" / "config.json").write_text(
                json.dumps(
                    {
                        "hidden_size": 4096,
                        "num_attention_heads": 32,
                        "num_layers": 36,
                        "text_in_dim": 3072,
                        "rope_axes_dim": [32, 48, 48],
                        "qk_layernorm": True,
                    }
                ),
                encoding="utf-8",
            )
            merge_ernie_image_config_from_bundle(cfg, root)
        self.assertEqual(cfg.num_layers, 36)
        self.assertEqual(cfg.num_heads, 32)
        self.assertEqual(cfg.rope_axes_dim, (32, 48, 48))
        self.assertTrue(cfg.qk_norm)

    def test_forward_smoke(self) -> None:
        import mlx.core as mx

        from backend.engine.config.model_configs import ErnieImageConfig
        from backend.engine.families.ernie_image.transformer_mlx import ErnieImageDiTMLX
        from backend.engine.runtime.mlx import MLXContext

        cfg = ErnieImageConfig(num_layers=2)
        model = ErnieImageDiTMLX(cfg, MLXContext())
        h, w = 64, 64
        latents = mx.zeros((1, 128, h, w), dtype=mx.bfloat16)
        txt = mx.zeros((1, 8, 3072), dtype=mx.bfloat16)
        tl = mx.array([8], dtype=mx.int32)
        sigmas = mx.array([1.0, 0.875, 0.75, 0.625], dtype=mx.float32)
        out = model.forward(
            latents,
            0,
            txt_embeds=txt,
            text_lens=tl,
            sigmas=sigmas,
        )
        self.assertEqual(tuple(out.shape), (1, 128, h, w))

    def test_timestep_resolves_from_sigmas(self) -> None:
        import mlx.core as mx

        from backend.engine.config.model_configs import ErnieImageConfig
        from backend.engine.families.ernie_image.transformer_mlx import ErnieImageDiTMLX
        from backend.engine.runtime.mlx import MLXContext

        model = ErnieImageDiTMLX(ErnieImageConfig(), MLXContext())
        sigmas = mx.array([1.0, 0.5, 0.0], dtype=mx.float32)
        t = model._resolve_timestep_value(1, 0, sigmas)
        self.assertAlmostEqual(float(t.item()), 1000.0, places=4)
        t1 = model._resolve_timestep_value(1, 1, sigmas)
        self.assertAlmostEqual(float(t1.item()), 500.0, places=4)

    def test_ernie_scheduler_explicit_sigmas(self) -> None:
        from backend.engine.common.ops.schedulers import FlowMatchEulerScheduler
        from backend.engine.runtime.mlx import MLXContext

        ctx = MLXContext()
        sched = FlowMatchEulerScheduler(ctx=ctx)
        sigmas = [1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125]
        sched.set_timesteps(8, use_empirical_mu=False, sigmas=sigmas)
        self.assertAlmostEqual(float(sched.sigmas[0]), 1.0, places=4)
        self.assertAlmostEqual(float(sched.sigmas[7]), 0.125, places=4)
        self.assertAlmostEqual(float(sched.timesteps[0]), 1000.0, places=4)

    def test_registry_declares_mlx_only(self) -> None:
        entry = _load_default_registry_expanded()["models"]["ernie-image-turbo"]
        self.assertEqual(entry.get("backends"), ["mlx"])
        self.assertEqual(entry.get("family"), "ernie_image")
        self.assertFalse(entry["parameters"].get("lora_support", True))


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
        from backend.engine.families.flux1.transformer_mlx import Flux1DiTMLX as Flux1MLX
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
        from backend.engine.families.ltx.transformer_mlx import LTX23Transformer as LTXMLX
        from backend.engine.runtime.mlx import MLXContext

        model = LTXTransformer(LTXConfig(), MLXContext())
        self.assertIsInstance(model._inner, LTXMLX)

    def test_ltx_distilled_stage1_steps_floor(self) -> None:
        from backend.engine.families.ltx.generation_mlx import _resolve_distilled_stage1_steps

        logs: list[tuple[str, str]] = []

        def _on_log(level: str, msg: str) -> None:
            logs.append((level, msg))

        self.assertEqual(_resolve_distilled_stage1_steps(4, on_log=_on_log), 8)
        self.assertTrue(any(lvl == "warning" for lvl, _ in logs))
        self.assertEqual(_resolve_distilled_stage1_steps(10, on_log=_on_log), 10)

    def test_ltx_dev_guider_defaults_match_reference(self) -> None:
        from backend.engine.families.ltx.pipeline_math import (
            ltx2_schedule,
            ltx_dev_audio_guider_params,
            ltx_dev_video_guider_params,
        )

        video = ltx_dev_video_guider_params(3.0)
        self.assertEqual(video.cfg_scale, 3.0)
        self.assertEqual(video.stg_scale, 1.0)
        self.assertEqual(video.modality_scale, 3.0)
        self.assertEqual(video.stg_blocks, (28,))

        audio = ltx_dev_audio_guider_params()
        self.assertEqual(audio.cfg_scale, 7.0)

        sigmas = ltx2_schedule(28, 1024)
        self.assertEqual(len(sigmas), 29)
        self.assertEqual(sigmas[0], 1.0)
        self.assertEqual(sigmas[-1], 0.0)

    def test_ltx_denoise_loop_emits_progress(self) -> None:
        from unittest.mock import MagicMock

        import mlx.core as mx

        from backend.engine.families.ltx.generation_mlx import _denoise_loop
        from backend.engine.families.ltx.pipeline_math import LatentState
        from backend.engine.runtime.mlx import MLXContext

        ctx = MLXContext()
        model = MagicMock()
        latent = mx.zeros((1, 4, 128), dtype=mx.bfloat16)
        model.return_value = (latent, latent)
        state = LatentState(
            latent=latent,
            clean_latent=latent,
            denoise_mask=mx.ones((1, 4, 1), dtype=mx.bfloat16),
            positions=None,
        )
        progress_calls: list[tuple[int, int]] = []

        def _on_progress(p, step, total, _msg=None, _phase=None):
            progress_calls.append((int(step), int(total)))

        _denoise_loop(
            ctx,
            model,
            state,
            state,
            latent,
            latent,
            [1.0, 0.5, 0.0],
            on_progress=_on_progress,
            progress_step_offset=0,
            progress_total_steps=2,
        )
        self.assertEqual(progress_calls, [(1, 2), (2, 2)])

    def test_ltx_dispatch_cuda_fail_loud(self) -> None:
        from backend.engine.config.model_configs import LTXConfig
        from backend.engine.families.ltx.transformer import LTXTransformer
        from backend.engine.runtime.cuda import CudaContext

        with self.assertRaises(RuntimeError):
            LTXTransformer(LTXConfig(), CudaContext("cpu"))

    def test_wan_dispatch_cuda(self) -> None:
        from backend.engine.config.model_configs import WanConfig
        from backend.engine.families.wan.transformer import WanTransformer
        from backend.engine.families.wan.transformer_cuda import WanModelCUDA
        from backend.engine.runtime.cuda import CudaContext

        model = WanTransformer(WanConfig(), CudaContext("cpu"))
        self.assertIsInstance(model._inner, WanModelCUDA)

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
        from backend.engine.families.flux1.flux1_dual_mlx import Flux1TextEncoder as Flux1Impl
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

    def test_registry_action_maps_to_video_upscale(self) -> None:
        from backend.core.task_kinds import VIDEO_UPSCALE, task_kind_for_registry_action

        self.assertEqual(task_kind_for_registry_action("video", "upscale"), VIDEO_UPSCALE)


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
    def test_ace_step_dit_param_map_flattens_mlx_leaves(self) -> None:
        from backend.engine.config.model_configs import AceStepConfig
        from backend.engine.families.ace_step.transformer import AceStepTransformer
        from backend.engine.runtime.mlx import MLXContext

        cfg = AceStepConfig()
        ctx = MLXContext()
        dit = AceStepTransformer(
            ctx,
            hidden_size=cfg.hidden_size,
            intermediate_size=cfg.intermediate_size,
            num_hidden_layers=2,
            num_attention_heads=cfg.num_attention_heads,
            num_key_value_heads=cfg.num_key_value_heads,
            head_dim=cfg.head_dim,
            rms_norm_eps=cfg.rms_norm_eps,
            attention_bias=cfg.attention_bias,
            in_channels=cfg.in_channels,
            audio_acoustic_hidden_dim=cfg.audio_acoustic_hidden_dim,
            patch_size=cfg.patch_size,
            sliding_window=cfg.sliding_window,
            layer_types=list(cfg.layer_types)[:2],
            rope_theta=cfg.rope_theta,
            max_position_embeddings=cfg.max_position_embeddings,
        )
        pm = dit._param_map
        self.assertGreater(len(pm), 50)
        self.assertIn("proj_in.weight", pm)
        self.assertIn("layers.0.self_attn.q_proj.weight", pm)
        self.assertIn("time_embed.time_proj.weight", pm)
        self.assertNotIn("layers", pm)
        self.assertNotIn("proj_in", pm)

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
            dit = bundle / "acestep-v15-xl-turbo"
            dit.mkdir()
            (dit / "config.json").write_text(
                json.dumps({"is_turbo": True}),
                encoding="utf-8",
            )
            (dit / "model.safetensors").write_bytes(b"")

            self.assertTrue(resolve_bundle_is_turbo(bundle, dit_subdir="acestep-v15-xl-turbo"))

            req = AudioGenerationRequest(
                model="ace-step-xl-turbo",
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

    def test_prepare_music_request_requires_vocal_lyrics(self) -> None:
        from backend.core.contracts import AudioGenerationRequest
        from backend.engine.config.model_configs import AceStepConfig
        from backend.engine.families.ace_step.generation import prepare_music_request

        req = AudioGenerationRequest(
            model="ace-step-xl-sft",
            prompt="upbeat pop song about summer",
            lyrics="",
            instrumental=False,
        )
        with __import__("tempfile").TemporaryDirectory() as tmp:
            from pathlib import Path

            bundle = Path(tmp)
            dit = bundle / "acestep-v15-xl-sft"
            dit.mkdir()
            (dit / "config.json").write_text("{}", encoding="utf-8")
            (dit / "model.safetensors").write_bytes(b"")
            with self.assertRaises(RuntimeError):
                prepare_music_request(req, AceStepConfig(), bundle, backend="mlx")

    def test_resolve_lm_expansion_state(self) -> None:
        from backend.core.contracts import AudioGenerationRequest
        from backend.engine.families.ace_step.generation import (
            prepare_music_request,
            resolve_lm_expansion_state,
        )
        from backend.engine.config.model_configs import AceStepConfig

        req = AudioGenerationRequest(
            model="ace-step-xl-sft",
            prompt="pop",
            lyrics="[verse]\nhello world",
        )
        use_lm, reason = resolve_lm_expansion_state(req, lm_enabled=True)
        self.assertTrue(use_lm)
        self.assertEqual(reason, "auto_format")

        off_req = AudioGenerationRequest(
            model="ace-step-xl-sft",
            prompt="pop",
            lyrics="[verse]\nhello",
            lm_expansion="off",
        )
        use_lm, reason = resolve_lm_expansion_state(off_req, lm_enabled=True)
        self.assertFalse(use_lm)
        self.assertEqual(reason, "override_off")

        with self.assertRaises(RuntimeError):
            resolve_lm_expansion_state(
                AudioGenerationRequest(
                    model="ace-step-xl-sft",
                    prompt="pop",
                    lyrics="[verse]\nhello",
                    lm_expansion="inspiration",
                ),
                lm_enabled=True,
            )

        with __import__("tempfile").TemporaryDirectory() as tmp:
            from pathlib import Path

            bundle = Path(tmp)
            dit = bundle / "acestep-v15-xl-sft"
            dit.mkdir()
            (dit / "config.json").write_text("{}", encoding="utf-8")
            (dit / "model.safetensors").write_bytes(b"")
            prepared = prepare_music_request(req, AceStepConfig(), bundle, backend="mlx")
            self.assertTrue(prepared.lm_enabled)
            self.assertEqual(prepared.lyrics, "[verse]\nhello world")

    def test_resource_policy_clamps_duration(self) -> None:
        from backend.engine.families.ace_step.quality.resource_policy import (
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

    def test_constrained_lm_finds_extended_audio_code_vocab(self) -> None:
        from unittest.mock import MagicMock

        from backend.engine.families.ace_step.lm.constrained_lm import (
            MetadataConstrainedLogitsProcessor,
        )
        from backend.engine.families.ace_step.lm.lm_constants import MAX_AUDIO_CODE

        gv = {f"<|audio_code_{i}|>": 200_000 + i for i in (0, 1, MAX_AUDIO_CODE)}
        gv["hello"] = 5
        tok = MagicMock()
        tok.vocab_size = 151643
        tok._tokenizer = tok
        tok.get_vocab.return_value = gv
        tok.encode = lambda s, add_special_tokens=False: [1]
        tok.decode = lambda ids: ""
        tok.eos_token_id = 0

        proc = MetadataConstrainedLogitsProcessor(tokenizer=tok, enabled=False, debug=False)
        self.assertGreaterEqual(proc.vocab_size, 200_000 + MAX_AUDIO_CODE)
        self.assertEqual(len(proc.audio_code_token_ids), 3)

    def test_constrained_processor_vocab_size_matches_lm_config(self) -> None:
        import json
        from pathlib import Path

        pointer = (
            Path(__file__).resolve().parents[1]
            / "default_config"
            / "workspace.pointer.json"
        )
        if not pointer.is_file():
            self.skipTest("workspace.pointer.json missing")
        ws = json.loads(pointer.read_text(encoding="utf-8")).get("custom_workspace_dir")
        lm = Path(ws) / "models" / "Audio" / "acestep-v15-xl-sft" / "acestep-5Hz-lm-1.7B"
        if not (lm / "config.json").is_file():
            self.skipTest("ace-step 5Hz LM bundle not present")

        from mlx_lm.utils import load_tokenizer

        from backend.engine.families.ace_step.lm.constrained_generate import (
            create_constrained_processor,
        )

        cfg_vocab = json.loads((lm / "config.json").read_text(encoding="utf-8"))["vocab_size"]
        proc = create_constrained_processor(load_tokenizer(str(lm)))
        self.assertEqual(proc.vocab_size, cfg_vocab)
        self.assertEqual(len(proc.audio_code_token_ids), 64000)

    def test_align_codec_latents_pads_shorter_hints(self) -> None:
        import mlx.core as mx

        from backend.engine.families.ace_step.audio.audio_codec_mlx import (
            _align_codec_latents_to_target,
        )

        hints = mx.zeros((1, 370, 64))
        pad_from = mx.ones((1, 374, 64)) * 0.25
        out = _align_codec_latents_to_target(hints, 374, pad_from=pad_from)
        self.assertEqual(tuple(out.shape), (1, 374, 64))
        self.assertAlmostEqual(float(mx.mean(out[:, 370:, :])), 0.25, places=5)

    def test_quality_assessment_flags_hum(self) -> None:
        from backend.engine.families.ace_step.quality.quality_score import assess_generation_quality

        q = assess_generation_quality(hum_ratio=0.4, mains_acf=0.5, latent_cos=0.5, latent_diff=0.2)
        self.assertLess(q.score, 60.0)
        self.assertEqual(q.grade, "poor")

    def test_audio_codec_fsq_roundtrip(self) -> None:
        import mlx.core as mx

        from backend.engine.families.ace_step.audio.audio_codec_mlx import _MlxResidualFSQ

        levels = [8, 8, 8, 5, 5, 5]
        rf = _MlxResidualFSQ(dim=64, levels=levels, num_quantizers=1)
        x = mx.random.normal((1, 4, 64))
        quantized, indices = rf(x)
        restored = rf.get_output_from_indices(indices)
        diff = float(mx.max(mx.abs(quantized - restored)))
        self.assertLess(diff, 1e-5)

    def test_audio_codec_indices_cast_float_to_int(self) -> None:
        import mlx.core as mx

        from backend.engine.families.ace_step.audio.audio_codec_mlx import _MlxResidualFSQ

        levels = [8, 8, 8, 5, 5, 5]
        rf = _MlxResidualFSQ(dim=64, levels=levels, num_quantizers=1)
        float_idx = mx.array([[[3.0]]], dtype=mx.float32)
        out = rf.get_output_from_indices(float_idx)
        mx.eval(out)
        self.assertEqual(out.shape, (1, 1, 64))

    def test_parse_audio_code_indices(self) -> None:
        from backend.engine.families.ace_step.lm.lm_format import parse_audio_code_indices

        text = "<|audio_code_12|><|audio_code_345|>"
        self.assertEqual(parse_audio_code_indices(text), (12, 345))

    def test_instrumental_lyrics_normalization(self) -> None:
        from backend.engine.families.ace_step.generation import finalize_lyrics_for_inference
        from backend.engine.families.ace_step.lm.lm_format import (
            extract_lm_generated_lyrics,
            is_instrumental_lyrics,
            normalize_lyrics_body,
        )

        self.assertTrue(is_instrumental_lyrics("# Lyric [Instrumental]"))
        self.assertEqual(normalize_lyrics_body("# Lyric [Instrumental]"), "[Instrumental]")
        self.assertEqual(
            extract_lm_generated_lyrics("[Verse 1]\nhello", prefilled_think_end=True),
            "[Verse 1]\nhello",
        )
        with self.assertRaises(RuntimeError):
            finalize_lyrics_for_inference(
                "# Lyric [Instrumental]",
                instrumental=False,
                lm_expanded=True,
            )

    def test_format_sample_preserves_user_lyrics(self) -> None:
        from backend.engine.families.ace_step.lm.lm_format import build_lm_format_result

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
        self.assertEqual(result.language, "zh")

    def test_format_sample_zh_lyrics_not_overridden_by_lm_en(self) -> None:
        from backend.engine.families.ace_step.lm.lm_format import build_lm_format_result

        user = "[Verse 1]\n月光洒在窗前\n[Chorus]\n轻轻唱"
        meta = {
            "caption": "English pop caption from LM",
            "lyrics_tail": "[Verse 1]\nSing in English",
            "language": "en",
        }
        result = build_lm_format_result(
            meta,
            caption_in="流行歌",
            lyrics_in=user,
            duration=30,
            bpm=100,
            keyscale="C major",
            timesignature="4",
            language="zh",
            preserve_user_lyrics=True,
        )
        self.assertIn("月光", result.lyrics)

    def test_resolve_vocal_language_from_lyrics(self) -> None:
        from backend.engine.families.ace_step.generation import resolve_vocal_language

        self.assertEqual(resolve_vocal_language("一首欢快的华语流行歌曲", ""), "zh")
        self.assertEqual(resolve_vocal_language("hello world", "en"), "en")

    def test_format_metadata_as_cot_and_codes_prompt(self) -> None:
        from backend.engine.families.ace_step.lm.lm_format import (
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
        from backend.engine.families.ace_step.lm.constrained_generate import (
            ConstrainedGenerationConfig,
            create_constrained_processor,
            resolve_hf_tokenizer,
        )
        from backend.engine.families.ace_step.lm.lm_format import (
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
        from transformers import AutoTokenizer

        from backend.engine.families.ace_step.lm.constrained_generate import resolve_hf_tokenizer
        from backend.engine.families.ace_step.lm.constrained_generate_mlx import mlx_logits_to_numpy
        from mlx_lm.tokenizer_utils import TokenizerWrapper

        try:
            inner = AutoTokenizer.from_pretrained("gpt2", local_files_only=True)
        except OSError:
            self.skipTest("gpt2 tokenizer not cached locally (offline test)")
        wrapped = TokenizerWrapper(inner)
        self.assertIs(resolve_hf_tokenizer(wrapped), inner)
        bf = mx.array([[[1.0, 2.0]]], dtype=mx.bfloat16)
        mx.eval(bf)
        arr = mlx_logits_to_numpy(bf)
        self.assertEqual(arr.dtype, np.float32)
        self.assertEqual(float(arr[0, 0, 0]), 1.0)

    def test_lyrics_alignment_structure_estimate(self) -> None:
        from backend.engine.families.ace_step.vocals.lyrics_alignment import (
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

    def test_audio_prepare_request_registry(self) -> None:
        from backend.engine._transformer_registry import get_audio_prepare_request

        ace_fn = get_audio_prepare_request("ace_step")
        dr_fn = get_audio_prepare_request("diffrhythm")
        self.assertEqual(ace_fn.__module__, "backend.engine.families.ace_step.generation")
        self.assertEqual(dr_fn.__module__, "backend.engine.families.diffrhythm.generation")
        with self.assertRaises(RuntimeError):
            get_audio_prepare_request("unknown_audio_family")

    def test_ace_step_dit_subdir_for_model(self) -> None:
        from backend.engine.families.ace_step.weights import ace_step_dit_subdir_for_model

        self.assertEqual(ace_step_dit_subdir_for_model("ace-step-xl-sft"), "acestep-v15-xl-sft")
        self.assertEqual(ace_step_dit_subdir_for_model("ace-step-xl-sft:int8"), "acestep-v15-xl-sft")
        self.assertIsNone(ace_step_dit_subdir_for_model("unknown-model"))

    def test_ace_step_lora_base_compatible(self) -> None:
        from backend.engine.families.ace_step.weights import ace_step_lora_base_compatible

        self.assertTrue(ace_step_lora_base_compatible("ace-step-xl-sft", "ace-step-xl-turbo"))
        self.assertFalse(ace_step_lora_base_compatible("ace-step-xl-sft", "flux1-dev"))

    def test_ace_step_registry_bundle_repos(self) -> None:
        from backend.core.bundle_repos import bundle_repos_from_version

        models = _load_default_registry_expanded().get("models") or {}
        ver = models["ace-step-xl-sft"]["versions"]["xl-sft"]
        repos = bundle_repos_from_version(ver)
        self.assertEqual(len(repos), 2)
        self.assertEqual(repos[0]["repo_id"], "ACE-Step/Ace-Step1.5")
        self.assertEqual(repos[0]["local_path"], "models/Audio/acestep-v15-xl-sft")
        self.assertEqual(repos[1]["repo_id"], "ACE-Step/acestep-v15-xl-sft")
        self.assertEqual(
            repos[1]["local_path"],
            "models/Audio/acestep-v15-xl-sft/acestep-v15-xl-sft",
        )

    def test_resolve_dit_bundle_requires_exact_model_subdir(self) -> None:
        import tempfile
        from pathlib import Path

        from backend.engine.families.ace_step.generation import resolve_dit_bundle

        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp)
            turbo = bundle / "acestep-v15-turbo"
            turbo.mkdir()
            (turbo / "model.safetensors").write_bytes(b"")
            with self.assertRaises(RuntimeError):
                resolve_dit_bundle(bundle, dit_subdir="acestep-v15-xl-sft")

            xl_sft = bundle / "acestep-v15-xl-sft"
            xl_sft.mkdir()
            (xl_sft / "model.safetensors.index.json").write_text("{}", encoding="utf-8")
            resolved = resolve_dit_bundle(bundle, dit_subdir="acestep-v15-xl-sft")
            self.assertEqual(resolved, xl_sft)

    def test_remap_ace_step_lora_keys_peft(self) -> None:
        import numpy as np

        from backend.engine.families.ace_step.weights import remap_ace_step_lora_keys

        weights = {
            "base_model.model.layers.0.self_attn.q_proj.lora_A.weight": np.zeros((4, 8), dtype=np.float32),
            "base_model.model.layers.0.self_attn.q_proj.lora_B.weight": np.zeros((16, 4), dtype=np.float32),
            "base_model.model.layers.0.self_attn.q_proj.alpha": np.array(64.0, dtype=np.float32),
        }
        groups = remap_ace_step_lora_keys(weights)
        self.assertIn("layers.0.self_attn.q_proj", groups)
        down, up, alpha = groups["layers.0.self_attn.q_proj"]
        self.assertEqual(tuple(down.shape), (4, 8))
        self.assertEqual(tuple(up.shape), (16, 4))
        self.assertEqual(alpha, 64.0)

    def test_merge_audio_lora_registry(self) -> None:
        from backend.engine._transformer_registry import get_audio_lora_merge

        merge_fn = get_audio_lora_merge("ace_step")
        self.assertIsNotNone(merge_fn)
        self.assertTrue(callable(merge_fn))
        self.assertIsNone(get_audio_lora_merge("diffrhythm"))


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
        from backend.engine.contracts.pipeline_registry import local_bundle_root

        class _Entry:
            raw = {
                "versions": {
                    "default": {
                        "bundle_repos": [
                            {
                                "repo_id": "AceStep/AceStepTokenizer",
                                "local_path": "models/Audio/ace-step-xl-sft",
                            },
                        ],
                    },
                },
            }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle = root / "models/Audio/ace-step-xl-sft"
            bundle.mkdir(parents=True)
            resolved = local_bundle_root(root, _Entry(), "default")
            self.assertEqual(resolved, bundle.resolve())
            missing = local_bundle_root(root, _Entry(), "missing")
            self.assertIsNone(missing)


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

        from backend.engine.common.ops.cfg_batch import broadcast_batch, merge_cfg_forward_kwargs

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

    def test_wan_predict_noise_cfg(self) -> None:
        from backend.engine.config.model_configs import WanConfig
        from backend.engine.families.wan.transformer import WanTransformer
        from backend.engine.families.wan.transformer_mlx import WanModelMLX
        from backend.engine.runtime.mlx import MLXContext

        self.assertTrue(hasattr(WanModelMLX, "predict_noise_cfg"))
        model = WanTransformer(WanConfig(), MLXContext(), num_frames=17)
        self.assertTrue(hasattr(model, "predict_noise_cfg"))
        self.assertTrue(callable(getattr(model, "predict_noise_cfg")))

    def test_wan_mlx_perf_hooks(self) -> None:
        from backend.engine.config.model_configs import WanConfig
        from backend.engine.families.wan.transformer_mlx import WanModelMLX

        cfg = WanConfig()
        self.assertFalse(cfg.use_mlx_compile)
        self.assertFalse(cfg.vae_spatial_tiling)
        self.assertTrue(hasattr(WanModelMLX, "after_load_weights"))
        self.assertTrue(hasattr(WanModelMLX, "invalidate_text_cache"))

    def test_remap_wan_weights_head_head_linear(self) -> None:
        from backend.engine.families.wan.weights import remap_wan_weights

        out = remap_wan_weights(
            {
                "head.head.weight": "w",
                "head.head.bias": "b",
                "head.modulation": "m",
                "blocks.0.self_attn.q.weight": "q",
            }
        )
        self.assertEqual(out["head.1.weight"], "w")
        self.assertEqual(out["head.1.bias"], "b")
        self.assertEqual(out["head.modulation"], "m")
        self.assertEqual(out["blocks.0.self_attn.q.weight"], "q")

    def test_wan_t2v_skips_expand_timesteps(self) -> None:
        """T2V must not set wan_expand_timesteps (scalar adaLN); I2V requires it."""
        import mlx.core as mx

        from backend.engine.config.model_configs import WanConfig
        from backend.engine.families.wan.transformer import WanTransformer
        from backend.engine.runtime.mlx import MLXContext

        ctx = MLXContext()
        model = WanTransformer(WanConfig(), ctx, num_frames=17)
        latents = mx.zeros((1, 48, 5, 44, 30), dtype=mx.float32)
        timesteps = mx.array([999.0], dtype=mx.float32)

        _, t2v_cond = model.before_denoise(latents, timesteps, None)
        self.assertNotIn("wan_expand_timesteps", t2v_cond)

        i2v_cond_in = {
            "wan_i2v": True,
            "wan_cond_latent": mx.zeros((48, 5, 44, 30), dtype=mx.float32),
            "wan_i2v_mask": mx.ones((48, 5, 44, 30), dtype=mx.float32),
        }
        _, i2v_cond = model.before_denoise(latents, timesteps, None, **i2v_cond_in)
        self.assertTrue(i2v_cond.get("wan_expand_timesteps"))
        self.assertIsNotNone(i2v_cond.get("wan_i2v_mask"))

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
        from tests.benchmark.registry_utils import resolve_benchmark_data_root

        wan_bundle = resolve_benchmark_data_root() / "models/Video/wan-2.2-ti2v-5b-original"
        if not wan_bundle.is_dir():
            self.skipTest("Wan bundle not installed")

        bundle = wan_bundle
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
        import numpy as np

        from backend.engine.common.ops.schedulers import WanFlowUniPCScheduler
        from backend.engine.runtime.mlx import MLXContext

        ctx = MLXContext()
        sched = WanFlowUniPCScheduler(1000, ctx=ctx, solver_order=2)
        sched.set_timesteps(4, shift=5.0)
        ts = np.array(sched.timesteps.tolist(), dtype=np.float32)
        self.assertTrue(np.all(np.abs(ts - np.round(ts)) < 1e-5))
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

        from backend.engine.common.ops.attention import wan_attention
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

    def test_wan_vae_encode_decode_roundtrip(self) -> None:
        """MLX Wan 2.2 VAE must decode with per-frame causal cache (not full-volume)."""
        from pathlib import Path

        import mlx.core as mx
        import numpy as np

        from backend.engine.families.wan.vae_mlx import (
            _vae_cache,
            decode_wan_vae_latents,
            encode_wan_vae_image,
        )
        from backend.engine.runtime.mlx import MLXContext
        from tests.benchmark.registry_utils import resolve_benchmark_data_root

        bundle = resolve_benchmark_data_root() / "models/Video/wan-2.2-ti2v-5b-original"
        if not bundle.is_dir():
            self.skipTest("Wan TI2V bundle not installed for VAE roundtrip test")

        _vae_cache.clear()
        ctx = MLXContext()
        rng = np.random.default_rng(0)
        chw = mx.array(rng.standard_normal((3, 64, 64), dtype=np.float32) * 0.25)
        z = encode_wan_vae_image(ctx, chw, bundle)
        z_vol = mx.concatenate(
            [
                z,
                mx.zeros(
                    (1, int(z.shape[1]), 2, int(z.shape[3]), int(z.shape[4])),
                    dtype=z.dtype,
                ),
            ],
            axis=2,
        )
        pixels = decode_wan_vae_latents(ctx, z_vol, bundle)
        mx.eval(pixels)
        self.assertEqual(int(pixels.shape[0]), 1)
        self.assertEqual(int(pixels.shape[1]), 3)
        self.assertGreater(int(pixels.shape[2]), 1)
        self.assertLess(float(mx.max(mx.abs(pixels))), 1.01)
        self.assertGreater(float(mx.std(pixels)), 0.01)
        # Frame 1+ must not collapse (Resample used to mis-flatten b/t).
        self.assertGreater(float(mx.std(pixels[:, :, 1])), 0.05)

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
        models = _load_default_registry_expanded().get("models") or {}
        self.assertIn("hunyuan-video-1.5-480p-t2v", models)
        self.assertEqual(models["hunyuan-video-1.5-480p-t2v"]["family"], "hunyuan")
        self.assertIn("hunyuan-video-1.5-i2v-step-distill", models)
        self.assertIn("hunyuan-video-1.5-t2v-step-distill", models)
        for mid in (
            "hunyuan-video-1.5-i2v-step-distill",
            "hunyuan-video-1.5-t2v-step-distill",
        ):
            distill = models[mid]["parameters"]
            self.assertFalse(distill.get("supports_guidance"), msg=mid)
            self.assertTrue(distill.get("step_distill"), msg=mid)
            self.assertFalse(distill.get("negative_prompt_support"), msg=mid)
            self.assertNotIn("guide_scale", distill, msg=mid)
        self.assertIn("animate", models["hunyuan-video-1.5-i2v-step-distill"].get("actions") or {})
        self.assertIn("create", models["hunyuan-video-1.5-t2v-step-distill"].get("actions") or {})

    def test_hunyuan_sr_scheduler_sigmas(self) -> None:
        import mlx.core as mx

        from backend.engine.common.ops.schedulers import FlowMatchEulerScheduler
        from backend.engine.families.hunyuan.sr_mlx import configure_hunyuan_step_distill_timesteps

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
        timesteps = configure_hunyuan_step_distill_timesteps(ctx, sched, 6)
        self.assertEqual(int(timesteps.shape[0]), 6)

    def test_hunyuan_registry_modelscope_repos(self) -> None:
        from backend.core.bundle_repos import bundle_repos_from_version

        models = _load_default_registry_expanded().get("models") or {}
        for mid in (
            "hunyuan-video-1.5-480p-t2v",
            "hunyuan-video-1.5-480p-i2v",
            "hunyuan-video-1.5-i2v-step-distill",
            "hunyuan-video-1.5-t2v-step-distill",
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
        from backend.engine.common.codecs.text_encoders.torch_device import resolve_torch_inference_device

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

        from backend.engine.common.bundle.hf_tokenizer_json import HFTokenizerJson, render_qwen_chat_messages

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

        from backend.engine.common.bundle.hf_tokenizer_json import HFTokenizerJson

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

        from backend.engine.runtime.mlx_dtype import cast_floating_mx_tree

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
        sr = _load_default_registry_expanded()["models"]["hunyuan-video-1.5-1080p-sr"]
        self.assertTrue(sr["parameters"].get("vae_spatial_tiling"))


class InstallHooksTests(unittest.TestCase):
    def test_install_hooks_from_version_parses_objects(self) -> None:
        from backend.core.install_hooks import install_hooks_from_version

        ver = {
            "install_hooks": [
                {"type": "ace_step_post_download", "dtype": "bfloat16"},
                "other_hook",
            ]
        }
        hooks = install_hooks_from_version(ver)
        self.assertEqual(len(hooks), 2)
        self.assertEqual(hooks[0]["type"], "ace_step_post_download")
        self.assertEqual(hooks[0]["dtype"], "bfloat16")
        self.assertEqual(hooks[1]["type"], "other_hook")

    def test_install_hooks_from_version_empty(self) -> None:
        from backend.core.install_hooks import install_hooks_from_version

        self.assertEqual(install_hooks_from_version(None), [])
        self.assertEqual(install_hooks_from_version({}), [])

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
    def test_eval_prompt_pack_has_p1(self) -> None:
        from tests.benchmark.eval_cases import load_prompt_pack

        pack = load_prompt_pack()
        ids = {p["id"] for p in pack["create"]}
        self.assertIn("P1", ids)

    def test_z_image_turbo_enable_thinking_default(self) -> None:
        turbo = _load_default_registry_expanded()["models"]["z-image-turbo"]
        self.assertTrue(
            turbo["parameters"].get("enable_thinking"),
            "z-image-turbo must keep enable_thinking=true for tokenizer chat-template semantics",
        )

    def test_z_image_turbo_lora_support(self) -> None:
        turbo = _load_default_registry_expanded()["models"]["z-image-turbo"]
        self.assertTrue(turbo["parameters"].get("lora_support", False))

    def test_z_image_lora_scope_base_turbo_compatible(self) -> None:
        from backend.engine.families.z_image.weights import (
            z_image_lora_base_compatible,
            z_image_lora_scope_key,
        )

        self.assertEqual(z_image_lora_scope_key("z-image"), "z_image")
        self.assertEqual(z_image_lora_scope_key("z-image-turbo"), "z_image")
        self.assertTrue(z_image_lora_base_compatible("z-image-turbo", "z-image"))
        self.assertTrue(z_image_lora_base_compatible("z-image", "z-image-turbo"))
        self.assertFalse(z_image_lora_base_compatible("z-image-turbo", "flux1-dev"))


class MemoryPolicyTests(unittest.TestCase):
    def test_clamp_mlx_memory_limit_gb(self) -> None:
        from backend.engine.memory_policy import clamp_mlx_memory_limit_gb

        self.assertEqual(clamp_mlx_memory_limit_gb(120), 120)
        self.assertEqual(clamp_mlx_memory_limit_gb(8), 16)
        self.assertEqual(clamp_mlx_memory_limit_gb(999), 512)
        self.assertEqual(clamp_mlx_memory_limit_gb("bad", default=64), 64)

    def test_resolve_lora_worker_memory_gb(self) -> None:
        from backend.core.interfaces import AppSettings
        from backend.engine.memory_policy import resolve_lora_worker_memory_gb

        settings = AppSettings(mlx_memory_limit=120)
        self.assertEqual(resolve_lora_worker_memory_gb(settings), 104)

    def test_classify_sigkill_as_oom(self) -> None:
        from backend.observability.error_codes import ErrorCode, classify_exception_message

        msg = "LoRA worker exited without result (code=-9). stderr='' stdout=''"
        self.assertEqual(classify_exception_message(msg), ErrorCode.OOM)

    def test_model_cache_single_slot(self) -> None:
        from backend.engine.cache import ModelCache

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
        from backend.engine.cache import ModelCache

        cache = ModelCache(lambda: 120.0, max_entries=2, ttl_minutes=10)
        cache.set_ttl_minutes(45)
        self.assertEqual(cache.stats["ttl_minutes"], 45)
        cache.put("x", {"v": 1}, 1.0)
        cache.evict("x")
        self.assertIsNone(cache.get("x"))

    def test_model_cache_purge_idle_after_ttl(self) -> None:
        from datetime import datetime, timedelta

        from backend.engine.cache import ModelCache

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

    def test_scan_components_wan_flat_bundle_layout(self) -> None:
        from backend.core.bundle_manifest import assert_bundle_ready_for_family, scan_components

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "diffusion_pytorch_model-00001-of-00003.safetensors").write_bytes(b"x")
            (root / "models_t5_umt5-xxl-enc-bf16.pth").write_bytes(b"y")
            (root / "Wan2.2_VAE.pth").write_bytes(b"z")
            tok_dir = root / "google" / "umt5-xxl"
            tok_dir.mkdir(parents=True)
            (tok_dir / "tokenizer.json").write_text("{}", encoding="utf-8")

            components = scan_components(root)
            self.assertIn("transformer", components)
            self.assertIn("text_encoder", components)
            self.assertIn("vae", components)
            self.assertIn("tokenizer", components)
            assert_bundle_ready_for_family(root, family="wan", model_id="wan-2.2-ti2v-5b")

    def test_scan_components_acestep_nested_dit_layout(self) -> None:
        from backend.core.bundle_manifest import assert_bundle_ready_for_family, scan_components

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dit = root / "acestep-v15-turbo"
            dit.mkdir(parents=True)
            (dit / "model.safetensors").write_bytes(b"x")
            (dit / "config.json").write_text("{}", encoding="utf-8")
            vae = root / "vae"
            vae.mkdir(parents=True)
            (vae / "diffusion_pytorch_model.safetensors").write_bytes(b"y")
            (root / "acestep-5Hz-lm-1.7B").mkdir(parents=True)
            (root / "acestep-5Hz-lm-1.7B" / "model.safetensors").write_bytes(b"z")

            components = scan_components(root)
            self.assertIn("transformer", components)
            self.assertIn("vae", components)
            self.assertNotIn(
                "acestep-5Hz-lm-1.7B/model.safetensors",
                components.get("transformer", []),
            )
            assert_bundle_ready_for_family(root, family="ace_step", model_id="ace-step-xl-sft")

    def test_assert_bundle_ready_esrgan_single_safetensors(self) -> None:
        from backend.core.bundle_manifest import assert_bundle_ready_for_family, scan_components

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "model.safetensors").write_bytes(b"x")
            (root / "config.json").write_text("{}", encoding="utf-8")

            components = scan_components(root)
            self.assertIn("transformer", components)
            assert_bundle_ready_for_family(root, family="esrgan", model_id="real-esrgan-x4plus")

    def test_t5_encoder_bundle_paths_flux_layout(self) -> None:
        from backend.engine.common.bundle.layout import t5_encoder_bundle_paths

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
        from backend.engine.common.bundle.layout import assert_media_bundle_ready

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

    def test_lora_registry_category_skips_full_bundle_contract(self) -> None:
        from backend.core.bundle_manifest import (
            is_registry_controlnet_category,
            is_registry_lora_category,
            skips_full_family_bundle_contract,
        )
        from backend.core.interfaces import ModelConfig
        from backend.services.services import SettingsService

        self.assertTrue(is_registry_lora_category("loras"))
        self.assertFalse(is_registry_lora_category("base_models"))
        self.assertTrue(is_registry_controlnet_category("controlnets"))
        self.assertFalse(is_registry_controlnet_category("base_models"))
        self.assertTrue(skips_full_family_bundle_contract("loras"))
        self.assertTrue(skips_full_family_bundle_contract("controlnets"))
        self.assertFalse(skips_full_family_bundle_contract("base_models"))

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lora_dir = root / "models" / "Lora" / "test-lora-fp16"
            lora_dir.mkdir(parents=True)
            (lora_dir / "adapter.safetensors").write_bytes(b"x")

            config = ModelConfig(
                engine="danqing-image",
                type="lora",
                name={"zh": "Test LoRA", "en": "Test LoRA"},
                category="loras",
                family="qwen_image",
                versions={
                    "fp16": {
                        "local_path": "models/Lora/test-lora-fp16",
                    }
                },
            )

            class _Resolver:
                def resolve_registry_local_path(self, local_path: str) -> Path:
                    return (root / local_path).resolve()

            svc = SettingsService.__new__(SettingsService)
            svc._path_resolver = _Resolver()
            model_dir = svc._resolve_registry_version_bundle_dir(
                "test-lora", "fp16", config.versions["fp16"]
            )
            self.assertTrue(model_dir.exists())
            self.assertTrue(SettingsService._path_has_bundle_weights(model_dir))

            category = config.category or ""
            family = config.family or ""
            components = None
            if (
                SettingsService._path_has_bundle_weights(model_dir)
                and family
                and not skips_full_family_bundle_contract(category)
            ):
                from backend.core.bundle_manifest import bundle_component_status

                components = bundle_component_status(model_dir, family=family)
            self.assertIsNone(components)

    def test_controlnet_registry_category_skips_full_bundle_contract(self) -> None:
        from backend.core.bundle_manifest import skips_full_family_bundle_contract
        from backend.core.interfaces import ModelConfig
        from backend.services.services import SettingsService

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cn_dir = root / "models" / "ControlNet" / "test-controlnet-8steps"
            cn_dir.mkdir(parents=True)
            (cn_dir / "controlnet.safetensors").write_bytes(b"x")

            config = ModelConfig(
                engine="danqing-image",
                type="controlnet",
                name={"zh": "Test ControlNet", "en": "Test ControlNet"},
                category="controlnets",
                family="z_image",
                versions={
                    "8steps": {
                        "local_path": "models/ControlNet/test-controlnet-8steps",
                    }
                },
            )

            class _Resolver:
                def resolve_registry_local_path(self, local_path: str) -> Path:
                    return (root / local_path).resolve()

            svc = SettingsService.__new__(SettingsService)
            svc._path_resolver = _Resolver()
            model_dir = svc._resolve_registry_version_bundle_dir(
                "test-controlnet", "8steps", config.versions["8steps"]
            )
            self.assertTrue(model_dir.exists())
            self.assertTrue(SettingsService._path_has_bundle_weights(model_dir))

            category = config.category or ""
            family = config.family or ""
            components = None
            if (
                SettingsService._path_has_bundle_weights(model_dir)
                and family
                and not skips_full_family_bundle_contract(category)
            ):
                from backend.core.bundle_manifest import bundle_component_status

                components = bundle_component_status(model_dir, family=family)
            self.assertIsNone(components)


class DiffRhythmDecodeTests(unittest.TestCase):
    def test_pytorch_bin_bfloat16_storage(self) -> None:
        import numpy as np

        from backend.engine.common.bundle.pytorch_bin_numpy import _storage_array

        raw = (np.float32(1.5).view(np.uint32) >> 16).astype(np.uint16).tobytes()
        arr = _storage_array(raw, type("BFloat16Storage", (), {"__name__": "BFloat16Storage"})(), 1)
        self.assertEqual(arr.shape, (1,))
        self.assertAlmostEqual(float(arr[0]), 1.5, places=5)

    def test_pytorch_bin_numpy_loader_decoder(self) -> None:
        import json

        pointer = (
            Path(__file__).resolve().parents[1]
            / "default_config"
            / "workspace.pointer.json"
        )
        if not pointer.is_file():
            self.skipTest("workspace.pointer.json missing")
        ws = json.loads(pointer.read_text(encoding="utf-8")).get("custom_workspace_dir")
        ckpt = Path(ws) / "models" / "Audio" / "diffrhythm-v2" / "decoder.bin"
        if not ckpt.is_file():
            self.skipTest("diffrhythm-v2 decoder.bin not installed")

        from backend.engine.common.bundle.pytorch_bin_numpy import load_pytorch_bin

        obj = load_pytorch_bin(ckpt)
        self.assertIn("generator", obj)
        gen = obj["generator"]
        self.assertGreater(len(gen), 100)
        sample = next(iter(gen.values()))
        import numpy as np

        self.assertIsInstance(sample, np.ndarray)

    def test_alias_free_upsample_matches_torch(self) -> None:
        import mlx.core as mx
        import numpy as np

        try:
            import torch
            import torch.nn.functional as F
        except ImportError:
            self.skipTest("torch not installed (benchmark/CUDA venv only)")

        from backend.engine.families.diffrhythm.vae_mlx import UpSample1d

        rng = np.random.default_rng(7)
        B, C, T = 1, 16, 24
        ratio = 2
        K = 12
        pad = K // ratio - 1
        pad_left = pad * ratio + (K - ratio) // 2
        pad_right = pad * ratio + (K - ratio + 1) // 2

        filt = rng.standard_normal((1, 1, K), dtype=np.float32)
        filt /= np.abs(filt).sum()
        x_pt = torch.from_numpy(rng.standard_normal((B, C, T), dtype=np.float32))

        up_mlx = UpSample1d(ratio=ratio, kernel_size=K)
        up_mlx.filter = mx.array(filt.transpose(0, 2, 1))
        y_mlx = np.array(up_mlx(mx.array(x_pt.numpy().transpose(0, 2, 1))))

        x_pad = F.pad(x_pt, (pad, pad), mode="replicate")
        y_pt = ratio * F.conv_transpose1d(
            x_pad,
            torch.from_numpy(filt).expand(C, -1, -1),
            stride=ratio,
            groups=C,
        )
        y_pt = y_pt[..., pad_left:-pad_right].detach().numpy().transpose(0, 2, 1)

        self.assertEqual(y_mlx.shape, y_pt.shape)
        self.assertLess(float(np.max(np.abs(y_mlx - y_pt))), 1e-5)

    def test_chinese_lyrics_g2p_without_torch(self) -> None:
        import json

        pointer = (
            Path(__file__).resolve().parents[1]
            / "default_config"
            / "workspace.pointer.json"
        )
        if not pointer.is_file():
            self.skipTest("workspace.pointer.json missing")
        ws = json.loads(pointer.read_text(encoding="utf-8")).get("custom_workspace_dir")
        bundle = Path(ws) / "models" / "Audio" / "diffrhythm-v2"
        if not (bundle / "g2p" / "g2p_generation.py").is_file():
            self.skipTest("diffrhythm-v2 g2p bundle not present")

        from backend.engine.families.diffrhythm.condition_mlx import (
            parse_lyrics_to_token_ids,
            set_g2p_bundle_root,
        )

        set_g2p_bundle_root(bundle)
        tokens = parse_lyrics_to_token_ids(
            "[verse]\n月光洒在窗前\n",
            vocal_language="zh",
        )
        self.assertGreater(len(tokens), 0)

    def test_english_lyrics_g2p_matches_upstream_bundle(self) -> None:
        import json
        import sys

        pointer = (
            Path(__file__).resolve().parents[1]
            / "default_config"
            / "workspace.pointer.json"
        )
        if not pointer.is_file():
            self.skipTest("workspace.pointer.json missing")
        ws = json.loads(pointer.read_text(encoding="utf-8")).get("custom_workspace_dir")
        bundle = Path(ws) / "models" / "Audio" / "diffrhythm-v2"
        if not (bundle / "g2p" / "g2p_generation.py").is_file():
            self.skipTest("diffrhythm-v2 g2p bundle not present")

        from backend.engine.families.diffrhythm import g2p as chinese_poly_g2p
        from backend.engine.families.diffrhythm.condition_mlx import (
            parse_lyrics_to_token_ids,
            set_g2p_bundle_root,
        )
        from backend.engine.families.diffrhythm.g2p import install_bundle_g2p_path

        install_bundle_g2p_path(bundle)
        sys.modules["g2p.g2p.chinese_model_g2p"] = chinese_poly_g2p
        from g2p.g2p_generation import chn_eng_g2p

        set_g2p_bundle_root(bundle)
        line = "La la la under the moonlight"
        _, upstream = chn_eng_g2p(line)
        upstream = [x + 1 for x in upstream]
        ours = parse_lyrics_to_token_ids(f"[verse]\n{line}\n", vocal_language="en")
        # strip auto [start] prefix and trailing [stop] after lyric line
        self.assertEqual(ours[:2], [500, 511])
        self.assertEqual(ours[2:4], [503, 511])
        self.assertEqual(ours[4:-1], upstream)
        self.assertEqual(ours[-1], 511)

    def test_muq_style_encoder_latent_dim_when_bundle_present(self) -> None:
        import json

        pointer = (
            Path(__file__).resolve().parents[1]
            / "default_config"
            / "workspace.pointer.json"
        )
        if not pointer.is_file():
            self.skipTest("workspace.pointer.json missing")
        ws = json.loads(pointer.read_text(encoding="utf-8")).get("custom_workspace_dir")
        bundle = Path(ws) / "models" / "Audio" / "diffrhythm-v2"
        if not (bundle / "mulan").is_dir():
            self.skipTest("diffrhythm-v2 mulan cache not present")

        from backend.engine.runtime.mlx import MLXContext
        from backend.engine.config.model_configs import DiffRhythmConfig
        from backend.engine.families.diffrhythm.mulan import MuQStyleEncoder

        cfg = DiffRhythmConfig()
        ctx = MLXContext()
        enc = MuQStyleEncoder(ctx, bundle / "mulan", cfg.mulan_repo_id)
        enc.load()
        latent = enc.encode_text("upbeat pop rock", array_fn=ctx.array)
        self.assertEqual(tuple(latent.shape), (512,))

    def test_muq_mlx_matches_torch_latent_when_bundle_present(self) -> None:
        import json

        import numpy as np

        pointer = (
            Path(__file__).resolve().parents[1]
            / "default_config"
            / "workspace.pointer.json"
        )
        if not pointer.is_file():
            self.skipTest("workspace.pointer.json missing")
        ws = json.loads(pointer.read_text(encoding="utf-8")).get("custom_workspace_dir")
        bundle = Path(ws) / "models" / "Audio" / "diffrhythm-v2" / "mulan"
        if not bundle.is_dir():
            self.skipTest("diffrhythm-v2 mulan cache not present")
        from backend.engine.config.model_configs import DiffRhythmConfig
        from backend.engine.families.diffrhythm.mulan_mlx import MuQStyleEncoderMLX
        from backend.engine.runtime.mlx import MLXContext

        cfg = DiffRhythmConfig()
        ctx = MLXContext()
        prompt = "upbeat pop rock with electric guitar"
        mlx_enc = MuQStyleEncoderMLX(bundle, cfg.mulan_repo_id, ctx)
        mlx_enc.load()
        lat_mlx = np.array(mlx_enc.encode_text(prompt), dtype=np.float32)
        self.assertEqual(lat_mlx.shape, (512,))
        self.assertAlmostEqual(float(np.linalg.norm(lat_mlx)), 1.0, places=3)

        try:
            import torch  # noqa: F401
            from backend.engine.families.diffrhythm.mulan_cuda import MuQStyleEncoderTorch
        except ImportError:
            return

        try:
            import muq  # noqa: F401
        except ImportError:
            return

        torch_enc = MuQStyleEncoderTorch(bundle, cfg.mulan_repo_id)
        torch_enc.load()
        lat_torch = np.array(torch_enc.encode_text(prompt, array_fn=ctx.array), dtype=np.float32)
        cos = float(
            np.dot(lat_mlx, lat_torch)
            / (np.linalg.norm(lat_mlx) * np.linalg.norm(lat_torch) + 1e-9)
        )
        self.assertGreater(cos, 0.999)
        self.assertLess(float(np.max(np.abs(lat_mlx - lat_torch))), 0.01)

    def test_bigvgan_mlx_decode_peak_when_bundle_present(self) -> None:
        import json
        import mlx.core as mx
        import numpy as np

        pointer = (
            Path(__file__).resolve().parents[1]
            / "default_config"
            / "workspace.pointer.json"
        )
        if not pointer.is_file():
            self.skipTest("workspace.pointer.json missing")
        ws = json.loads(pointer.read_text(encoding="utf-8")).get("custom_workspace_dir")
        bundle = Path(ws) / "models" / "Audio" / "diffrhythm-v2"
        if not (bundle / "decoder.bin").is_file():
            self.skipTest("diffrhythm-v2 bundle not installed")

        from backend.engine.runtime.mlx import MLXContext
        from backend.engine.families.diffrhythm.vae_mlx import DiffRhythm2DecoderMLX

        ctx = MLXContext()
        dec = DiffRhythm2DecoderMLX(ctx, vae_dir=str(bundle))
        lat = mx.array(np.random.default_rng(42).standard_normal((1, 20, 64), dtype=np.float32) * 0.5)
        audio = np.array(dec.generator(lat), dtype=np.float32)
        peak = float(np.max(np.abs(audio)))
        self.assertGreater(peak, 0.05, f"MLX BigVGAN decode too quiet (peak={peak})")


class LLMServiceTests(unittest.TestCase):
    def _load_service(self):
        from backend.core.model_registry import ModelRegistry
        from backend.engine.llm.service import LLMService
        from backend.utils.path_utils import PathResolver

        root = Path(__file__).resolve().parents[1]
        pr = PathResolver(root)
        mr = ModelRegistry.load(pr.get_models_registry_path())
        return LLMService(mr, pr)

    def test_get_model_info_reads_registry_name(self) -> None:
        svc = self._load_service()
        info = svc.get_model_info()
        self.assertEqual(info["model_id"], "qwen3-4b-thinking-2507")
        self.assertIsInstance(info["name"], dict)
        self.assertIn("zh", info["name"])
        self.assertIn("en", info["name"])
        self.assertIsInstance(info["available"], bool)

        vision = svc.get_vision_model_info()
        self.assertEqual(vision["model_id"], "qwen3-vl-4b-instruct")
        self.assertIsInstance(vision["name"], dict)

    def test_normalize_app_llm_settings_uses_saved_defaults(self) -> None:
        from backend.core.interfaces import AppSettings
        from backend.core.model_registry import ModelRegistry
        from backend.engine.llm.service import normalize_app_llm_settings, resolve_llm_model_id, resolve_vlm_model_id
        from backend.utils.path_utils import PathResolver

        root = Path(__file__).resolve().parents[1]
        registry = ModelRegistry.load(root / "default_config" / "models_registry.json")

        settings = AppSettings(
            default_model_llm="qwen2.5-1.5b",
            default_model_vlm="qwen2.5-vl-7b-instruct",
        )
        self.assertTrue(normalize_app_llm_settings(settings, registry))
        self.assertEqual(settings.default_model_llm, "qwen3-4b-thinking-2507")
        self.assertEqual(settings.default_model_vlm, "qwen3-vl-4b-instruct")
        self.assertEqual(resolve_llm_model_id(settings, registry), settings.default_model_llm)
        self.assertEqual(resolve_vlm_model_id(settings, registry), settings.default_model_vlm)

    def test_llm_think_mode_setting(self) -> None:
        from backend.core.model_registry import ModelRegistry
        from backend.engine.llm.service import LLMService
        from backend.utils.path_utils import PathResolver

        root = Path(__file__).resolve().parents[1]
        registry = ModelRegistry.load(root / "default_config" / "models_registry.json")
        svc = LLMService(
            registry,
            PathResolver(root),
            default_model_id="qwen3-4b-thinking-2507",
            llm_think_enabled=False,
        )
        self.assertFalse(svc.get_model_info()["think_enabled"])
        self.assertTrue(svc.get_model_info()["think_supported"])
        self.assertFalse(svc._resolve_enable_thinking(None))
        svc.apply_model_settings(llm_think_enabled=True)
        self.assertTrue(svc._resolve_enable_thinking(None))
        svc.apply_model_settings(default_model_id="qwen3-4b-thinking-2507")
        self.assertTrue(svc._llm_think_enabled)

    def test_enhance_system_prompt_by_target_action(self) -> None:
        from backend.engine.llm.service import LLMService

        self.assertIn("video", LLMService._enhance_system_prompt("video_create").lower())
        self.assertIn("music", LLMService._enhance_system_prompt("audio_create").lower())
        self.assertIn("image", LLMService._enhance_system_prompt("image_create").lower())

    def test_sanitize_lyrics_strips_word_loops(self) -> None:
        from backend.engine.llm.lyrics_sanitize import sanitize_lyrics_output
        from backend.engine.llm.service import LLMService
        from backend.engine.llm.think_parse import extract_final_llm_content

        think_open = "<" + "think" + ">"
        think_close = "</" + "think" + ">"
        planning = (
            f"{think_open}\nWe are writing in Chinese wuxia lyrics.\n{think_close}\n"
            "[Verse 1]\n剑气如虹破长空\n江湖夜雨十年灯\n[Chorus]\n侠骨柔情写春秋"
        )
        self.assertIn(
            "[Verse 1]",
            sanitize_lyrics_output(planning, think_enabled=True),
        )
        self.assertEqual(
            extract_final_llm_content(f"{think_open}\nOnly reasoning, no answer", think_enabled=True),
            "",
        )

        chinese = "[Verse 1]\n夏日海边\n浪花轻敲沙滩\n[Chorus]\n青春不散场"
        self.assertTrue(LLMService._lyrics_quality_ok(chinese))
        self.assertTrue(LLMService._lyrics_quality_ok("[Instrumental]"))
        self.assertFalse(
            LLMService._lyrics_quality_ok(
                "Okay, let's write ACE-Step lyrics for this summer pop track."
            )
        )
        self.assertFalse(
            LLMService._lyrics_quality_ok(
                "We are writing in Chinese. We'll structure [Verse 1] -> 2-4 lines"
            )
        )

        raw = """[Intro]
The melody so soft yet so divine

[Verse 1]
The wind blows gently on my face
The sky is vast yet my soul divine divine divine divine divine divine

[Chorus]
Should not appear
"""
        out = sanitize_lyrics_output(raw)
        self.assertIn("[Verse 1]", out)
        self.assertIn("The wind blows gently", out)
        self.assertNotIn("divine divine divine", out)
        self.assertNotIn("Should not appear", out)

        annotated = """[Verse 1]
青峰云海间 (5 chars) - "Green peaks, cloud sea"
古剑映寒月 (6 chars) - "Ancient sword reflects cold moon"
[Chorus]
我辈当如龙 (6 chars) - "We should be like dragons"
[Verse 2]
"""
        cleaned = sanitize_lyrics_output(annotated)
        self.assertIn("[Verse 1]", cleaned)
        self.assertIn("青峰云海间", cleaned)
        self.assertNotIn("chars)", cleaned)
        self.assertNotIn("Green peaks", cleaned)
        self.assertNotIn("[Verse 2]", cleaned)
        self.assertTrue(LLMService._lyrics_quality_ok(cleaned))

    def test_build_lyrics_user_message_markdown(self) -> None:
        from backend.engine.llm.service import LLMService

        msg = LLMService._build_lyrics_user_message("武侠电子，苍劲男声", "epic orchestral")
        self.assertIn("## Music description", msg)
        self.assertIn("武侠电子", msg)
        self.assertIn("## Style", msg)
        self.assertIn("epic orchestral", msg)

        plain = LLMService._build_lyrics_user_message("summer pop")
        self.assertNotIn("## Style", plain)

    def test_sanitize_enhanced_prompt_strips_comma_loops(self) -> None:
        from backend.engine.llm.prompt_sanitize import (
            prompt_enhance_quality_ok,
            sanitize_enhanced_prompt,
        )

        prefix = "赵今麦，古装写真，性感的妆容，素颜，灯光柔和，映衬着素色素衣"
        raw = prefix + "，" + "，".join(["素色素发"] * 60)
        out = sanitize_enhanced_prompt(raw)
        self.assertIn("赵今麦", out)
        self.assertIn("古装写真", out)
        self.assertLessEqual(out.count("素色素发"), 1)
        self.assertNotIn("素色素发，素色素发", out)
        self.assertTrue(prompt_enhance_quality_ok(out))

    def test_sanitize_enhanced_prompt_strips_tail_phrase_loops(self) -> None:
        from backend.engine.llm.prompt_sanitize import sanitize_enhanced_prompt

        raw = "A portrait of a woman in soft light" + (" plain hair" * 8)
        out = sanitize_enhanced_prompt(raw)
        self.assertIn("soft light", out)
        self.assertLessEqual(out.count("plain hair"), 2)

    def test_sanitize_enhanced_prompt_strips_cjk_tail_loops(self) -> None:
        from backend.engine.llm.prompt_sanitize import (
            prompt_enhance_quality_ok,
            sanitize_enhanced_prompt,
        )

        prefix = (
            "赵今麦身着透视旗袍，展示真实人体质感，其乳房和下体部分裸露，"
            "背景中可见丰富细节，整体呈现自然质感，"
        )
        raw = prefix + "背景中" * 20
        out = sanitize_enhanced_prompt(raw)
        self.assertIn("赵今麦", out)
        self.assertIn("整体呈现自然质感", out)
        self.assertLessEqual(out.count("背景中"), 2)
        self.assertFalse(out.endswith("背景中背景中"))
        self.assertTrue(prompt_enhance_quality_ok(out))

    def test_extract_final_llm_content_strips_thinking_blocks(self) -> None:
        from backend.engine.llm.prompt_sanitize import (
            looks_like_reasoning_trace,
            prompt_enhance_quality_ok,
        )
        from backend.engine.llm.think_parse import extract_final_llm_content

        think_open = "<" + "think" + ">"
        think_close = "</" + "think" + ">"
        tagged = (
            f"{think_open}\nOkay, let's tackle this prompt rewrite request.\n{think_close}\n"
            "杨紫，15岁，坐在阶梯教室听课，柔和自然光，写实摄影"
        )
        self.assertEqual(
            extract_final_llm_content(tagged),
            "杨紫，15岁，坐在阶梯教室听课，柔和自然光，写实摄影",
        )
        self.assertEqual(
            extract_final_llm_content(f"{think_open}\nOnly reasoning, no answer"),
            "",
        )
        self.assertTrue(
            looks_like_reasoning_trace(
                "Okay, let's tackle this prompt rewrite request. The user wants a Chinese-to-Chinese prompt."
            )
        )
        self.assertFalse(
            prompt_enhance_quality_ok(
                "Okay, let's tackle this prompt rewrite request. The user wants a Chinese-to-Chinese prompt."
            )
        )

    def test_generate_lyrics_forces_no_think_mode(self) -> None:
        from backend.engine.llm.service import LLMService

        svc = self._load_service()
        svc.apply_model_settings(llm_think_enabled=True)
        calls: list[tuple[bool | None, str]] = []

        def capture(_request, enable_thinking=None):
            calls.append((enable_thinking, _request.messages[-1].content))
            raise RuntimeError("stop-after-capture")

        svc.chat_completion = capture  # type: ignore[method-assign]
        with self.assertRaises(RuntimeError):
            svc.generate_lyrics("夏日流行，海边青春")
        self.assertGreaterEqual(len(calls), 1)
        for enable_thinking, user_content in calls:
            self.assertIs(enable_thinking, False)
            self.assertIn("/no_think", user_content)
            self.assertNotIn("/think", user_content)

    def test_generation_kwargs_uses_sampler(self) -> None:
        from backend.core.contracts import ChatCompletionRequest, ChatMessage
        from backend.core.model_registry import ModelRegistry
        from backend.engine.llm.service import LLMService
        from backend.utils.path_utils import PathResolver

        root = Path(__file__).resolve().parents[1]
        registry = ModelRegistry.load(root / "default_config" / "models_registry.json")
        svc = LLMService(registry, PathResolver(root), default_model_id="qwen3-4b-thinking-2507")

        request = ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="hi")],
            temperature=0.7,
            top_p=0.9,
            max_tokens=128,
        )
        kwargs = svc._generation_kwargs(request)
        self.assertEqual(kwargs["max_tokens"], 128)
        self.assertIn("sampler", kwargs)
        self.assertNotIn("temp", kwargs)
        self.assertTrue(callable(kwargs["sampler"]))

        think_kwargs = svc._generation_kwargs(request, think_active=True)
        self.assertGreater(think_kwargs["max_tokens"], 128)
        self.assertLessEqual(think_kwargs["max_tokens"], 8192)

    def test_think_mode_suffix_respects_settings(self) -> None:
        from backend.core.model_registry import ModelRegistry
        from backend.engine.llm.service import LLMService
        from backend.utils.path_utils import PathResolver

        root = Path(__file__).resolve().parents[1]
        registry = ModelRegistry.load(root / "default_config" / "models_registry.json")
        svc = LLMService(
            registry,
            PathResolver(root),
            default_model_id="qwen3-4b-thinking-2507",
            llm_think_enabled=True,
        )
        self.assertTrue(svc._apply_think_mode_to_text("hello").endswith("/think"))
        svc.apply_model_settings(llm_think_enabled=False)
        self.assertTrue(svc._apply_think_mode_to_text("hello").endswith("/no_think"))

    def test_coerce_vlm_output_text_handles_generation_result(self) -> None:
        from backend.engine.llm.vision import _coerce_vlm_output_text

        class _FakeGen:
            text = "  a red apple  "

        self.assertEqual(_coerce_vlm_output_text(_FakeGen()), "a red apple")
        self.assertEqual(_coerce_vlm_output_text("plain"), "plain")

    def test_enhance_prompt_when_model_installed(self) -> None:
        from backend.core.contracts import EnhanceRequest

        svc = self._load_service()
        if not svc.is_available():
            self.skipTest("no LLM installed in workspace")

        result = svc.enhance_prompt(EnhanceRequest(prompt="一只橘猫"))
        self.assertTrue(result.enhanced_prompt.strip())


class QuantizedInferenceModeTests(unittest.TestCase):
    def test_resolve_dense_for_full_precision_version(self) -> None:
        from backend.engine.common.bundle.quant_inference import resolve_inference_weight_mode

        entry = type("E", (), {"raw": {"versions": {"fp16": {"source_type": "full"}}}})()
        mode = resolve_inference_weight_mode(entry, "fp16", type("C", (), {"backend": "mlx"})())
        self.assertEqual(mode.kind, "dense")

    def test_resolve_quantized_for_registry_bits(self) -> None:
        from backend.engine.common.bundle.quant_inference import resolve_inference_weight_mode

        entry = type(
            "E",
            (),
            {
                "raw": {
                    "versions": {
                        "int4": {
                            "quantization": {"bits": 4, "scheme": "mlx_affine"},
                        }
                    }
                }
            },
        )()
        mode = resolve_inference_weight_mode(
            entry,
            "int4",
            type("C", (), {"backend": "mlx"})(),
            weight_keys=frozenset({"layers.0.weight", "layers.0.scales", "layers.0.biases"}),
            bundle_affine_bits=4,
        )
        self.assertEqual(mode.kind, "quantized")
        self.assertEqual(mode.bits, 4)
        self.assertEqual(mode.cache_suffix(), ":q4")

    def test_read_affine_quant_bits_from_quantize_config(self) -> None:
        import json
        import tempfile
        from pathlib import Path

        from backend.engine.common.bundle.safetensors_affine_quant import (
            read_affine_quant_bits_from_quantize_config,
            read_affine_quant_group_size_from_quantize_config,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "quantize_config.json").write_text(
                json.dumps({"quantization": {"bits": 4, "group_size": 64}}),
                encoding="utf-8",
            )
            self.assertEqual(read_affine_quant_bits_from_quantize_config(root), 4)
            self.assertEqual(read_affine_quant_group_size_from_quantize_config(root), 64)

    def test_resolve_quantized_group_size_from_registry(self) -> None:
        from backend.engine.common.bundle.quant_inference import resolve_inference_weight_mode

        entry = type(
            "E",
            (),
            {
                "raw": {
                    "versions": {
                        "mlx-q4": {
                            "quantization": {
                                "bits": 4,
                                "scheme": "mlx_affine",
                                "group_size": 64,
                            },
                        }
                    }
                }
            },
        )()
        mode = resolve_inference_weight_mode(
            entry,
            "mlx-q4",
            type("C", (), {"backend": "mlx"})(),
            weight_keys=frozenset({"blocks.0.weight", "blocks.0.scales"}),
        )
        self.assertEqual(mode.kind, "quantized")
        self.assertEqual(mode.bits, 4)
        self.assertEqual(mode.group_size, 64)

    def test_quantized_inference_rejects_non_mlx_backend(self) -> None:
        from backend.engine.common.bundle.quant_inference import resolve_inference_weight_mode

        entry = type(
            "E",
            (),
            {"raw": {"versions": {"int4": {"quantization": {"bits": 4, "scheme": "mlx_affine"}}}}},
        )()
        with self.assertRaises(RuntimeError):
            resolve_inference_weight_mode(
                entry,
                "int4",
                type("C", (), {"backend": "cuda"})(),
                weight_keys=frozenset({"a.weight", "a.scales"}),
            )

    def test_quantized_bundle_without_scales_fails_loud(self) -> None:
        from backend.engine.common.bundle.quant_inference import resolve_inference_weight_mode

        entry = type(
            "E",
            (),
            {"raw": {"versions": {"int4": {"quantization": {"bits": 4, "scheme": "mlx_affine"}}}}},
        )()
        with self.assertRaises(RuntimeError):
            resolve_inference_weight_mode(
                entry,
                "int4",
                type("C", (), {"backend": "mlx"})(),
                weight_keys=frozenset({"layers.0.weight"}),
            )

    def test_cache_size_estimate_scales_with_bits(self) -> None:
        from backend.engine.common.bundle.quant_inference import (
            WeightInferenceMode,
            estimate_dit_cache_size_gb,
        )

        dense = estimate_dit_cache_size_gb(24.0, WeightInferenceMode(kind="dense"))
        q4 = estimate_dit_cache_size_gb(24.0, WeightInferenceMode(kind="quantized", bits=4))
        self.assertEqual(dense, 24.0)
        self.assertLess(q4, dense)
        self.assertAlmostEqual(q4, 6.0)

    def test_quantized_lora_allowed_by_default(self) -> None:
        from backend.engine.common.bundle.quant_inference import (
            assert_quantized_dit_lora_compatible,
            entry_allows_quantized_lora,
        )

        entry = type(
            "E",
            (),
            {"raw": {"versions": {"int4": {"quantization": {"bits": 4, "scheme": "mlx_affine"}}}}},
        )()
        self.assertTrue(entry_allows_quantized_lora(entry, "int4"))
        assert_quantized_dit_lora_compatible(entry, "int4", [{"id": "lora"}])

    def test_quantized_lora_opt_out_in_registry(self) -> None:
        from backend.engine.common.bundle.quant_inference import (
            assert_quantized_dit_lora_compatible,
            entry_allows_quantized_lora,
        )

        entry = type(
            "E",
            (),
            {
                "raw": {
                    "versions": {
                        "int4": {
                            "quantization": {
                                "bits": 4,
                                "scheme": "mlx_affine",
                                "quantized_lora": False,
                            }
                        }
                    }
                }
            },
        )()
        self.assertFalse(entry_allows_quantized_lora(entry, "int4"))
        with self.assertRaises(RuntimeError):
            assert_quantized_dit_lora_compatible(entry, "int4", [{"id": "lora"}])

    def test_resolve_from_bundle_without_registry_entry(self) -> None:
        from backend.engine.common.bundle.quant_inference import (
            resolve_dit_inference_weight_mode,
            resolve_inference_weight_mode_from_bundle,
        )

        ctx = type("C", (), {"backend": "mlx"})()
        mode = resolve_inference_weight_mode_from_bundle(
            ctx,
            weight_keys=frozenset({"layers.0.weight", "layers.0.scales"}),
            bundle_affine_bits=4,
        )
        self.assertEqual(mode.kind, "quantized")
        self.assertEqual(mode.bits, 4)

        dense = resolve_dit_inference_weight_mode(
            ctx,
            entry=None,
            version_key=None,
            weight_keys=frozenset({"layers.0.weight"}),
            bundle_affine_bits=None,
        )
        self.assertEqual(dense.kind, "dense")

    def test_vae_stays_dense_when_dit_quant_but_no_affine_tensors(self) -> None:
        from backend.engine.common.bundle.quant_inference import resolve_component_inference_weight_mode

        entry = type(
            "E",
            (),
            {"raw": {"versions": {"int4": {"quantization": {"bits": 4, "scheme": "mlx_affine"}}}}},
        )()
        mode = resolve_component_inference_weight_mode(
            entry,
            "int4",
            type("C", (), {"backend": "mlx"})(),
            component="vae",
            weight_keys=frozenset({"decoder.conv_in.weight"}),
        )
        self.assertEqual(mode.kind, "dense")

    def test_vae_quantized_when_affine_tensors_present(self) -> None:
        from backend.engine.common.bundle.quant_inference import resolve_component_inference_weight_mode

        entry = type(
            "E",
            (),
            {"raw": {"versions": {"int4": {"quantization": {"bits": 4, "scheme": "mlx_affine"}}}}},
        )()
        mode = resolve_component_inference_weight_mode(
            entry,
            "int4",
            type("C", (), {"backend": "mlx"})(),
            component="vae",
            weight_keys=frozenset({"mid_attn.to_q.weight", "mid_attn.to_q.scales"}),
            bundle_affine_bits=4,
        )
        self.assertEqual(mode.kind, "quantized")
        self.assertEqual(mode.bits, 4)

    def test_explicit_vae_quant_without_scales_fails_loud(self) -> None:
        from backend.engine.common.bundle.quant_inference import resolve_component_inference_weight_mode

        entry = type(
            "E",
            (),
            {
                "raw": {
                    "versions": {
                        "int4": {
                            "quantization": {
                                "bits": 4,
                                "scheme": "mlx_affine",
                                "vae": {"bits": 4},
                            }
                        }
                    }
                }
            },
        )()
        with self.assertRaises(RuntimeError):
            resolve_component_inference_weight_mode(
                entry,
                "int4",
                type("C", (), {"backend": "mlx"})(),
                component="vae",
                weight_keys=frozenset({"decoder.conv_in.weight"}),
            )


class DerivedQuantLayoutTests(unittest.TestCase):
    def test_component_quant_targets_exclude_copy_subdirs(self) -> None:
        import tempfile
        from pathlib import Path

        from backend.core.derived_quant_layout import resolve_derived_quant_layout

        with tempfile.TemporaryDirectory() as tmp:
            from_root = Path(tmp) / "from"
            to_root = Path(tmp) / "to"
            te_dir = from_root / "text_encoder"
            tr_dir = from_root / "transformer"
            te_dir.mkdir(parents=True)
            tr_dir.mkdir(parents=True)
            (te_dir / "model.safetensors").write_bytes(b"stub")
            (tr_dir / "model_00000.safetensors").write_bytes(b"stub")

            plan = resolve_derived_quant_layout(
                family="flux2",
                from_root=from_root,
                to_root=to_root,
                to_ver_config={
                    "quantization": {
                        "bits": 4,
                        "text_encoder": {"bits": 4},
                    }
                },
            )
            self.assertEqual(len(plan.component_targets), 1)
            self.assertEqual(plan.component_targets[0].subdir, "text_encoder")
            self.assertEqual(plan.component_targets[0].bits, 4)
            self.assertNotIn("text_encoder", plan.copy_subdirs)
            self.assertIn("text_encoder", plan.exclude_subdirs)

    def test_family_layout_defaults(self) -> None:
        from backend.core.derived_quant_layout import _FAMILY_DEFAULT_LAYOUT

        self.assertEqual(_FAMILY_DEFAULT_LAYOUT["wan"], "wan_dit_shards")
        self.assertEqual(_FAMILY_DEFAULT_LAYOUT["diffrhythm"], "dit_single_file")
        self.assertEqual(_FAMILY_DEFAULT_LAYOUT["ace_step"], "dit_single_file")

    def test_flux2_klein_4b_derived_declares_text_encoder_quant(self) -> None:
        model = _load_default_registry_expanded()["models"]["flux2-klein-4b"]
        for ver_key in ("int4", "int8"):
            quant = (model["versions"][ver_key].get("quantization") or {})
            te = quant.get("text_encoder") or {}
            self.assertEqual(te.get("bits"), quant.get("bits"), ver_key)

    def test_registry_derived_versions_declare_parent_and_bits(self) -> None:
        for model_id, model in _load_default_registry_expanded()["models"].items():
            for ver_key, ver in (model.get("versions") or {}).items():
                if ver.get("source_type") != "derived":
                    continue
                self.assertTrue(
                    ver.get("from_version"),
                    f"{model_id}/{ver_key} missing from_version",
                )
                bits = (ver.get("quantization") or {}).get("bits")
                self.assertIn(bits, (4, 8), f"{model_id}/{ver_key} missing quantization.bits")


class InferenceLayerTests(unittest.TestCase):
    def test_dual_forward_cfg_renorm(self) -> None:
        from backend.engine.inference.cfg_strategies import DualForwardCfgStrategy

        class _Model:
            def __call__(self, latents, t, **kwargs):
                branch = kwargs.get("branch", "cond")
                return f"noise_{branch}"

            def combine_cfg_noise(self, cond, uncond, guidance):
                return f"combined(g={guidance})"

            def refine_cfg_noise(self, cond, pred, *, cfg_renorm_min):
                return f"refined(min={cfg_renorm_min})"

        model = _Model()
        strat = DualForwardCfgStrategy()
        out = strat.predict_noise(
            model,
            "latents",
            0.5,
            cond_kwargs={"branch": "cond"},
            uncond_kwargs={"branch": "uncond"},
            guidance=3.5,
            cfg_renorm=True,
            cfg_renorm_min=0.12,
        )
        self.assertEqual(out, "refined(min=0.12)")

    def test_batched_cfg_skips_when_no_uncond_embeds(self) -> None:
        from backend.engine.inference.cfg_strategies import BatchedCfgStrategy

        class _Model:
            def __init__(self) -> None:
                self.calls: list[str] = []

            def __call__(self, latents, t, **kwargs):
                self.calls.append("forward")
                return "single"

            def predict_noise_cfg(self, *args, **kwargs):
                raise AssertionError("batched CFG should not run without uncond txt_embeds")

        model = _Model()
        strat = BatchedCfgStrategy()
        out = strat.predict_noise(
            model,
            "latents",
            0.5,
            cond_kwargs={"txt_embeds": "pos"},
            uncond_kwargs=None,
            guidance=1.0,
        )
        self.assertEqual(out, "single")
        self.assertEqual(model.calls, ["forward"])

    def test_diffusion_inference_passes_cfg_renorm(self) -> None:
        from backend.engine.inference._protocols import InferenceBundle
        from backend.engine.inference.diffusion import DiffusionInference

        captured: dict[str, float | bool] = {}

        class _CfgStrategy:
            def predict_noise(self, model, latents, t, **kwargs):
                captured["cfg_renorm"] = kwargs.get("cfg_renorm", False)
                captured["cfg_renorm_min"] = kwargs.get("cfg_renorm_min", 0.0)
                return latents

        class _Scheduler:
            def step(self, noise_pred, t, latents):
                return latents

        class _Ctx:
            backend = "cpu"

            def eval(self, *_args):
                return None

            def clear_cache(self):
                return None

            def active_memory_gb(self):
                return 0.0

        class _Model:
            def step_callback(self, *_args):
                return None

        bundle = InferenceBundle(
            ctx=_Ctx(),
            model=_Model(),
            config=object(),
            scheduler=_Scheduler(),
            timesteps=[1.0],
            init_latents="latents",
            cfg_strategy=_CfgStrategy(),
            cfg_renorm=True,
            cfg_renorm_min=0.25,
        )
        DiffusionInference(_Ctx()).run(bundle)
        self.assertTrue(captured["cfg_renorm"])
        self.assertEqual(captured["cfg_renorm_min"], 0.25)

    def test_cancel_token_requires_is_cancelled(self) -> None:
        from backend.engine.inference._runtime import is_cancelled

        with self.assertRaises(RuntimeError):
            is_cancelled(object())

    def test_flow_matching_spec_bundle(self) -> None:
        from backend.engine.inference._protocols import AudioInferenceBundle, FlowMatchingSpec
        from backend.engine.inference.flow_matching import FlowMatchingInference

        seen: list[float] = []

        def _forward(x, t, state):
            seen.append(t)
            return x

        bundle = AudioInferenceBundle(
            ctx=None,
            model_forward=_forward,
            latent_shape=(1,),
            flow=FlowMatchingSpec(
                timestep_schedule=[1.0, 0.0],
                init_noise_fn=lambda _shape, _seed: 0.0,
                euler_step_fn=lambda x, _v, _tc, _tn, _i: x,
            ),
        )
        result = FlowMatchingInference().run(bundle)
        self.assertEqual(seen, [1.0, 0.0])
        self.assertEqual(result["latents"], 0.0)

    def test_run_diffusion_denoise_helper(self) -> None:
        from backend.engine.inference.diffusion_bundle import run_diffusion_denoise

        seen: list[int] = []

        class _Scheduler:
            def step(self, noise_pred, t, latents):
                return latents

        class _Model:
            def __call__(self, latents, t, **kwargs):
                return latents

            def step_callback(self, *_args):
                return None

        class _Ctx:
            backend = "cpu"

            def eval(self, *_args):
                return None

            def clear_cache(self):
                return None

            def active_memory_gb(self):
                return 0.0

        def _on_step(result) -> None:
            seen.append(result.step_idx)

        out = run_diffusion_denoise(
            _Ctx(),
            model=_Model(),
            config=object(),
            scheduler=_Scheduler(),
            timesteps=[1.0, 0.5],
            latents="latents",
            guidance=0.0,
            cancel_token=None,
            step_kwargs_builder=None,
            on_step_complete=_on_step,
        )
        self.assertEqual(out, "latents")
        self.assertEqual(seen, [0, 1])

    def test_get_audio_edit_handler_resolves_cover(self) -> None:
        from backend.engine._transformer_registry import get_audio_edit_handler
        from backend.engine.families.ace_step.generation import run_cover_edit

        handler = get_audio_edit_handler("ace_step", "cover")
        self.assertIs(handler, run_cover_edit)

    def test_image_fill_edit_routes_to_session(self) -> None:
        from unittest.mock import MagicMock

        from backend.engine.registry.bootstrap import bootstrap_family_plugins
        from backend.engine.sessions.image_session import routes_to_image_edit_session

        bootstrap_family_plugins()
        registry = MagicMock()
        registry.get.return_value = MagicMock(
            family="flux1",
            media="image",
            actions=frozenset({"edit"}),
        )
        self.assertTrue(routes_to_image_edit_session("flux-fill-controlnet", registry))

    def test_get_audio_post_generation_optional(self) -> None:
        from backend.engine._transformer_registry import get_audio_post_generation

        self.assertIsNotNone(get_audio_post_generation("ace_step"))
        self.assertIsNone(get_audio_post_generation("diffrhythm"))

    def test_autoregressive_inference_scaffold(self) -> None:
        from backend.engine.inference.autoregressive import AutoregressiveBundle, AutoregressiveInference

        def prefill_fn(_tokens):
            return [0.0, 1.0], {"n": 0}

        def logits_fn(_token_id, state):
            state["n"] += 1
            return [float(state["n"]), 0.0]

        def sample_fn(logits):
            return int(logits.index(max(logits)))

        result = AutoregressiveInference().run(
            AutoregressiveBundle(
                prompt_tokens=[0],
                max_new_tokens=2,
                prefill_fn=prefill_fn,
                logits_fn=logits_fn,
                sample_fn=sample_fn,
                eos_token_ids={99},
            )
        )
        self.assertEqual(result["num_tokens"], 2)
        self.assertEqual(len(result["tokens"]), 2)

    def test_validate_video_generation_params_ltx_distilled(self) -> None:
        from types import SimpleNamespace

        from backend.engine._transformer_registry import validate_video_generation_params
        from backend.engine.config.model_configs import LTXConfig

        entry = SimpleNamespace(id="ltx-2.3-distilled")
        config = LTXConfig()
        with self.assertRaises(RuntimeError) as ctx:
            validate_video_generation_params(
                "ltx",
                entry=entry,
                config=config,
                step_distill=False,
            )
        self.assertIn("step_distill", str(ctx.exception))

    def test_ltx_connector_bundle_key_remap(self) -> None:
        import mlx.core as mx

        from backend.engine.families.ltx.text_encoder_mlx import (
            _GemmaFeaturesExtractor,
            _load_connector_weights,
            _materialize,
            _remap_connector_bundle_keys,
        )
        from tests.benchmark.registry_utils import resolve_benchmark_data_root

        src_key = "audio_embeddings_connector.transformer_1d_blocks.0.ff.net.0.proj.weight"
        dst_key = "audio_embeddings_connector.transformer_1d_blocks.0.ff.net.0.weight"
        remapped = _remap_connector_bundle_keys({src_key: mx.zeros((4, 8))})
        self.assertIn(dst_key, remapped)
        self.assertNotIn(src_key, remapped)

        bundle = resolve_benchmark_data_root() / "models/Video/ltx-2.3-distilled-mlx-q4"
        if not (bundle / "connector.safetensors").is_file():
            return

        from backend.engine.runtime.mlx import MLXContext

        ctx = MLXContext()
        weights = _load_connector_weights(bundle, ctx.load_weights)
        ext = _GemmaFeaturesExtractor()
        ext.connector.load_weights(list(weights.items()))
        conn = ext.connector
        _materialize(
            conn.video_embeddings_connector.learnable_registers,
            conn.audio_embeddings_connector.learnable_registers,
        )

    def test_ltx_connector_attention_output_rank(self) -> None:
        import mlx.core as mx

        from backend.engine.families.ltx.text_encoder_mlx import _ConnectorAttention

        attn = _ConnectorAttention(dim=4096, num_heads=32, head_dim=128)
        x = mx.zeros((1, 16, 4096))
        seq = 16
        inner = 4096
        num_heads = 32
        head_dim_half = inner // (2 * num_heads)
        cos_f = mx.zeros((1, num_heads, seq, head_dim_half))
        sin_f = mx.zeros((1, num_heads, seq, head_dim_half))
        out = attn(x, cos_f, sin_f)
        self.assertEqual(tuple(out.shape), (1, 16, 4096))

    def test_attach_image_conditioning_noop_without_guide(self) -> None:
        from types import SimpleNamespace

        from backend.engine._transformer_registry import attach_image_conditioning

        request = SimpleNamespace(structural_guide=None)
        extra = {"txt": 1}
        out, cleanup = attach_image_conditioning(
            pipeline=object(),
            request=request,
            family="flux1",
            model=object(),
            entry=SimpleNamespace(id="flux1-dev"),
            version_key=None,
            extra_cond=extra,
            width=512,
            height=512,
            ctx_exec=None,
            on_log=None,
        )
        self.assertIs(out, extra)
        self.assertIsNone(cleanup)

    def test_vae_preview_handler_lookup_flux2(self) -> None:
        from backend.engine.vae_codec_registry import (
            get_vae_preview_decode_handler,
            get_vae_preview_warmup_handler,
        )

        self.assertIsNotNone(get_vae_preview_warmup_handler("AutoencoderKLFlux2"))
        self.assertIsNotNone(get_vae_preview_decode_handler("AutoencoderKLFlux2"))
        self.assertIsNone(get_vae_preview_warmup_handler("UnknownVAE"))

    def test_load_mlx_encoder_stack_unknown_kind(self) -> None:
        from backend.engine._transformer_registry import load_mlx_encoder_stack

        with self.assertRaises(RuntimeError) as ctx:
            load_mlx_encoder_stack("unknown_kind")
        self.assertIn("qwen25vl", str(ctx.exception))


class ArchitectureWrapUpTests(unittest.TestCase):
    def test_hunyuan_vae_chunk_from_registry(self) -> None:
        from types import SimpleNamespace

        import numpy as np

        from backend.engine.video_codec_registry import (
            resolve_hunyuan_vae_spatial_tiling,
            resolve_hunyuan_vae_temporal_chunk,
        )

        entry = SimpleNamespace(
            id="hunyuan-video-1.5-1080p-sr",
            parameters={"vae_temporal_chunk_size": 8, "vae_spatial_tiling": True},
        )

        def scalar(entry, key, default):
            return (getattr(entry, "parameters", None) or {}).get(key, default)

        short = np.zeros((1, 16, 4, 8, 8), dtype=np.float32)
        self.assertEqual(resolve_hunyuan_vae_temporal_chunk(entry, short, scalar), 0)
        long = np.zeros((1, 16, 16, 8, 8), dtype=np.float32)
        self.assertEqual(resolve_hunyuan_vae_temporal_chunk(entry, long, scalar), 8)
        self.assertTrue(resolve_hunyuan_vae_spatial_tiling(entry, scalar))

    def test_video_bundle_layout_handlers(self) -> None:
        from backend.engine.pipelines.video_bundle_layout import (
            resolve_video_transformer_weight_sources,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "transformer-dev.safetensors").write_bytes(b"x")
            tensor_root, shards = resolve_video_transformer_weight_sources(
                root, "ltx", "ltx-2.3-dev"
            )
            self.assertEqual(tensor_root, root)
            self.assertEqual(len(shards), 1)


    def test_flux2_encode_handler_registered(self) -> None:
        from backend.engine.vae_codec_registry import (
            get_vae_encode_handler,
            registered_vae_encode_classes,
        )

        self.assertIn("AutoencoderKLFlux2", registered_vae_encode_classes())
        self.assertIsNotNone(get_vae_encode_handler("AutoencoderKLFlux2"))

    def test_require_entry_family_fail_loud(self) -> None:
        from types import SimpleNamespace

        from backend.engine.contracts import require_entry_family

        with self.assertRaises(RuntimeError):
            require_entry_family(SimpleNamespace(id="bad"), model_id="bad")

    def test_video_upscale_registry_resolve(self) -> None:
        from types import SimpleNamespace

        from backend.engine.video_upscale_registry import (
            get_video_upscale_runner,
            resolve_video_upscale_kind,
        )

        entry = SimpleNamespace(
            id="hunyuan-video-1.5-1080p-sr",
            raw={"versions": {"default": {"hunyuan_ms_variant": "1080p_sr_distilled"}}},
        )
        self.assertEqual(resolve_video_upscale_kind(entry, "default"), "hunyuan_1080p_sr")
        self.assertIsNotNone(get_video_upscale_runner("hunyuan_1080p_sr"))

        seed_entry = SimpleNamespace(
            id="seedvr2-video-7b",
            family="seedvr2",
            media="video",
            raw={"versions": {"fp16": {"video_upscale_kind": "seedvr2_spatiotemporal"}}},
        )
        self.assertEqual(resolve_video_upscale_kind(seed_entry, "fp16"), "seedvr2_spatiotemporal")
        self.assertIsNotNone(get_video_upscale_runner("seedvr2_spatiotemporal"))

    def test_video_hunyuan_step_distill_flag(self) -> None:
        from types import SimpleNamespace

        from backend.engine.contracts import (
            video_uses_hunyuan_step_distill_timesteps,
        )

        cfg = SimpleNamespace(video_i2v_style="hunyuan")
        self.assertTrue(
            video_uses_hunyuan_step_distill_timesteps(
                cfg, step_distill=True, scheduler_default="flow_match_euler",
            )
        )
        self.assertFalse(
            video_uses_hunyuan_step_distill_timesteps(
                cfg, step_distill=False, scheduler_default="flow_match_euler",
            )
        )

    def test_video_ltx_distilled_flag(self) -> None:
        from types import SimpleNamespace

        from backend.engine.contracts import (
            video_uses_ltx_distilled_timesteps,
        )

        cfg = SimpleNamespace(video_i2v_style="ltx23")
        self.assertTrue(
            video_uses_ltx_distilled_timesteps(
                cfg, step_distill=True, scheduler_default="flow_match_euler",
            )
        )
        self.assertFalse(
            video_uses_ltx_distilled_timesteps(
                cfg, step_distill=False, scheduler_default="flow_match_euler",
            )
        )

    def test_ltx_weight_prepare_registry(self) -> None:
        from types import SimpleNamespace

        from backend.engine._transformer_registry import prepare_video_transformer_weights

        cfg = SimpleNamespace(uses_mlx_forge_weight_restore=False, validate_ltx_block_depth=False)
        w = {"blocks.0.weight": [1.0]}
        self.assertIs(prepare_video_transformer_weights("wan", cfg, w), w)
        self.assertEqual(
            prepare_video_transformer_weights("ltx", cfg, w),
            w,
        )

    def test_upscale_pipeline_loader_registered(self) -> None:
        from backend.engine.upscale_job_registry import get_upscale_pipeline_loader

        self.assertIsNotNone(get_upscale_pipeline_loader("seedvr2"))
        self.assertIsNone(get_upscale_pipeline_loader("flux2"))


class LoraDatasetStoreTests(unittest.TestCase):
    def test_create_and_import_assets(self) -> None:
        from backend.engine.training import dataset_store

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            assets = root / "outputs" / "assets"
            assets.mkdir(parents=True)
            sample = assets / "sample.png"
            sample.write_bytes(
                b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
                b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xdb\x00\x00\x00\x00IEND\xaeB`\x82"
            )

            ds = dataset_store.create_dataset(root, name="test", default_prompt="A photo of sks")
            dataset_id = ds["id"]
            self.assertEqual(ds["image_count"], 0)

            def resolve(aid: str) -> Path:
                if aid == "ast_test":
                    return sample
                raise FileNotFoundError(aid)

            out = dataset_store.import_dataset_from_assets(
                root,
                dataset_id,
                ["asset:ast_test"],
                resolve_asset_path=resolve,
            )
            self.assertEqual(out["image_count"], 1)
            self.assertTrue(out.get("cover_image"))
            listed = dataset_store.list_datasets(root)
            self.assertEqual(listed[0].get("cover_image"), out.get("cover_image"))
            rows = dataset_store.load_training_pairs(root, dataset_id)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0][1], "A photo of sks")

    def test_user_lora_registry_delete(self) -> None:
        from backend.engine.training.user_lora_registry import (
            delete_user_lora,
            list_user_loras,
            register_user_lora,
        )

        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp)
            entry = register_user_lora(
                cfg,
                name="demo",
                base_model="flux1-dev",
                local_path="models/Lora/demo",
            )
            self.assertEqual(len(list_user_loras(cfg)), 1)
            ok = delete_user_lora(cfg, entry["id"])
            self.assertTrue(ok)
            self.assertEqual(list_user_loras(cfg), [])

    def test_import_bundled_dog6_example(self) -> None:
        from backend.engine.training import dataset_store

        bundled = dataset_store.bundled_dog6_example_dir(
            Path(__file__).resolve().parents[1] / "default_config"
        )
        self.assertTrue((bundled / "train.jsonl").is_file(), bundled)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ds = dataset_store.import_dog6_example(root, bundled_root=bundled)
            self.assertGreaterEqual(int(ds.get("image_count") or 0), 3)
            rows = dataset_store.load_training_pairs(root, ds["id"])
            self.assertGreaterEqual(len(rows), 3)
            self.assertTrue(all("sks dog" in p for _, p in rows))

    def test_prepare_dit_for_lora_training_z_image(self) -> None:
        from backend.engine.runtime.mlx import MLXContext
        from backend.engine.config.model_configs import ZImageConfig
        from backend.engine.families.z_image.transformer import ZImageTransformer
        from backend.engine.training.lora_layers import (
            apply_lora_to_zimage_dit,
            prepare_dit_for_lora_training,
        )
        import mlx.nn as nn

        ctx = MLXContext()
        model = ZImageTransformer(ZImageConfig(), ctx)
        dit, train_module = prepare_dit_for_lora_training(
            model,
            apply_lora_to_zimage_dit,
            rank=4,
            lora_blocks=1,
        )
        self.assertIs(model, dit)
        self.assertIsInstance(train_module, nn.Module)
        self.assertTrue(train_module.trainable_parameters())

    def test_delete_dataset(self) -> None:
        from backend.engine.training import dataset_store

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ds = dataset_store.create_dataset(root, name="to-delete")
            dataset_id = ds["id"]
            dataset_store.delete_dataset(root, dataset_id)
            self.assertFalse((root / "datasets" / dataset_id).exists())
            with self.assertRaises(FileNotFoundError):
                dataset_store.get_dataset(root, dataset_id)


class LoraQualityTests(unittest.TestCase):
    def test_parse_vlm_audit_output(self) -> None:
        from backend.engine.training.lora_quality_vlm import parse_vlm_audit_output

        text = "SCORE: 2\nISSUES: blurry,small_face\nREASON: Face is tiny and soft."
        parsed = parse_vlm_audit_output(text)
        self.assertEqual(parsed["score"], 2.0)
        self.assertIn("blurry", parsed["issues"])
        self.assertIn("small_face", parsed["issues"])

    def test_merge_vlm_hints_downgrades_level(self) -> None:
        from backend.engine.training.lora_quality_vlm import merge_vlm_hints

        base = {"level": "good", "score": 90, "hints": []}
        vlm = {
            "hints": [{"code": "vlm_low_portrait_score", "severity": "error", "params": {}, "source": "vlm"}],
            "avg_score": 2.0,
            "samples": [],
        }
        merged = merge_vlm_hints(base, vlm)
        self.assertEqual(merged["level"], "poor")

    def test_build_dataset_audit_instruction_style(self) -> None:
        from backend.engine.training.lora_quality_vlm import build_dataset_audit_instruction

        style = build_dataset_audit_instruction("style")
        concept = build_dataset_audit_instruction("concept")
        self.assertIn("style", style.lower())
        self.assertIn("face", concept.lower())

    def test_resolve_audit_paths_all_images(self) -> None:
        from backend.engine.training.lora_quality_vlm import resolve_audit_paths

        paths = [Path(f"img_{i}.jpg") for i in range(12)]
        picked, truncated = resolve_audit_paths(paths, max_samples=0)
        self.assertEqual(len(picked), 12)
        self.assertFalse(truncated)

    def test_compile_dataset_vlm_report(self) -> None:
        from backend.engine.training.lora_quality_vlm import compile_dataset_vlm_report

        paths = [Path("a.jpg"), Path("b.jpg")]
        texts = [
            "SCORE: 2\nISSUES: blurry,small_face\nREASON: Face too small.",
            "SCORE: 4\nISSUES: good\nREASON: Clear portrait.",
        ]
        report = compile_dataset_vlm_report(paths, texts)
        self.assertIn("hints", report)
        self.assertEqual(len(report["samples"]), 2)

    def test_portrait_heuristic_penalizes_landscape_full_body(self) -> None:
        from PIL import Image

        from backend.engine.training.portrait_lora_suitability import analyze_portrait_training_image

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "wide.jpg"
            Image.new("RGB", (1600, 900), color=(180, 140, 120)).save(path, format="JPEG")
            result = analyze_portrait_training_image(path)
            self.assertLess(result["score_1_5"], 3.0)
            self.assertIn("landscape_framing", result["issues"])

    def test_portrait_heuristic_good_close_portrait(self) -> None:
        from PIL import Image, ImageDraw

        from backend.engine.training.portrait_lora_suitability import analyze_portrait_training_image

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "portrait.jpg"
            img = Image.new("RGB", (1080, 1440), color=(200, 180, 170))
            draw = ImageDraw.Draw(img)
            draw.ellipse((340, 220, 740, 720), fill=(240, 210, 190))
            draw.ellipse((430, 380, 520, 470), fill=(40, 40, 40))
            draw.ellipse((560, 380, 650, 470), fill=(40, 40, 40))
            img.save(path, format="JPEG")
            result = analyze_portrait_training_image(path)
            self.assertGreaterEqual(result["score_1_5"], 3.0)

    def test_compile_portrait_dataset_audit_merges_vlm_and_heuristic(self) -> None:
        from PIL import Image

        from backend.engine.training.lora_quality_vlm import compile_portrait_dataset_audit

        with tempfile.TemporaryDirectory() as tmp:
            good = Path(tmp) / "good.jpg"
            bad = Path(tmp) / "bad.jpg"
            Image.new("RGB", (1080, 1440), color=(128, 64, 32)).save(good, format="JPEG")
            Image.new("RGB", (1600, 900), color=(64, 32, 16)).save(bad, format="JPEG")
            texts = ["SCORE: 5\nISSUES: good\nREASON: Perfect portrait."]
            report = compile_portrait_dataset_audit(
                [good, bad],
                [good],
                texts,
                all_file_keys=["good.jpg", "bad.jpg"],
                vlm_file_keys=["good.jpg"],
            )
            self.assertEqual(len(report["samples"]), 2)
            by_file = {s["file"]: s for s in report["samples"]}
            self.assertLess(by_file["bad.jpg"]["score"], 3.0)
            self.assertIsNotNone(by_file["good.jpg"].get("vlm_score"))
            self.assertIsNotNone(by_file["bad.jpg"].get("heuristic_score"))

    def test_merge_vlm_and_heuristic_takes_minimum(self) -> None:
        from backend.engine.training.portrait_lora_suitability import merge_vlm_and_heuristic_sample

        merged = merge_vlm_and_heuristic_sample(
            file_key="a.jpg",
            vlm_parsed={"score": 5.0, "issues": ["good"], "reason": "VLM says great"},
            heuristic={"score_1_5": 2.0, "issues": ["tiny_face_in_crop"], "reason": "Face tiny"},
        )
        self.assertEqual(merged["score"], 2.0)
        self.assertIn("tiny_face_in_crop", merged["issues"])
        self.assertFalse(merged["suitable_for_training"])

    def test_compose_person_caption_cjk(self) -> None:
        from backend.engine.training.lora_auto_caption import compose_person_caption

        self.assertEqual(
            compose_person_caption("杨紫", "半身照，白色衬衫，室内"),
            "杨紫，半身照，白色衬衫，室内",
        )

    def test_compose_person_caption_english(self) -> None:
        from backend.engine.training.lora_auto_caption import compose_person_caption

        self.assertEqual(
            compose_person_caption("cyq", "bust portrait, red dress, outdoor"),
            "cyq, bust portrait, red dress, outdoor",
        )

    def test_resolve_lora_subject_name_prefers_name_over_legacy_template(self) -> None:
        from backend.engine.training.lora_auto_caption import resolve_lora_subject_name

        self.assertEqual(
            resolve_lora_subject_name(
                {
                    "default_prompt": "A photo of sks person",
                    "trigger_word": "",
                    "name": "陈钰琪",
                }
            ),
            "陈钰琪",
        )
        self.assertEqual(
            resolve_lora_subject_name({"default_prompt": "杨紫", "trigger_word": "", "name": "dataset-1"}),
            "杨紫",
        )

    def test_caption_dataset_image_concept_prefixes_subject(self) -> None:
        from pathlib import Path

        from backend.engine.training.lora_auto_caption import caption_dataset_image

        def fake_analyze(_path: Path, _instruction: str) -> str:
            self.assertIn("杨紫", _instruction)
            return "半身照，白色连衣裙，自然光"

        caption = caption_dataset_image(
            Path("x.jpg"),
            Path("/tmp/vlm"),
            audit_kind="concept",
            subject_name="杨紫",
            analyze_fn=fake_analyze,
        )
        self.assertEqual(caption, "杨紫，半身照，白色连衣裙，自然光")

    def test_normalize_scene_caption_rejects_exclamation_garbage(self) -> None:
        from backend.engine.training.lora_auto_caption import compose_person_caption, normalize_scene_caption

        raw = "!" * 80
        self.assertEqual(normalize_scene_caption(raw), "")
        self.assertEqual(compose_person_caption("陈钰琪", raw), "陈钰琪")

    def test_caption_dataset_image_retries_after_garbage_vlm_output(self) -> None:
        from pathlib import Path

        from backend.engine.training.lora_auto_caption import caption_dataset_image

        calls: list[str] = []

        def fake_analyze(_path: Path, _instruction: str) -> str:
            calls.append(_instruction)
            if len(calls) == 1:
                return "!" * 100
            return "半身照，白色衬衫，室内"

        caption = caption_dataset_image(
            Path("x.jpg"),
            Path("/tmp/vlm"),
            audit_kind="concept",
            subject_name="陈钰琪",
            analyze_fn=fake_analyze,
        )
        self.assertEqual(caption, "陈钰琪，半身照，白色衬衫，室内")
        self.assertEqual(len(calls), 2)

    def test_caption_dataset_images_batch_retries_garbage_in_second_pass(self) -> None:
        from pathlib import Path

        from backend.engine.training.lora_auto_caption import caption_dataset_images_batch

        calls: list[tuple[str, int]] = []

        def fake_batch(image_paths: list[Path], instruction: str, max_tokens=128, temperature=0.2) -> list[str]:
            calls.append((instruction, len(image_paths)))
            if len(calls) == 1:
                return ["!" * 80, "半身照，红色连衣裙"]
            return ["半身照，白色衬衫"]

        caps = caption_dataset_images_batch(
            [Path("a.jpg"), Path("b.jpg")],
            Path("/tmp/vlm"),
            audit_kind="concept",
            subject_name="陈钰琪",
            batch_analyze_fn=fake_batch,
        )
        self.assertEqual(caps[0], "陈钰琪，半身照，白色衬衫")
        self.assertEqual(caps[1], "陈钰琪，半身照，红色连衣裙")
        self.assertEqual(len(calls), 2)

    def test_vlm_audit_worker_lora_caption_mode(self) -> None:
        from backend.engine.llm.vlm_audit_worker import run_job

        import backend.engine.training.lora_auto_caption as mod

        original = mod.caption_dataset_images_batch
        mod.caption_dataset_images_batch = lambda paths, model_dir, **kw: ["陈钰琪，半身照"]
        try:
            with tempfile.TemporaryDirectory() as tmp:
                model_dir = Path(tmp)
                out = run_job(
                    {
                        "mode": "lora_caption",
                        "image_paths": [str(Path(tmp) / "a.jpg")],
                        "model_dir": str(model_dir),
                        "audit_kind": "concept",
                        "subject_name": "陈钰琪",
                    }
                )
        finally:
            mod.caption_dataset_images_batch = original
        self.assertEqual(out["captions"], ["陈钰琪，半身照"])

    def test_create_dataset_defaults_prompt_to_name(self) -> None:
        from backend.engine.training import dataset_store

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ds = dataset_store.create_dataset(root, name="杨紫", kind="concept")
            self.assertEqual(ds.get("default_prompt"), "杨紫")

    def test_prepare_image_for_vlm_downscales(self) -> None:
        from PIL import Image

        from backend.engine.llm.vision import prepare_image_for_vlm

        with tempfile.TemporaryDirectory() as tmp:
            big = Path(tmp) / "big.jpg"
            Image.new("RGB", (2000, 3000), color=(10, 20, 30)).save(big, format="JPEG")
            use_path, is_temp = prepare_image_for_vlm(big, max_edge=768)
            try:
                with Image.open(use_path) as img:
                    self.assertLessEqual(min(img.size), 768)
            finally:
                if is_temp:
                    use_path.unlink(missing_ok=True)

    def _add_jpeg(self, root: Path, dataset_id: str, name: str, size: tuple[int, int]) -> None:
        from PIL import Image

        from backend.engine.training import dataset_store

        path = dataset_store.datasets_root(root) / dataset_id / "images" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", size, color=(128, 64, 32)).save(path, format="JPEG")

    def test_analyze_dataset_health_good(self) -> None:
        from backend.engine.training import dataset_store
        from backend.engine.training.lora_quality import analyze_dataset_health

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ds = dataset_store.create_dataset(root, name="good-set", default_prompt="A photo of sks")
            dataset_id = ds["id"]
            for i in range(12):
                self._add_jpeg(root, dataset_id, f"img_{i:02d}.jpg", (1080, 1440))
            rows = [
                {"image": f"images/img_{i:02d}.jpg", "prompt": "A photo of sks"}
                for i in range(12)
            ]
            jsonl = dataset_store.datasets_root(root) / dataset_id / "train.jsonl"
            jsonl.write_text(
                "\n".join(
                    __import__("json").dumps({"image": r["image"], "prompt": r["prompt"]})
                    for r in rows
                ),
                encoding="utf-8",
            )
            report = analyze_dataset_health(root, dataset_id)
            self.assertEqual(report["level"], "good")
            self.assertGreaterEqual(int(report["stats"]["median_short_edge"]), 1080)

    def test_analyze_dataset_health_poor_small_images(self) -> None:
        from backend.engine.training import dataset_store
        from backend.engine.training.lora_quality import analyze_dataset_health

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ds = dataset_store.create_dataset(root, name="cyq-like", default_prompt="A photo of sks")
            dataset_id = ds["id"]
            for i in range(6):
                self._add_jpeg(root, dataset_id, f"tiny_{i}.jpg", (320, 320))
            rows = [
                {"image": f"images/tiny_{i}.jpg", "prompt": "A photo of sks"} for i in range(6)
            ]
            jsonl = dataset_store.datasets_root(root) / dataset_id / "train.jsonl"
            jsonl.write_text(
                "\n".join(
                    __import__("json").dumps({"image": r["image"], "prompt": r["prompt"]})
                    for r in rows
                ),
                encoding="utf-8",
            )
            report = analyze_dataset_health(root, dataset_id)
            self.assertIn(report["level"], ("fair", "poor"))
            codes = {h["code"] for h in report["hints"]}
            self.assertTrue(codes & {"many_small_512", "many_small_600", "low_resolution_median"})

    def test_analyze_training_quality_high_initial_loss(self) -> None:
        from backend.engine.training.lora_quality import analyze_training_quality

        loss_history = [
            {"step": 10, "loss": 0.52},
            {"step": 20, "loss": 0.41},
            {"step": 30, "loss": 0.35},
        ]
        report = analyze_training_quality(loss_history)
        self.assertIn(report["level"], ("fair", "poor"))
        codes = {h["code"] for h in report["hints"]}
        self.assertIn("high_initial_loss", codes)

    def test_analyze_training_quality_healthy(self) -> None:
        from backend.engine.training.lora_quality import analyze_training_quality

        loss_history = [
            {"step": 10, "loss": 0.30},
            {"step": 100, "loss": 0.12},
            {"step": 200, "loss": 0.08},
        ]
        report = analyze_training_quality(loss_history)
        self.assertEqual(report["level"], "good")
        codes = {h["code"] for h in report["hints"]}
        self.assertIn("training_healthy", codes)


class LoraTrainRuntimeTests(unittest.TestCase):
    def test_train_min_memory_gb_qlora_lowers_threshold(self) -> None:
        from backend.engine.training.lora_train_runtime import train_min_memory_gb

        dense = train_min_memory_gb("flux1-dev")
        qlora4 = train_min_memory_gb("flux1-dev", qlora_bits=4)
        self.assertLess(qlora4, dense)

    def test_split_train_val_indices(self) -> None:
        from backend.engine.training.lora_train_runtime import split_train_val_indices

        train, val = split_train_val_indices(10, val_split=0.2)
        self.assertEqual(len(train) + len(val), 10)
        self.assertGreaterEqual(len(val), 1)
        self.assertGreaterEqual(len(train), 2)

    def test_parse_lora_train_runtime_config(self) -> None:
        from backend.engine.training.lora_train_runtime import parse_lora_train_runtime_config

        cfg = parse_lora_train_runtime_config(
            {
                "qlora_bits": 4,
                "optimizer": "adamw",
                "lora_scale": 16.0,
                "train_type": "dora",
                "min_snr_gamma": 5.0,
                "prior_loss_weight": 1.0,
                "early_stop_patience": 3,
                "fuse_adapters": True,
            },
            defaults={"lora_rank": 8, "iterations": 100},
        )
        self.assertEqual(cfg.qlora_bits, 4)
        self.assertEqual(cfg.optimizer_name, "adamw")
        self.assertEqual(cfg.lora_scale, 16.0)
        self.assertEqual(cfg.train_type, "dora")
        self.assertEqual(cfg.min_snr_gamma, 5.0)
        self.assertEqual(cfg.prior_loss_weight, 1.0)
        self.assertEqual(cfg.early_stop_patience, 3)
        self.assertTrue(cfg.fuse_adapters)

    def test_parse_lora_train_runtime_config_default_scale(self) -> None:
        from backend.engine.training.lora_train_runtime import (
            DEFAULT_LORA_SCALE,
            parse_lora_train_runtime_config,
        )

        cfg = parse_lora_train_runtime_config({}, defaults={"lora_rank": 16, "iterations": 100})
        self.assertEqual(cfg.lora_scale, DEFAULT_LORA_SCALE)

    def test_latent_cache_fingerprint(self) -> None:
        from backend.engine.training.latent_cache import LatentCache

        cache = LatentCache("/tmp/dq_lora_test")
        cache.begin(
            dataset_id="ds_test",
            n_pairs=3,
            num_augmentations=5,
            resolution=(768, 768),
            family="z_image",
            tensor_keys=["latent", "cap"],
        )
        self.assertEqual(cache._manifest["n_samples"], 0)

    def test_latent_cache_restores_batch_dim(self) -> None:
        import tempfile
        from pathlib import Path

        import mlx.core as mx

        from backend.engine.training.latent_cache import LatentCache

        with tempfile.TemporaryDirectory() as td:
            cache = LatentCache(Path(td))
            cache.begin(
                dataset_id="ds_batch",
                n_pairs=1,
                num_augmentations=1,
                resolution=(512, 512),
                family="z_image",
                tensor_keys=["latent", "cap"],
            )
            latent = mx.zeros((16, 64, 64), dtype=mx.bfloat16)
            cap = mx.zeros((1, 8, 2560), dtype=mx.bfloat16)
            cache.write_sample(0, {"latent": latent, "cap": cap})
            cache.finalize()

            x0, loaded_cap = cache.sample_z_image(0)
            self.assertEqual(tuple(x0.shape), (1, 16, 64, 64))
            self.assertEqual(tuple(loaded_cap.shape), (1, 8, 2560))

    def test_min_snr_weight_disabled(self) -> None:
        import mlx.core as mx
        from backend.engine.training.dit_training_loss import min_snr_weight

        sigma = mx.array([0.5])
        w = min_snr_weight(sigma, 0.0)
        self.assertEqual(float(w.item()), 1.0)

    def test_load_lora_train_config_json(self) -> None:
        import json
        import tempfile
        from pathlib import Path

        from backend.engine.training.lora_train_config import load_lora_train_config_file

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "train.json"
            path.write_text(json.dumps({"qlora_bits": 4, "grad_checkpoint": True}), encoding="utf-8")
            cfg = load_lora_train_config_file(path)
            self.assertEqual(cfg["qlora_bits"], 4)
            self.assertTrue(cfg["grad_checkpoint"])

    def test_lora_module_keys_limit_injection(self) -> None:
        from backend.engine.runtime.mlx import MLXContext
        from backend.engine.config.model_configs import ZImageConfig
        from backend.engine.families.z_image.transformer import ZImageTransformer
        from backend.engine.training.lora_layers import (
            apply_lora_to_zimage_dit,
            iter_lora_linears,
        )

        ctx = MLXContext()
        model = ZImageTransformer(ZImageConfig(), ctx)
        apply_lora_to_zimage_dit(
            model,
            rank=4,
            lora_blocks=1,
            module_keys=["to_q", "to_k"],
        )
        all_layers = iter_lora_linears(model)
        self.assertGreater(len(all_layers), 0)
        self.assertLess(len(all_layers), 40)


class LoraTrainingPresetsTests(unittest.TestCase):
    def test_trainable_base_models_include_z_image_and_qwen(self) -> None:
        from backend.engine.training.presets import TRAINABLE_BASE_MODELS

        self.assertIn("z-image", TRAINABLE_BASE_MODELS)
        self.assertIn("qwen-image", TRAINABLE_BASE_MODELS)
        self.assertIn("flux1-dev", TRAINABLE_BASE_MODELS)
        self.assertIn("z-image-turbo", TRAINABLE_BASE_MODELS)

    def test_resolve_preset_by_base_model(self) -> None:
        from backend.engine.training.crop import resolve_training_resolution
        from backend.engine.training.presets import resolve_preset

        flux = resolve_preset("standard", base_model="flux1-dev")
        zimg = resolve_preset("standard", base_model="z-image")
        zturbo = resolve_preset("standard", base_model="z-image-turbo")
        qwen = resolve_preset("standard", base_model="qwen-image")
        self.assertEqual(flux["lora_rank"], 16)
        self.assertEqual(zimg["lora_rank"], 16)
        self.assertEqual(zturbo["lora_rank"], 16)
        self.assertEqual(zturbo["guidance"], 0.0)
        self.assertEqual(zturbo["timestep_low"], 1)
        self.assertEqual(zturbo["timestep_high"], 9)
        self.assertEqual(zturbo["timestep_bias"], "uniform")
        self.assertEqual(zturbo["min_snr_gamma"], 5.0)
        self.assertEqual(zturbo["turbo_infer_steps"], 9)
        self.assertNotIn("lora_module_keys", zturbo)
        self.assertEqual(zturbo["val_split"], 0.1)
        self.assertEqual(zturbo["val_every"], 100)
        self.assertEqual(zturbo["prior_loss_weight"], 0.0)
        self.assertEqual(qwen["lora_rank"], 16)
        self.assertEqual(
            resolve_training_resolution("z-image-turbo", zturbo, preset="standard"),
            (512, 512),
        )
        self.assertEqual(
            resolve_training_resolution("z-image", zimg, preset="standard"),
            (512, 512),
        )
        self.assertEqual(
            resolve_training_resolution("qwen-image", qwen, preset="standard"),
            (512, 512),
        )

    def test_z_image_turbo_standard_preset_mflux_aligned(self) -> None:
        from backend.engine.training.presets import resolve_preset

        turbo = resolve_preset("standard", base_model="z-image-turbo")
        self.assertEqual(turbo["lora_rank"], 16)
        self.assertNotIn("lora_module_keys", turbo)
        self.assertEqual(turbo["timestep_low"], 1)
        self.assertEqual(turbo["learning_rate"], 1e-4)
        self.assertEqual(resolve_preset("mflux", base_model="z-image-turbo"), turbo)

    def test_merge_training_request_config_keeps_preset(self) -> None:
        from backend.core.contracts import LoraTrainingRequest
        from backend.engine.training.crop import resolve_training_resolution
        from backend.engine.training.presets import merge_training_request_config, resolve_preset

        preset = resolve_preset("quick", base_model="z-image")
        req = LoraTrainingRequest(
            base_model="z-image",
            dataset_id="ds_test",
            progress_prompt="test",
            preset="quick",
        )
        cfg = merge_training_request_config(req, preset)
        self.assertEqual(cfg["iterations"], 600)
        self.assertEqual(cfg["lora_rank"], 16)
        self.assertEqual(
            resolve_training_resolution("z-image", cfg, preset="quick"),
            (512, 512),
        )


class ZImageTurboTrainingTests(unittest.TestCase):
    def test_turbo_training_sigmas_band(self) -> None:
        from backend.engine.runtime.mlx import MLXContext
        from backend.engine.training.dit_training_loss import sample_noisy_latent_turbo, turbo_training_sigmas

        ctx = MLXContext()
        sigmas = turbo_training_sigmas(ctx, infer_steps=9, width=768, height=1280)
        self.assertEqual(int(sigmas.shape[0]), 9)
        x0 = ctx.zeros((1, 16, 96, 48), dtype=ctx.bfloat16())
        x_t, eps, t = sample_noisy_latent_turbo(
            x0,
            ctx,
            infer_steps=9,
            timestep_low=4,
            timestep_high=9,
            width=768,
            height=1280,
        )
        self.assertEqual(x_t.shape, x0.shape)
        self.assertEqual(eps.shape, x0.shape)
        self.assertEqual(t.shape[0], 1)

    def test_turbo_low_bias_prefers_low_sigma(self) -> None:
        import mlx.core as mx

        from backend.engine.runtime.mlx import MLXContext
        from backend.engine.training.dit_training_loss import sample_noisy_latent_turbo

        ctx = MLXContext()
        x0 = ctx.zeros((256, 16, 96, 48), dtype=ctx.bfloat16())
        _, _, t_low = sample_noisy_latent_turbo(
            x0,
            ctx,
            infer_steps=8,
            timestep_low=5,
            timestep_high=8,
            width=768,
            height=1280,
            timestep_bias="low",
        )
        _, _, t_uni = sample_noisy_latent_turbo(
            x0,
            ctx,
            infer_steps=8,
            timestep_low=5,
            timestep_high=8,
            width=768,
            height=1280,
            timestep_bias="uniform",
        )
        self.assertLess(float(mx.mean(t_low)), float(mx.mean(t_uni)))

    def test_apply_lora_recursive_wraps_matching_linears(self) -> None:
        import mlx.nn as nn

        from backend.engine.training.lora_layers import (
            _apply_lora_recursive,
            iter_lora_linears_with_paths,
            LoRALinear,
        )

        class Block:
            def __init__(self):
                self.to_q = nn.Linear(8, 4, bias=False)
                self.to_k = nn.Linear(8, 4, bias=False)

        block = Block()
        _apply_lora_recursive(
            block,
            rank=4,
            module_keys=["to_q", "to_k"],
            adapter_cls=LoRALinear,
        )
        self.assertIsInstance(block.to_q, LoRALinear)
        self.assertIsInstance(block.to_k, LoRALinear)

    def test_training_assistant_factors_shape_and_forward(self) -> None:
        import mlx.core as mx
        import mlx.nn as nn

        from backend.engine.training.lora_layers import LoRALinear

        in_d, out_d, rank = 256, 1024, 32
        down = mx.random.normal((rank, in_d))
        up = mx.random.normal((out_d, rank))
        base = nn.Linear(in_d, out_d, bias=False)
        lora = LoRALinear.from_base(base, r=4)
        lora.attach_frozen_assistant(down, up, alpha=float(rank))
        self.assertEqual(tuple(lora.assistant_lora_a.shape), (in_d, rank))
        self.assertEqual(tuple(lora.assistant_lora_b.shape), (rank, out_d))
        x = mx.random.normal((1, in_d))
        out = lora._assistant_contribution(x)
        self.assertEqual(tuple(out.shape), (1, out_d))

    def test_training_assistant_does_not_mutate_base_weight(self) -> None:
        import mlx.core as mx
        import mlx.nn as nn

        from backend.engine.training.lora_layers import LoRALinear

        base = nn.Linear(8, 4)
        base.weight = mx.ones((4, 8))
        lora = LoRALinear.from_base(base, r=4, scale=1.0)
        w_before = mx.array(lora.linear.weight)
        lora.assistant_lora_a = mx.ones((8, 2)) * 0.01
        lora.assistant_lora_b = mx.ones((2, 4)) * 0.01
        lora.assistant_scale = 0.5
        lora.assistant_enabled = True
        self.assertTrue(mx.allclose(lora.linear.weight, w_before))
        x = mx.ones((1, 8))
        with_assistant = lora(x)
        lora.assistant_enabled = False
        without_assistant = lora(x)
        self.assertFalse(mx.allclose(with_assistant, without_assistant))

    def test_training_adapter_path_prefers_local(self) -> None:
        import tempfile
        from pathlib import Path

        from backend.engine.training.z_image_turbo_adapter import (
            LOCAL_ADAPTER_REL,
            TRAINING_ADAPTER_FILE,
            resolve_zimage_turbo_training_adapter_path,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            local = root / LOCAL_ADAPTER_REL
            local.parent.mkdir(parents=True, exist_ok=True)
            local.write_bytes(b"not-a-real-safetensors")
            resolved = resolve_zimage_turbo_training_adapter_path(root)
            self.assertEqual(resolved, local)
            self.assertEqual(resolved.name, TRAINING_ADAPTER_FILE)


class LoraTrainingCropTests(unittest.TestCase):
    def test_align_resolution_to_vae_grid(self) -> None:
        from backend.engine.training.crop import resolve_training_resolution

        self.assertEqual(
            resolve_training_resolution("qwen-image", {"resolution": [770, 770]}),
            (768, 768),
        )
        self.assertEqual(
            resolve_training_resolution("flux1-dev", {"resolution": [515, 515]}),
            (512, 512),
        )

    def test_prepare_training_rgb_image_center_crop(self) -> None:
        from backend.engine.training.crop import prepare_training_rgb_image
        import tempfile
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "wide.png"
            Image.new("RGB", (800, 400), color=(255, 0, 0)).save(path)
            arr, (w, h) = prepare_training_rgb_image(
                path, "z-image", {}, preset="quick"
            )
            self.assertEqual((w, h), (512, 512))
            self.assertEqual(arr.shape[0], 512)
            self.assertEqual(arr.shape[1], 512)

    def test_is_heif_payload_detects_apple_heic(self) -> None:
        from backend.engine.training.dataset_store import _is_heif_payload

        header = bytes.fromhex("000000186674797068656963")
        self.assertTrue(_is_heif_payload(header + b"\x00" * 32))
        self.assertFalse(_is_heif_payload(b"\x89PNG\r\n\x1a\n"))

    def test_open_rgb_image_heic_misnamed_png(self) -> None:
        heic_path = Path(
            "/Users/nil.luo/Workspace/studio-workspace/datasets/"
            "ds_6e173d161fb2ac33/images/20.HEIC.png"
        )
        if not heic_path.is_file():
            self.skipTest("fixture HEIC dataset image not present")
        from backend.engine.training.dataset_store import open_rgb_image

        img = open_rgb_image(heic_path)
        self.assertEqual(img.mode, "RGB")
        self.assertGreater(img.size[0], 0)

    def test_portrait_resize_keeps_upper_region(self) -> None:
        from backend.engine.training.dataset_store import resize_rgb_image
        import tempfile
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "portrait.png"
            img = Image.new("RGB", (600, 900))
            for y in range(900):
                for x in range(600):
                    if y < 300:
                        img.putpixel((x, y), (255, 0, 0))
                    else:
                        img.putpixel((x, y), (0, 0, 255))
            img.save(path)
            arr = resize_rgb_image(path, (512, 512))
            self.assertEqual(arr.shape[:2], (512, 512))
            red_rows = sum(1 for y in range(512) if arr[y, 256, 0] > 0.5)
            self.assertGreater(red_rows, 80, "portrait crop should retain upper (face) region")

    def test_resolve_dreambooth_caption_prefers_progress_prompt(self) -> None:
        from backend.engine.training.dataset_store import resolve_dreambooth_caption
        from pathlib import Path

        pairs = [(Path("a.jpg"), "auto caption one"), (Path("b.jpg"), "auto caption two")]
        cap = resolve_dreambooth_caption(
            pairs,
            progress_prompt="A photo of sks person",
            dataset_meta={"default_prompt": "other"},
        )
        self.assertEqual(cap, "A photo of sks person")

    def test_resolve_dreambooth_caption_injects_trigger_word(self) -> None:
        from backend.engine.training.dataset_store import resolve_dreambooth_caption
        from pathlib import Path

        pairs = [(Path("a.jpg"), "caption")]
        cap = resolve_dreambooth_caption(
            pairs,
            progress_prompt="a beautiful portrait",
            dataset_meta={"trigger_word": "sks"},
        )
        self.assertIn("sks", cap.lower())
        self.assertIn("beautiful portrait", cap)

    def test_resolve_per_image_captions_keeps_row_text(self) -> None:
        from backend.engine.training.dataset_store import resolve_per_image_captions
        from pathlib import Path

        pairs = [
            (Path("a.jpg"), "陈钰琪，白色连衣裙"),
            (Path("b.jpg"), "陈钰琪，古装"),
        ]
        out = resolve_per_image_captions(
            pairs,
            progress_prompt="陈钰琪",
            dataset_meta={"trigger_word": "", "default_prompt": "陈钰琪"},
        )
        self.assertEqual(out[0][1], "陈钰琪，白色连衣裙")
        self.assertEqual(out[1][1], "陈钰琪，古装")

    def test_collect_lora_safetensors_remaps_for_qwen_image(self) -> None:
        from backend.engine.runtime.mlx import MLXContext
        from backend.engine.config.model_configs import QwenImageConfig
        from backend.engine.families.qwen.transformer import QwenImageTransformer
        from backend.engine.families.qwen.weights import remap_qwen_lora_keys
        from backend.engine.training.lora_layers import (
            apply_lora_to_qwen_dit,
            collect_lora_safetensors,
            prepare_dit_for_lora_training,
        )
        from backend.engine.training.qwen_image_dreambooth_mlx import _validate_saved_lora
        import mlx.core as mx
        import tempfile
        from pathlib import Path

        ctx = MLXContext()
        model = QwenImageTransformer(QwenImageConfig(), ctx)
        _, train_module = prepare_dit_for_lora_training(
            model,
            apply_lora_to_qwen_dit,
            rank=4,
            lora_blocks=1,
        )
        from backend.engine.training.qwen_image_dreambooth_mlx import _strip_dit_lora_paths

        _strip_dit_lora_paths(train_module)
        weights = collect_lora_safetensors(train_module, rank=4)
        weights.pop("lora_rank", None)
        self.assertTrue(
            any(k.startswith("transformer_blocks.") and ".lora_A.weight" in k for k in weights),
            "expected DiT module paths in exported LoRA keys",
        )
        remapped = remap_qwen_lora_keys(weights)
        self.assertGreater(len(remapped), 0)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "adapter.safetensors"
            mx.save_safetensors(str(path), weights)
            _validate_saved_lora(path, lora_blocks=1)


class ZImageLoraExportTests(unittest.TestCase):
    def test_collect_lora_safetensors_remaps_for_z_image(self) -> None:
        from backend.engine.runtime.mlx import MLXContext
        from backend.engine.config.model_configs import ZImageConfig
        from backend.engine.families.z_image.transformer import ZImageTransformer
        from backend.engine.families.z_image.weights import remap_zimage_lora_keys
        from backend.engine.training.lora_layers import (
            apply_lora_to_zimage_dit,
            collect_lora_safetensors,
            prepare_dit_for_lora_training,
        )
        from backend.engine.training.z_image_dreambooth_mlx import _validate_saved_lora
        import mlx.core as mx
        import tempfile
        from pathlib import Path

        ctx = MLXContext()
        model = ZImageTransformer(ZImageConfig(), ctx)
        _, train_module = prepare_dit_for_lora_training(
            model,
            apply_lora_to_zimage_dit,
            rank=4,
            lora_blocks=1,
        )
        weights = collect_lora_safetensors(train_module, rank=4)
        weights.pop("lora_rank", None)
        self.assertTrue(
            any(k.startswith("layers.") and ".lora_A.weight" in k for k in weights),
            "expected DiT module paths in exported LoRA keys",
        )
        remapped = remap_zimage_lora_keys(weights)
        self.assertGreater(len(remapped), 0)
        self.assertTrue(all("." in tgt for tgt in remapped))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "adapter.safetensors"
            mx.save_safetensors(str(path), weights)
            _validate_saved_lora(path)


class LoraSearchQueryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import json
        from pathlib import Path

        raw = json.loads(Path("default_config/models_registry.json").read_text())
        cls.registry = raw.get("models") or raw

    def test_build_search_query_keeps_user_keywords(self) -> None:
        from backend.services.lora_search import build_search_query

        self.assertEqual(
            build_search_query(self.registry, "z-image-turbo", "StarFace"),
            "StarFace",
        )

    def test_build_search_query_uses_registry_terms(self) -> None:
        from backend.services.lora_search import build_search_query, resolve_lora_browse_queries

        queries = resolve_lora_browse_queries(self.registry, "z-image-turbo")
        self.assertIn("Z-Image-Turbo lora", queries)
        self.assertIn("z-image lora", queries)
        self.assertEqual(
            build_search_query(self.registry, "z-image-turbo", ""),
            queries[0],
        )


if __name__ == "__main__":
    unittest.main()
