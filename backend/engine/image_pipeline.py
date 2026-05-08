"""
ImagePipeline — 图像请求 → 模型推理 → 资产落盘。

MLX 操作（文本编码+模型加载+去噪+VAE解码）统一在单线程 executor 中执行。
进度回调和结果处理在事件循环线程。
"""
from __future__ import annotations

import random
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from backend.core.contracts import (
    EngineResult, ExecutionContext, ImageGenerationRequest,
    ImageEditRequest, ImageUpscaleRequest,
    LogEvent, ProgressEvent, parse_model_version, parse_size,
)
from backend.engine.common.cache import ModelCache
from backend.engine.common.schedulers import get_scheduler
from backend.engine.common.text_encoders import T5Encoder, Qwen3TextEncoder
from backend.engine.common.weights import remap_zimage_weights, remap_flux2_weights
from backend.engine.config.model_configs import get_config_class
from backend.engine.runtime._base import RuntimeContext


class ImagePipeline:
    """图像生成管线。"""

    def __init__(
        self,
        ctx: RuntimeContext,
        model_registry: Any,
        asset_store: Any,
        model_cache: ModelCache | None = None,
        project_root: Path | None = None,
    ):
        self.ctx = ctx
        self._registry = model_registry
        self._asset_store = asset_store
        self._cache = model_cache
        self._project_root = project_root or Path.cwd()

    def _resolve_path(self, local_path: str) -> Path:
        p = Path(local_path)
        if p.is_absolute():
            return p
        return (self._project_root / local_path).resolve()

    @staticmethod
    def _registry_scalar_default(entry, key: str, fallback):
        spec = (entry.parameters or {}).get(key)
        if isinstance(spec, dict) and "default" in spec:
            return spec["default"]
        return fallback

    def _resolve_version_block(self, entry, version_key: str | None) -> dict | None:
        raw = getattr(entry, "raw", {}) or {}
        versions = raw.get("versions") or {}
        if version_key and version_key in versions and isinstance(versions[version_key], dict):
            return versions[version_key]
        for vinfo in versions.values():
            if isinstance(vinfo, dict) and vinfo.get("default"):
                return vinfo
        return None

    def _local_bundle_root(self, entry, version_key: str | None) -> Path | None:
        block = self._resolve_version_block(entry, version_key)
        if not block:
            return None
        lp = (block.get("local_path") or "").strip()
        if not lp:
            return None
        path = self._resolve_path(lp)
        return path if path.exists() else None

    def _scheduler_name_for_family(self, family: str) -> str:
        """调度器与 Transformer 的时间嵌入约定一致（见 mflux 各 CLI 默认值）。

        ``Flux2Transformer`` / ``ZImageTransformer``：timestep 为 [0,1]，模型内再缩放。
        ``FIBOTransformer`` / ``Flux1Transformer``：``TimestepEmbedding`` 配合 **linear** 的 0–1000 标度。
        """
        if family in ("z_image", "flux2", "longcat"):
            return "flow_match_euler"
        return "linear"

    def run_mlx(
        self,
        request: ImageGenerationRequest,
        ctx_exec: ExecutionContext,
        *,
        on_progress: Callable | None = None,
        on_log: Callable | None = None,
    ):
        """同步执行 MLX 管线（调度器 worker 内调用）。

        取消：各步检查 ``ctx_exec.cancel_token``，取消时返回 ``None``。
        输出：写入 ``ctx_exec.work_dir``（与调度器为该任务分配的工作目录一致）。

        Returns: ``(output_path, metadata_dict)`` 或 ``None``（已取消）
        """
        model_key, version_key = parse_model_version(request.model)
        w, h = parse_size(request.size)
        seed = request.seed if request.seed is not None else random.randint(0, 2 ** 32 - 1)
        entry = self._registry.require(model_key)
        config_cls = get_config_class(entry.family)
        config = config_cls()
        family = getattr(entry, "family", "flux1")

        if family == "seedvr2":
            raise RuntimeError(
                "SeedVR2 is an upscaling model in this registry; "
                "standard ImagePipeline txt2img does not apply. Use an upscale workflow when implemented."
            )
        if family == "qwen_image":
            raise RuntimeError(
                "Qwen-Image requires Qwen-VL text/image conditioning; only T5/Qwen3 paths are wired here. "
                "Use flux1, flux2, z_image, fibo, or longcat."
            )

        # mflux: Z-Image-Turbo 不用 CFG；基础 Z-Image 用 flow_match_euler_discrete + CFG
        if family == "z_image" and model_key == "z-image-turbo":
            config.supports_guidance = False

        if ctx_exec.cancel_token.is_cancelled():
            return None

        bundle_root = self._local_bundle_root(entry, version_key or None)

        steps_default = self._registry_scalar_default(entry, "steps", None)
        guidance_default = self._registry_scalar_default(entry, "guidance", None)
        if steps_default is None:
            steps_default = 50 if family in ("z_image", "longcat") else 4
        if guidance_default is None:
            guidance_default = 0.0 if not getattr(config, "supports_guidance", True) else (
                4.0 if family == "z_image" else 3.5
            )

        steps = int(request.steps) if request.steps is not None else int(steps_default)
        steps = max(1, steps)
        guidance = float(request.guidance) if request.guidance is not None else float(guidance_default)
        if family == "z_image" and not getattr(config, "supports_guidance", True):
            guidance = 0.0

        # 1. 文本编码
        txt_embeds = None
        neg_embeds = None
        encoder_type = getattr(config, "encoder_type", "t5")
        if request.prompt and encoder_type in ("qwen3", "qwen2.5_vl"):
            if bundle_root is None:
                raise RuntimeError(
                    f"Model {model_key!r} has no installed bundle at registry local_path "
                    f"(version={version_key or 'default'}); cannot load text encoder."
                )
            if encoder_type == "qwen2.5_vl":
                txt_embeds = self._qwen25vl_encode(request.prompt, bundle_root=bundle_root)
            else:
                txt_embeds = self._qwen3_encode(request.prompt, family=family, bundle_root=bundle_root)
            use_cfg = (
                family in ("z_image", "longcat")
                and getattr(config, "supports_guidance", True)
                and guidance > 1.0
            )
            if use_cfg:
                neg_txt = request.negative_prompt.strip() if request.negative_prompt else " "
                if encoder_type == "qwen2.5_vl":
                    neg_embeds = self._qwen25vl_encode(neg_txt, bundle_root=bundle_root)
                else:
                    neg_embeds = self._qwen3_encode(neg_txt, family=family, bundle_root=bundle_root)
        elif request.prompt and config.text_dim > 0:
            enc = T5Encoder(self.ctx, "google/t5-v1_1-xxl")
            txt_embeds = enc.encode([request.prompt])

        if ctx_exec.cancel_token.is_cancelled():
            return None

        # 2. 模型加载
        model = self._load_model(family, config, entry, version_key or None)
        if model is None:
            raise RuntimeError(f"Failed to load model: {model_key}")

        # 3. 调度器（见 _scheduler_name_for_family）
        scheduler = get_scheduler(self._scheduler_name_for_family(family), ctx=self.ctx)
        timesteps = scheduler.set_timesteps(steps)
        sigmas = getattr(scheduler, 'sigmas', None)

        # 使用 seed 创建确定性 latent
        import mlx.core as mx
        latent_shape = (1, config.in_channels, h // 8, w // 8)
        if seed is not None:
            key = mx.random.key(seed)
            latents = mx.random.normal(latent_shape, dtype=mx.float32, key=key)
        else:
            latents = self.ctx.randn(latent_shape, dtype=self.ctx.float32())

        if family == "z_image":
            for i, t in enumerate(timesteps):
                if ctx_exec.cancel_token.is_cancelled():
                    return None
                noise_pred = model(latents, t, txt_embeds=txt_embeds, sigmas=sigmas)
                if neg_embeds is not None:
                    noise_neg = model(latents, t, txt_embeds=neg_embeds, sigmas=sigmas)
                    noise_pred = noise_pred + guidance * (noise_pred - noise_neg)
                latents = scheduler.step(noise_pred, t, latents)

                if on_progress:
                    on_progress((i + 1) / len(timesteps), i + 1, len(timesteps), None)
                if on_log:
                    on_log("info", f"Step {i+1}/{len(timesteps)}")
        else:
            for i, t in enumerate(timesteps):
                if ctx_exec.cancel_token.is_cancelled():
                    return None
                noise_pred = model(
                    latents,
                    t,
                    **({"txt_embeds": txt_embeds} if txt_embeds is not None else {}),
                )
                latents = scheduler.step(noise_pred, t, latents)

                if on_progress:
                    on_progress((i + 1) / len(timesteps), i + 1, len(timesteps), None)
                if on_log:
                    on_log("info", f"Step {i+1}/{len(timesteps)}")

        if ctx_exec.cancel_token.is_cancelled():
            return None

        # 5. VAE 解码
        image = self._vae_decode(latents, entry, version_key or None)

        if ctx_exec.cancel_token.is_cancelled():
            return None

        # 6. 保存（任务工作目录，见 TaskScheduler._work_dir）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        work = Path(ctx_exec.work_dir)
        work.mkdir(parents=True, exist_ok=True)
        out_path = work / f"{model_key}_{seed}_{timestamp}.png"
        if hasattr(image, 'save'):
            image.save(str(out_path))

        return str(out_path), {
            "model": request.model, "seed": seed,
            "prompt": request.prompt, "steps": steps,
            "guidance": guidance,
            "width": w, "height": h, "mime_type": "image/png",
        }

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    def _qwen3_encode(self, text: str, *, family: str = "", bundle_root: Path) -> Any:
        """Qwen3 文本编码 — z_image / flux2 系列。

        z_image 使用倒数第二层输出（与 mflux TextEncoder 一致）。
        flux2 使用最后一层输出（与 mflux Qwen3TextEncoder 一致）。
        """
        enc_dir = bundle_root / "text_encoder"
        tok_dir = bundle_root / "tokenizer"
        if not tok_dir.exists():
            tok_dir = enc_dir
        if not enc_dir.exists():
            enc_dir = bundle_root
            tok_dir = bundle_root

        enc = Qwen3TextEncoder(self.ctx, str(enc_dir), tokenizer_path=str(tok_dir))
        # z_image 兼容：使用倒数第二层
        if family == "z_image":
            enc.use_second_to_last = True
        return enc.encode([text])

    def _qwen25vl_encode(self, text: str, *, bundle_root: Path) -> Any:
        """Qwen2.5-VL 文本编码 — longcat 系列。"""
        from backend.engine.common.text_encoders_qwen25vl import Qwen25VLEncoder
        enc = Qwen25VLEncoder(str(bundle_root))
        return enc.encode([text])

    def _load_model(self, family: str, config, entry, version_key: str | None):
        import mlx.core as mx

        loaded = False
        if family == "z_image":
            from backend.engine.models.image.z_image import ZImageTransformer
            model = ZImageTransformer(config, self.ctx)
        elif family == "flux2":
            from backend.engine.models.image.flux2 import Flux2Transformer
            model = Flux2Transformer(config, self.ctx)
        elif family == "fibo":
            from backend.engine.models.image.fibo import FIBOTransformer
            model = FIBOTransformer(config, self.ctx)
        elif family == "longcat":
            from backend.engine.models.image.longcat import LongCatTransformer
            model = LongCatTransformer(config, self.ctx)
        else:
            from backend.engine.models.image.flux1 import Flux1Transformer
            model = Flux1Transformer(config, self.ctx)

        bundle_root = self._local_bundle_root(entry, version_key)
        tp = (bundle_root / "transformer") if bundle_root else None
        if tp is not None and tp.exists():
            w = {}
            for sf in sorted(tp.glob("*.safetensors")):
                w.update(dict(mx.load(str(sf))))
            if family == "z_image":
                w = remap_zimage_weights(w)
            elif family == "flux2":
                w = remap_flux2_weights(w)
            model.load_weights(list(w.items()), strict=False)
            mx.eval([p for _, p in model.parameters()])
            loaded = True

        if not loaded:
            raw = getattr(entry, "raw", {})
            versions = raw.get("versions", {})
            for vkey, vinfo in versions.items():
                if isinstance(vinfo, dict) and vinfo.get("default"):
                    lp = vinfo.get("local_path", "")
                    if lp:
                        root = self._resolve_path(lp)
                        tpath = root / "transformer"
                        if tpath.exists():
                            w = {}
                            for sf in sorted(tpath.glob("*.safetensors")):
                                w.update(dict(mx.load(str(sf))))
                            if family == "z_image":
                                w = remap_zimage_weights(w)
                            elif family == "flux2":
                                w = remap_flux2_weights(w)
                            model.load_weights(list(w.items()), strict=False)
                            mx.eval([p for _, p in model.parameters()])
                            loaded = True

        return model if loaded else None

    def _vae_decode(self, latents, entry, version_key):
        """VAE 解码 latent → PIL Image。"""
        import mlx.core as mx
        import numpy as np
        from PIL import Image
        from backend.engine.vae.image_vae import VAEDecoder

        bundle_root = self._local_bundle_root(entry, version_key)
        vae_dir = (bundle_root / "vae") if bundle_root else None

        # 加载 VAE 配置
        scaling_factor = 1.0
        shift_factor = 0.0
        if vae_dir and (vae_dir / "config.json").exists():
            import json
            with open(vae_dir / "config.json") as f:
                vae_cfg = json.load(f)
            scaling_factor = vae_cfg.get("scaling_factor", 1.0)
            shift_factor = vae_cfg.get("shift_factor", 0.0)

        # 创建 VAE decoder
        C = latents.shape[1] if latents.ndim >= 4 else 16
        vae = VAEDecoder(
            latent_channels=C,
            ctx=self.ctx,
            scaling_factor=scaling_factor,
            shift_factor=shift_factor,
        )

        # 加载 VAE 权重
        if vae_dir and vae_dir.exists():
            w = {}
            for sf in sorted(vae_dir.glob("*.safetensors")):
                w.update(dict(mx.load(str(sf))))
            from backend.engine.common.weights import remap_vae_weights
            w = remap_vae_weights(w)
            vae.load_weights(list(w.items()), strict=False)

        # 解码
        image = vae.forward(latents)

        # 转 numpy + PIL
        if isinstance(image, mx.array):
            image = np.array(image)
        # NCHW → NHWC
        if image.ndim == 4:
            image = image[0]  # 去掉 batch
        if image.shape[0] <= 4:  # CHW format
            image = np.transpose(image, (1, 2, 0))
        # 归一化到 0-255
        image = (image - image.min()) / (image.max() - image.min() + 1e-8) * 255
        image = image.clip(0, 255).astype(np.uint8)
        return Image.fromarray(image)
