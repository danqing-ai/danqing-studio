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
from backend.engine.common.text_encoders import T5Encoder
from backend.engine._transformer_registry import (
    get_transformer_class as _get_transformer_class,
    get_weight_remap as _get_weight_remap,
    get_text_encoder as _get_text_encoder,
)
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
        if spec is None:
            return fallback
        if isinstance(spec, dict):
            return spec.get("default", fallback)
        return spec  # 直接值（list / int / str 等），非 dict 包裹

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

    def run(
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
            raise RuntimeError("SeedVR2 is an upscaling model; not for txt2img.")
        if family == "qwen_image":
            raise RuntimeError("Qwen-Image requires Qwen-VL conditioning; not wired.")

        # ── 注册表驱动的参数注入 ──
        for param_key in ("text_encoder_out_layers", "vae_scale", "enable_thinking"):
            val = self._registry_scalar_default(entry, param_key, None)
            if val is not None:
                setattr(config, param_key, val)

        # 注册表驱动的 supports_guidance 覆盖（如 z-image-turbo）
        sg = self._registry_scalar_default(entry, "supports_guidance", None)
        if sg is not None:
            config.supports_guidance = bool(sg)

        if ctx_exec.cancel_token.is_cancelled():
            return None

        bundle_root = self._local_bundle_root(entry, version_key or None)

        steps_default = self._registry_scalar_default(entry, "steps", 4)
        guidance_default = self._registry_scalar_default(entry, "guidance", 0.0)
        scheduler_registry = self._registry_scalar_default(entry, "scheduler", None)
        scheduler_request = request.scheduler or request.metadata.get("scheduler") if request.metadata else None
        scheduler_default = scheduler_request or scheduler_registry or "flow_match_euler"

        steps = int(request.steps) if request.steps is not None else int(steps_default)
        steps = max(1, steps)
        guidance = float(request.guidance) if request.guidance is not None else float(guidance_default)
        if not getattr(config, "supports_guidance", True):
            guidance = 0.0

        # 1. 文本编码（由 config.encoder_type 驱动，零 family 分支）
        txt_embeds = None
        neg_embeds = None
        encoder_type = getattr(config, "encoder_type", "t5")
        if request.prompt and encoder_type != "t5":
            if bundle_root is None:
                raise RuntimeError(
                    f"Model {model_key!r} has no installed bundle at local_path "
                    f"(version={version_key or 'default'}); cannot load text encoder."
                )
            txt_embeds = self._text_encode(request.prompt, bundle_root=bundle_root, encoder_type=encoder_type, config=config)
            if getattr(config, "supports_guidance", False) and guidance > 1.0:
                neg_txt = request.negative_prompt.strip() if request.negative_prompt else " "
                neg_embeds = self._text_encode(neg_txt, bundle_root=bundle_root, encoder_type=encoder_type, config=config)
        elif request.prompt and config.text_dim > 0:
            enc = T5Encoder(self.ctx, "google/t5-v1_1-xxl")
            txt_embeds = enc.encode([request.prompt])

        if ctx_exec.cancel_token.is_cancelled():
            return None

        # 2. 模型加载
        model = self._load_model(family, config, entry, version_key or None)
        if model is None:
            raise RuntimeError(f"Failed to load model: {model_key}")

        # ── Hook ①: 权重加载后（LoRA / Adapter 合并）──
        model.after_load_weights(bundle_root=str(bundle_root) if bundle_root else None)

        # ── Hook ②: 条件准备（ControlNet 编码控制图）──
        extra_cond = model.prepare_conditioning(request, bundle_root=str(bundle_root) if bundle_root else None)

        # 3. 调度器（注册表默认 + 请求参数，零 family 分支）
        scheduler = get_scheduler(scheduler_default, ctx=self.ctx)
        vae_scale = getattr(config, "vae_scale", 8)
        # image_seq_len 用于 sigma shift，与 mflux Config 一致：固定 //16
        image_seq_len = (h // 16) * (w // 16)
        timesteps = scheduler.set_timesteps(steps, image_seq_len=image_seq_len,
                                             image_width=int(w), image_height=int(h),
                                             requires_sigma_shift=self._registry_scalar_default(entry, "requires_sigma_shift", False))
        sigmas = getattr(scheduler, 'sigmas', None)

        # 使用 seed 创建确定性 latent
        latent_shape = (1, config.in_channels, h // vae_scale, w // vae_scale)
        if seed is not None:
            latents = self.ctx.seeded_randn(latent_shape, seed, dtype=self.ctx.float32())
        else:
            latents = self.ctx.randn(latent_shape, dtype=self.ctx.float32())

        # ── Hook ③: 去噪前（ControlNet 信号注入 / latent 修改）──
        latents, extra_cond = model.before_denoise(latents, timesteps, sigmas, **extra_cond)

        # ------------------------------------------------------------------
        # 4. 去噪循环 — 完全通用，模型自己处理 timestep 转换和特殊参数
        # ------------------------------------------------------------------
        for i, t in enumerate(timesteps):
            if ctx_exec.cancel_token.is_cancelled():
                return None

            # 统一的模型调用接口：传入原始 timestep 索引 + sigmas，由模型自己转换
            model_kwargs = {"txt_embeds": txt_embeds} if txt_embeds is not None else {}
            model_kwargs.update(extra_cond)
            if sigmas is not None:
                model_kwargs["sigmas"] = sigmas

            noise_pred = model(latents, t, **model_kwargs)

            # CFG — 模型族自己决定是否支持
            if neg_embeds is not None and getattr(config, "supports_guidance", False):
                noise_neg = model(latents, t, txt_embeds=neg_embeds, **({"sigmas": sigmas} if sigmas is not None else {}))
                noise_pred = noise_pred + guidance * (noise_pred - noise_neg)

            latents = scheduler.step(noise_pred, t, latents)

            # ── Hook ④: 每步回调（动态条件 / 日志）──
            model.step_callback(i, latents, noise_pred)

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

    def _text_encode(self, text: str, *, bundle_root: Path, encoder_type: str, config: Any) -> Any:
        """文本编码 — 由注册表 encoder_type 路由到具体实现。"""
        enc_dir = bundle_root / "text_encoder"
        tok_dir = bundle_root / "tokenizer"
        if not tok_dir.exists():
            tok_dir = enc_dir
        if not enc_dir.exists():
            enc_dir = bundle_root
            tok_dir = bundle_root

        enc_cls = _get_text_encoder(encoder_type)
        if encoder_type == "z_image":
            out_layers = getattr(config, "text_encoder_out_layers", None)
            if out_layers is not None:
                out_layers = tuple(out_layers)
            enc = enc_cls(self.ctx, str(enc_dir), tokenizer_path=str(tok_dir),
                          hidden_state_layers=out_layers,
                          enable_thinking=getattr(config, "enable_thinking", False))
        else:
            enc = enc_cls(self.ctx, str(enc_dir), tokenizer_path=str(tok_dir))
        return enc.encode([text])

    def _load_model(self, family: str, config, entry, version_key: str | None):
        trans_cls = _get_transformer_class(family)
        model = trans_cls(config, self.ctx)
        remap_fn = _get_weight_remap(family)

        bundle_root = self._local_bundle_root(entry, version_key)
        tp = (bundle_root / "transformer") if bundle_root else None
        if tp is None or not tp.exists():
            return None

        w = {}
        for sf in sorted(tp.glob("*.safetensors")):
            w.update(self.ctx.load_weights(str(sf)))
        if remap_fn:
            w = remap_fn(w)
        model.load_weights(list(w.items()), strict=False)
        self.ctx.eval(*[p for _, p in model.parameters()])
        return model

    def _vae_preprocess_special(self, latents, vae_weights, scaling_factor, shift_factor):
        """特殊 VAE 预处理 — flux2 风格（通过权重检测触发，非 family 硬编码）。"""
        ctx = self.ctx

        bn_mean = vae_weights.get("bn.running_mean", ctx.zeros((128,))).reshape(1, -1, 1, 1)
        bn_var = vae_weights.get("bn.running_var", ctx.ones((128,))).reshape(1, -1, 1, 1)
        latents = latents * ctx.sqrt(bn_var + 1e-4) + bn_mean

        B, C_, H_, W_ = latents.shape
        latents = latents.reshape(B, C_ // 4, 2, 2, H_, W_)
        latents = ctx.permute(latents, (0, 1, 4, 2, 5, 3))
        latents = latents.reshape(B, C_ // 4, H_ * 2, W_ * 2)

        latents = (latents / scaling_factor) + shift_factor
        latents = ctx.permute(latents, (0, 2, 3, 1))

        pw = vae_weights.get("post_quant_conv.weight")
        pb = vae_weights.get("post_quant_conv.bias")
        if pw is not None and pb is not None:
            latents = ctx.conv2d(latents, ctx.permute(pw, (0, 2, 3, 1)), stride=1, padding=0)
            latents = latents + pb.reshape(1, 1, 1, -1)

        latents = ctx.permute(latents, (0, 3, 1, 2))
        return latents

    def _vae_decode(self, latents, entry, version_key):
        """VAE 解码 latent → PIL Image。"""
        ctx = self.ctx
        import numpy as np
        from PIL import Image
        from backend.engine.common._vae import VAEDecoder

        bundle_root = self._local_bundle_root(entry, version_key)
        vae_dir = (bundle_root / "vae") if bundle_root else None

        scaling_factor = 1.0
        shift_factor = 0.0
        latent_cfg = 16
        if vae_dir and (vae_dir / "config.json").exists():
            import json
            with open(vae_dir / "config.json") as f:
                vae_cfg = json.load(f)
            scaling_factor = vae_cfg.get("scaling_factor", 1.0)
            shift_factor = vae_cfg.get("shift_factor", 0.0)
            latent_cfg = vae_cfg.get("latent_channels", 16)

        if latents.ndim == 3:
            B, seq_len, channels = latents.shape
            latent_h = int(seq_len ** 0.5)
            latent_w = seq_len // latent_h
            latents = latents.reshape(B, latent_h, latent_w, channels).transpose(0, 3, 1, 2)

        vae_weights = {}
        if vae_dir and vae_dir.exists():
            for sf in sorted(vae_dir.glob("*.safetensors")):
                vae_weights.update(ctx.load_weights(str(sf)))

        if "bn.running_mean" in vae_weights or "post_quant_conv.weight" in vae_weights:
            latents = self._vae_preprocess_special(latents, vae_weights, scaling_factor, shift_factor)
            scaling_factor = 1.0
            shift_factor = 0.0
            ci = vae_weights.get("decoder.conv_in.weight", ctx.zeros((1,))).shape[0] if "decoder.conv_in.weight" in vae_weights else 16
            latent_cfg = ci

        C = latents.shape[1] if latents.ndim >= 4 else 16
        vae = VAEDecoder(latent_channels=C, ctx=ctx, scaling_factor=scaling_factor, shift_factor=shift_factor)
        if vae_weights:
            from backend.engine.common.weights import remap_vae_weights
            decoder_w = remap_vae_weights(vae_weights)
            vae.load_weights(list(decoder_w.items()), strict=False)
        image = vae.forward(latents)

        if hasattr(image, 'numpy'):
            image = image.numpy()
        else:
            image = np.array(image)
        if image.ndim == 4:
            image = image[0]
        if image.shape[0] <= 4:
            image = np.transpose(image, (1, 2, 0))
        # 归一化到 0-255（VAE 输出 [-1, 1] → [0, 255]）
        image = (image + 1) / 2 * 255
        image = image.clip(0, 255).astype(np.uint8)
        return Image.fromarray(image)
