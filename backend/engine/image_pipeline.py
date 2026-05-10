"""
ImagePipeline — image request → model inference → asset persistence.

MLX operations (text encoding + model loading + denoising + VAE decoding) are executed
in a single-threaded executor. Progress callbacks and result handling run in the event loop thread.
"""
from __future__ import annotations

import random
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np

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
    """Image generation pipeline."""

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
        return spec  # direct value (list / int / str etc.), not wrapped in dict

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

    @staticmethod
    def _align_hw_multiples(w0: int, h0: int, *, align: int) -> tuple[int, int]:
        """Width/height floored to multiples of ``align`` (at least ``align``)."""
        w = max(align, (w0 // align) * align)
        h = max(align, (h0 // align) * align)
        return w, h

    @staticmethod
    def _center_crop_pil(pil: Any, w: int, h: int) -> Any:
        from PIL import Image

        w0, h0 = pil.size
        left = max(0, (w0 - w) // 2)
        top = max(0, (h0 - h) // 2)
        box = (left, top, min(left + w, w0), min(top + h, h0))
        cropped = pil.crop(box)
        if cropped.size != (w, h):
            return cropped.resize((w, h), Image.Resampling.LANCZOS)
        return cropped

    def _pil_to_nchw_float01(self, pil: Any, w: int, h: int) -> Any:
        """Resize PIL RGB → float01 tensor ``[1,3,H,W]`` (NCHW)."""
        from PIL import Image

        if pil.size != (w, h):
            pil = pil.resize((w, h), Image.Resampling.LANCZOS)
        arr = np.asarray(pil.convert("RGB"), dtype=np.float32) / 255.0
        arr = arr[None, ...]
        t = self.ctx.array(arr)
        return self.ctx.permute(t, (0, 3, 1, 2))

    def _vae_latent_channels_from_bundle(self, entry, version_key: str | None) -> int:
        bundle_root = self._local_bundle_root(entry, version_key)
        vae_dir = (bundle_root / "vae") if bundle_root else None
        if vae_dir and (vae_dir / "config.json").exists():
            import json

            with open(vae_dir / "config.json") as f:
                cfg = json.load(f)
            lc = cfg.get("latent_channels")
            if lc is not None:
                return int(lc)
        vae_weights: dict[str, Any] = {}
        if vae_dir and vae_dir.exists():
            for sf in sorted(vae_dir.glob("*.safetensors")):
                vae_weights.update(self.ctx.load_weights(str(sf)))
        wkey = "encoder.conv_out.weight"
        if wkey in vae_weights:
            t = vae_weights[wkey]
            sh = getattr(t, "shape", ())
            if len(sh) >= 1:
                return int(sh[0]) // 2
        return 16

    def _vae_encode_tensor(
        self,
        image_nchw_f01: Any,
        entry,
        version_key: str | None,
        *,
        on_log: Callable | None = None,
    ) -> Any:
        """Encode ``[1,3,H,W]`` float01 → ``[1,C,H/8,W/8]`` latent (mean, inference)."""
        from backend.engine.common._vae_encoder import VAEEncoder
        from backend.engine.common.weights._vae import prepare_vae_encoder_weight_items

        bundle_root = self._local_bundle_root(entry, version_key)
        vae_dir = (bundle_root / "vae") if bundle_root else None
        if vae_dir is None or not vae_dir.exists():
            raise RuntimeError(f"VAE encode: no vae directory under bundle {bundle_root}")

        scaling_factor = 1.0
        shift_factor = 0.0
        vae_cfg: dict[str, Any] = {}
        if (vae_dir / "config.json").exists():
            import json

            with open(vae_dir / "config.json") as f:
                vae_cfg = json.load(f)
            scaling_factor = float(vae_cfg.get("scaling_factor", 1.0))
            shift_factor = float(vae_cfg.get("shift_factor", 0.0))

        vae_weights: dict[str, Any] = {}
        for sf in sorted(vae_dir.glob("*.safetensors")):
            vae_weights.update(self.ctx.load_weights(str(sf)))
        if not vae_weights:
            raise RuntimeError(f"VAE encode: no weights under {vae_dir}")

        latent_c = self._vae_latent_channels_from_bundle(entry, version_key)
        enc = VAEEncoder(
            latent_channels=latent_c,
            ctx=self.ctx,
            scaling_factor=scaling_factor,
            shift_factor=shift_factor,
        )
        enc_items = prepare_vae_encoder_weight_items(vae_weights)
        loaded, skipped = enc.load_weights(enc_items, strict=False)
        if on_log:
            on_log(
                "info",
                f"vae_encode loaded={len(loaded)} skipped={len(skipped)} latent_channels={latent_c}",
            )
        if not any(k.startswith("conv_in.") for k in loaded):
            raise RuntimeError(
                "VAE encoder failed to load conv_in weights; check bundle encoder.* tensors. "
                f"skipped_sample={skipped[:8]}"
            )

        latent5 = enc.encode(image_nchw_f01)
        if getattr(latent5, "ndim", 0) == 5:
            return latent5[:, :, 0, :, :]
        return latent5

    def run(
        self,
        request: ImageGenerationRequest,
        ctx_exec: ExecutionContext,
        *,
        on_progress: Callable | None = None,
        on_log: Callable | None = None,
    ):
        """Execute MLX pipeline synchronously (called inside scheduler worker).

        Cancellation: check ``ctx_exec.cancel_token`` at each step; return ``None`` if cancelled.
        Output: written to ``ctx_exec.work_dir`` (same as the work dir assigned by scheduler).

        Returns: ``(output_path, metadata_dict)`` or ``None`` (cancelled)
        """
        model_key, version_key = parse_model_version(request.model)
        w, h = parse_size(request.size)
        seed = request.seed if request.seed is not None else random.randint(0, 2 ** 32 - 1)
        entry = self._registry.require(model_key)
        config_cls = get_config_class(entry.family)
        config = config_cls()
        family = getattr(entry, "family", "flux1")

        # ── Registry-driven parameter injection ──
        for param_key in ("text_encoder_out_layers", "vae_scale", "enable_thinking"):
            val = self._registry_scalar_default(entry, param_key, None)
            if val is not None:
                setattr(config, param_key, val)

        # Registry-driven supports_guidance override (e.g. z-image-turbo)
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

        # 1. Text encoding (driven by config.encoder_type, zero family branching)
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

        # 2. Load model
        model = self._load_model(family, config, entry, version_key or None)
        if model is None:
            raise RuntimeError(f"Failed to load model: {model_key}")

        # ── Hook ①: after weight loading (LoRA / Adapter merging) ──
        model.after_load_weights(bundle_root=str(bundle_root) if bundle_root else None)

        # ── Hook ②: condition preparation (ControlNet encode control image) ──
        extra_cond = model.prepare_conditioning(request, bundle_root=str(bundle_root) if bundle_root else None)

        # 3. Scheduler (registry default + request params, zero family branching)
        scheduler = get_scheduler(scheduler_default, ctx=self.ctx)
        vae_scale = getattr(config, "vae_scale", 8)
        # image_seq_len for sigma shift, matching reference implementation: fixed //16
        image_seq_len = (h // 16) * (w // 16)
        sched_extra: dict[str, Any] = {}
        _mu = self._registry_scalar_default(entry, "scheduler_mu", None)
        if _mu is not None:
            sched_extra["mu"] = float(_mu)
        for _k in (
            "scheduler_base_image_seq_len",
            "scheduler_max_image_seq_len",
            "scheduler_base_shift",
            "scheduler_max_shift",
        ):
            _v = self._registry_scalar_default(entry, _k, None)
            if _v is not None:
                sched_extra[_k] = _v
        # Registry-driven σ ladder (e.g. LongCat reference: linspace before μ shift).
        _sigma_sched = self._registry_scalar_default(entry, "scheduler_sigma_schedule", None)
        _cfg_renorm = bool(self._registry_scalar_default(entry, "enable_cfg_renorm", False))
        _cfg_renorm_min = float(self._registry_scalar_default(entry, "cfg_renorm_min", 0.0))
        if _sigma_sched == "linspace_1_to_inv_steps":
            sched_extra["sigmas"] = np.linspace(
                1.0, 1.0 / float(steps), steps, dtype=np.float64
            ).tolist()
        timesteps = scheduler.set_timesteps(
            steps,
            image_seq_len=image_seq_len,
            image_width=int(w),
            image_height=int(h),
            use_empirical_mu=self._registry_scalar_default(entry, "use_empirical_mu", True),
            requires_sigma_shift=self._registry_scalar_default(entry, "requires_sigma_shift", False),
            **sched_extra,
        )
        sigmas = getattr(scheduler, 'sigmas', None)
        # Match diffusers: time MLP input = scheduler.timesteps[i] (not unsafe float() on MLX sigmas).
        sched_ts = getattr(scheduler, "timesteps", None)
        timestep_embed_schedule: list[float] | None = None
        if sched_ts is not None:
            arr = np.asarray(sched_ts, dtype=np.float64).reshape(-1)
            timestep_embed_schedule = [float(x) for x in arr.tolist()]

        if on_log:
            parts = [
                f"infer model={model_key}",
                f"family={family}",
                f"version={version_key or 'default'}",
                f"size={w}x{h}",
                f"seed={seed}",
                f"steps={steps}",
                f"guidance={guidance}",
                f"scheduler={scheduler_default}",
                f"supports_guidance={getattr(config, 'supports_guidance', False)}",
                f"cfg_on={bool(neg_embeds is not None)}",
                f"image_seq_len={image_seq_len}",
                f"vae_scale={vae_scale}",
            ]
            if _sigma_sched is not None:
                parts.append(f"sigma_schedule={_sigma_sched}")
            _emu = self._registry_scalar_default(entry, "use_empirical_mu", True)
            parts.append(f"use_empirical_mu={_emu}")
            if sched_extra.get("mu") is not None:
                parts.append(f"scheduler_mu={sched_extra['mu']}")
            if timestep_embed_schedule and len(timestep_embed_schedule) >= 2:
                parts.append(
                    f"t_embed_ends=[{timestep_embed_schedule[0]:.6g},{timestep_embed_schedule[-1]:.6g}]"
                )
            elif timestep_embed_schedule and len(timestep_embed_schedule) == 1:
                parts.append(f"t_embed=[{timestep_embed_schedule[0]:.6g}]")
            if _cfg_renorm:
                parts.append(f"cfg_renorm=True cfg_renorm_min={_cfg_renorm_min}")
            on_log("info", " ".join(parts))

        # Create deterministic latent using seed
        latent_shape = (1, config.in_channels, h // vae_scale, w // vae_scale)
        if seed is not None:
            latents = self.ctx.seeded_randn(latent_shape, seed, dtype=self.ctx.float32())
        else:
            latents = self.ctx.randn(latent_shape, dtype=self.ctx.float32())

        # ── Hook ③: before denoise (ControlNet signal injection / latent modification) ──
        latents, extra_cond = model.before_denoise(latents, timesteps, sigmas, **extra_cond)

        # ------------------------------------------------------------------
        # 4. Denoising loop — fully generic: model handles timestep conversion and special params
        # ------------------------------------------------------------------
        for i, t in enumerate(timesteps):
            if ctx_exec.cancel_token.is_cancelled():
                return None

            # Unified model call interface: pass raw timestep index + sigmas, model converts internally
            model_kwargs = {"txt_embeds": txt_embeds} if txt_embeds is not None else {}
            model_kwargs.update(extra_cond)
            if sigmas is not None:
                model_kwargs["sigmas"] = sigmas
            if timestep_embed_schedule is not None and i < len(timestep_embed_schedule):
                model_kwargs["timestep_embed_value"] = timestep_embed_schedule[i]

            noise_cond = model(latents, t, **model_kwargs)

            # CFG — diffusers: eps = eps_u + guidance * (eps_c - eps_u)
            if neg_embeds is not None and getattr(config, "supports_guidance", False):
                uncond_kwargs = {"txt_embeds": neg_embeds}
                uncond_kwargs.update(extra_cond)
                if sigmas is not None:
                    uncond_kwargs["sigmas"] = sigmas
                if timestep_embed_schedule is not None and i < len(timestep_embed_schedule):
                    uncond_kwargs["timestep_embed_value"] = timestep_embed_schedule[i]
                noise_uncond = model(latents, t, **uncond_kwargs)
                noise_pred = noise_uncond + guidance * (noise_cond - noise_uncond)
                if _cfg_renorm and getattr(config, "supports_guidance", False):
                    noise_pred = model.refine_cfg_noise(
                        noise_cond, noise_pred, cfg_renorm_min=_cfg_renorm_min,
                    )
            else:
                noise_pred = noise_cond

            latents = scheduler.step(noise_pred, t, latents)

            # ── Hook ④: per-step callback (dynamic condition / logging) ──
            model.step_callback(i, latents, noise_pred)

            if on_progress:
                on_progress((i + 1) / len(timesteps), i + 1, len(timesteps), None)
            if on_log:
                extra = ""
                if timestep_embed_schedule is not None and i < len(timestep_embed_schedule):
                    extra = f" t_embed={timestep_embed_schedule[i]:.6g}"
                on_log("info", f"Step {i + 1}/{len(timesteps)}{extra}")

        if ctx_exec.cancel_token.is_cancelled():
            return None

        # 5. VAE decode
        image = self._vae_decode(latents, entry, version_key or None, on_log=on_log)

        if ctx_exec.cancel_token.is_cancelled():
            return None

        # 6. Save (task work dir, see TaskScheduler._work_dir)
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

    def run_edit(
        self,
        request: ImageEditRequest,
        ctx_exec: ExecutionContext,
        *,
        on_progress: Callable | None = None,
        on_log: Callable | None = None,
    ):
        """图像编辑：``operation=rewrite`` 走 VAE 编码 + latent 与噪声按 ``source_fidelity`` 混合 + 标准去噪。

        ``retouch`` / ``extend`` 需蒙版或画布扩展，尚未接线 — 显式 ``RuntimeError``。
        ``rewrite_mode=instruct``（flux1-kontext 指令编辑）未在此管线实现。
        """
        if request.operation != "rewrite":
            raise RuntimeError(
                f"ImagePipeline.run_edit: operation {request.operation!r} is not implemented on MLX "
                "(only rewrite / img2img is wired; retouch and extend require masks or canvas logic)."
            )
        if request.rewrite_mode == "instruct":
            raise RuntimeError(
                "ImagePipeline.run_edit: rewrite_mode instruct (flux1-kontext instruction editing) "
                "is not implemented in this pipeline."
            )

        model_key, version_key = parse_model_version(request.model)
        entry = self._registry.require(model_key)
        config_cls = get_config_class(entry.family)
        config = config_cls()
        family = getattr(entry, "family", "flux1")

        for param_key in ("text_encoder_out_layers", "vae_scale", "enable_thinking"):
            val = self._registry_scalar_default(entry, param_key, None)
            if val is not None:
                setattr(config, param_key, val)
        sg = self._registry_scalar_default(entry, "supports_guidance", None)
        if sg is not None:
            config.supports_guidance = bool(sg)

        vae_scale = int(getattr(config, "vae_scale", 8))
        enc_spatial_div = 8  # ``VAEEncoder`` 固定 8× 空间下采样
        # Latent grid from ``vae_scale`` must match VAE encoder output grid (encoder 固定 ÷8).
        # e.g. Flux2 ``vae_scale=16`` ⇒ latent H/16 vs encode H/8 — 未接线 img2img。
        def _latent_hw(hpx: int, wpx: int) -> tuple[int, int]:
            return hpx // vae_scale, wpx // vae_scale

        def _enc_hw(hpx: int, wpx: int) -> tuple[int, int]:
            return hpx // enc_spatial_div, wpx // enc_spatial_div

        if ctx_exec.cancel_token.is_cancelled():
            return None

        bundle_root = self._local_bundle_root(entry, version_key or None)
        from PIL import Image

        src_path = ctx_exec.asset_store.get_file_path(request.source_asset_id)
        pil = Image.open(src_path).convert("RGB")
        w0, h0 = pil.size
        w, h = self._align_hw_multiples(w0, h0, align=16)
        pil = self._center_crop_pil(pil, w, h)
        if _latent_hw(h, w) != _enc_hw(h, w):
            raise RuntimeError(
                f"Image edit (rewrite) is not wired for vae_scale={vae_scale} when the VAE encoder "
                f"outputs a {_enc_hw(h, w)[0]}×{_enc_hw(h, w)[1]} latent grid but the transformer expects "
                f"{_latent_hw(h, w)[0]}×{_latent_hw(h, w)[1]} (image {w}×{h}). Models with vae_scale≠8 "
                f"(e.g. Flux2) need a dedicated encode / pack bridge before img2img can run."
            )

        seed = request.seed if request.seed is not None else random.randint(0, 2 ** 32 - 1)
        img_f01 = self._pil_to_nchw_float01(pil, w, h)
        encoded = self._vae_encode_tensor(img_f01, entry, version_key or None, on_log=on_log)
        if encoded.shape[1] != config.in_channels:
            raise RuntimeError(
                f"VAE encode produced {encoded.shape[1]} latent channels but model {model_key!r} "
                f"(family={family}) expects in_channels={config.in_channels}. "
                "Check bundle VAE config / model family alignment."
            )

        fidelity = float(request.source_fidelity)
        fidelity = max(0.0, min(1.0, fidelity))
        noise_shape = encoded.shape
        noise = self.ctx.seeded_randn(noise_shape, seed, dtype=self.ctx.float32())
        latents = encoded * fidelity + noise * (1.0 - fidelity)

        steps_default = self._registry_scalar_default(entry, "steps", 4)
        scheduler_registry = self._registry_scalar_default(entry, "scheduler", None)
        scheduler_default = scheduler_registry or "flow_match_euler"
        steps = int(request.steps) if request.steps is not None else int(steps_default)
        steps = max(1, steps)
        guidance_default = self._registry_scalar_default(entry, "guidance", 0.0)
        guidance = float(guidance_default)
        if not getattr(config, "supports_guidance", True):
            guidance = 0.0

        txt_embeds = None
        neg_embeds = None
        encoder_type = getattr(config, "encoder_type", "t5")
        if request.prompt and encoder_type != "t5":
            if bundle_root is None:
                raise RuntimeError(
                    f"Model {model_key!r} has no installed bundle at local_path "
                    f"(version={version_key or 'default'}); cannot load text encoder."
                )
            txt_embeds = self._text_encode(
                request.prompt, bundle_root=bundle_root, encoder_type=encoder_type, config=config,
            )
            if getattr(config, "supports_guidance", False) and guidance > 1.0:
                neg_txt = request.negative_prompt.strip() if request.negative_prompt else " "
                neg_embeds = self._text_encode(
                    neg_txt, bundle_root=bundle_root, encoder_type=encoder_type, config=config,
                )
        elif request.prompt and config.text_dim > 0:
            enc = T5Encoder(self.ctx, "google/t5-v1_1-xxl")
            txt_embeds = enc.encode([request.prompt])

        if ctx_exec.cancel_token.is_cancelled():
            return None

        model = self._load_model(family, config, entry, version_key or None)
        if model is None:
            raise RuntimeError(f"Failed to load model: {model_key}")
        model.after_load_weights(bundle_root=str(bundle_root) if bundle_root else None)
        extra_cond = model.prepare_conditioning(request, bundle_root=str(bundle_root) if bundle_root else None)

        scheduler = get_scheduler(scheduler_default, ctx=self.ctx)
        image_seq_len = (h // 16) * (w // 16)
        sched_extra: dict[str, Any] = {}
        _mu = self._registry_scalar_default(entry, "scheduler_mu", None)
        if _mu is not None:
            sched_extra["mu"] = float(_mu)
        for _k in (
            "scheduler_base_image_seq_len",
            "scheduler_max_image_seq_len",
            "scheduler_base_shift",
            "scheduler_max_shift",
        ):
            _v = self._registry_scalar_default(entry, _k, None)
            if _v is not None:
                sched_extra[_k] = _v
        _sigma_sched = self._registry_scalar_default(entry, "scheduler_sigma_schedule", None)
        _cfg_renorm = bool(self._registry_scalar_default(entry, "enable_cfg_renorm", False))
        _cfg_renorm_min = float(self._registry_scalar_default(entry, "cfg_renorm_min", 0.0))
        if _sigma_sched == "linspace_1_to_inv_steps":
            sched_extra["sigmas"] = np.linspace(
                1.0, 1.0 / float(steps), steps, dtype=np.float64
            ).tolist()
        timesteps = scheduler.set_timesteps(
            steps,
            image_seq_len=image_seq_len,
            image_width=int(w),
            image_height=int(h),
            use_empirical_mu=self._registry_scalar_default(entry, "use_empirical_mu", True),
            requires_sigma_shift=self._registry_scalar_default(entry, "requires_sigma_shift", False),
            **sched_extra,
        )
        sigmas = getattr(scheduler, "sigmas", None)
        sched_ts = getattr(scheduler, "timesteps", None)
        timestep_embed_schedule: list[float] | None = None
        if sched_ts is not None:
            arr = np.asarray(sched_ts, dtype=np.float64).reshape(-1)
            timestep_embed_schedule = [float(x) for x in arr.tolist()]

        if on_log:
            on_log(
                "info",
                f"edit rewrite model={model_key} family={family} size={w}x{h} seed={seed} "
                f"steps={steps} scheduler={scheduler_default} source_fidelity={fidelity}",
            )

        latents, extra_cond = model.before_denoise(latents, timesteps, sigmas, **extra_cond)

        for i, t in enumerate(timesteps):
            if ctx_exec.cancel_token.is_cancelled():
                return None
            model_kwargs = {"txt_embeds": txt_embeds} if txt_embeds is not None else {}
            model_kwargs.update(extra_cond)
            if sigmas is not None:
                model_kwargs["sigmas"] = sigmas
            if timestep_embed_schedule is not None and i < len(timestep_embed_schedule):
                model_kwargs["timestep_embed_value"] = timestep_embed_schedule[i]

            noise_cond = model(latents, t, **model_kwargs)

            if neg_embeds is not None and getattr(config, "supports_guidance", False):
                uncond_kwargs = {"txt_embeds": neg_embeds}
                uncond_kwargs.update(extra_cond)
                if sigmas is not None:
                    uncond_kwargs["sigmas"] = sigmas
                if timestep_embed_schedule is not None and i < len(timestep_embed_schedule):
                    uncond_kwargs["timestep_embed_value"] = timestep_embed_schedule[i]
                noise_uncond = model(latents, t, **uncond_kwargs)
                noise_pred = noise_uncond + guidance * (noise_cond - noise_uncond)
                if _cfg_renorm and getattr(config, "supports_guidance", False):
                    noise_pred = model.refine_cfg_noise(
                        noise_cond, noise_pred, cfg_renorm_min=_cfg_renorm_min,
                    )
            else:
                noise_pred = noise_cond

            latents = scheduler.step(noise_pred, t, latents)
            model.step_callback(i, latents, noise_pred)
            if on_progress:
                on_progress((i + 1) / len(timesteps), i + 1, len(timesteps), None)
            if on_log:
                extra = ""
                if timestep_embed_schedule is not None and i < len(timestep_embed_schedule):
                    extra = f" t_embed={timestep_embed_schedule[i]:.6g}"
                on_log("info", f"Step {i + 1}/{len(timesteps)}{extra}")

        if ctx_exec.cancel_token.is_cancelled():
            return None

        image = self._vae_decode(latents, entry, version_key or None, on_log=on_log)
        if ctx_exec.cancel_token.is_cancelled():
            return None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        work = Path(ctx_exec.work_dir)
        work.mkdir(parents=True, exist_ok=True)
        out_path = work / f"{model_key}_edit_{seed}_{timestamp}.png"
        if hasattr(image, "save"):
            image.save(str(out_path))

        return str(out_path), {
            "model": request.model,
            "seed": seed,
            "prompt": request.prompt,
            "steps": steps,
            "guidance": guidance,
            "width": w,
            "height": h,
            "mime_type": "image/png",
            "operation": request.operation,
            "source_fidelity": fidelity,
        }

    def run_upscale(
        self,
        request: ImageUpscaleRequest,
        ctx_exec: ExecutionContext,
        *,
        on_progress: Callable | None = None,
        on_log: Callable | None = None,
    ):
        """SeedVR2 超分：``seedvr2.upscale_pipeline``（经 ``run_seedvr2_upscale`` 入口）。"""
        model_key, version_key = parse_model_version(request.model)
        entry = self._registry.require(model_key)
        family = getattr(entry, "family", "")
        if family != "seedvr2":
            raise RuntimeError(
                f"Image upscale on MLX is only implemented for family 'seedvr2'; "
                f"model {model_key!r} has family={family!r}."
            )

        if ctx_exec.cancel_token.is_cancelled():
            return None

        bundle_root = self._local_bundle_root(entry, version_key or None)
        if bundle_root is None:
            raise RuntimeError(
                f"Model {model_key!r} has no installed bundle (version={version_key or 'default'}); "
                "cannot run SeedVR2 upscale."
            )

        from PIL import Image

        src_path = ctx_exec.asset_store.get_file_path(request.source_asset_id)
        if not src_path.is_file():
            raise RuntimeError(f"Source asset file missing: {src_path}")

        scale = int(request.scale)
        seed = (request.metadata or {}).get("seed")
        if seed is not None:
            seed = int(seed)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        work = Path(ctx_exec.work_dir)
        work.mkdir(parents=True, exist_ok=True)
        out_path = work / f"{model_key}_up_{timestamp}.png"

        def _log(level: str, msg: str) -> None:
            if on_log:
                on_log(level, msg)

        from backend.engine.seedvr2.transformer import run_seedvr2_upscale

        extra = run_seedvr2_upscale(
            bundle_path=bundle_root,
            model_key=model_key,
            source_image=src_path,
            scale=scale,
            softness=float(request.denoise),
            seed=seed,
            output_png=out_path,
            on_log=_log,
        )

        if ctx_exec.cancel_token.is_cancelled():
            return None

        pil = Image.open(out_path)
        w, h = pil.size
        if on_progress:
            on_progress(1.0, 1, 1, None)

        meta = {
            "model": request.model,
            "width": w,
            "height": h,
            "mime_type": "image/png",
            "scale": scale,
            "denoise": float(request.denoise),
        }
        meta.update(extra)
        return str(out_path), meta

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _text_encode(self, text: str, *, bundle_root: Path, encoder_type: str, config: Any) -> Any:
        """Text encoding — routed to specific implementation by registry encoder_type."""
        enc_dir = bundle_root / "text_encoder"
        tok_dir = bundle_root / "tokenizer"
        if not tok_dir.exists():
            tok_dir = enc_dir
        if not enc_dir.exists():
            enc_dir = bundle_root
            tok_dir = bundle_root

        enc_cls = _get_text_encoder(encoder_type)
        enc_kwargs: dict[str, Any] = {}
        out_layers = getattr(config, "text_encoder_out_layers", None)
        if out_layers is not None:
            enc_kwargs["hidden_state_layers"] = tuple(out_layers)
        enc_kwargs["enable_thinking"] = getattr(config, "enable_thinking", False)
        enc = enc_cls(self.ctx, str(enc_dir), tokenizer_path=str(tok_dir), **enc_kwargs)
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

        # Detect quantized weights (e.g. int8/int4) → load missing biases from fp16
        fallback_weights = None
        if any(k.endswith(".scales") for k in w.keys()):
            fp16_root = self._resolve_fallback_fp16(bundle_root)
            if fp16_root:
                fb = {}
                fb_tp = fp16_root / "transformer"
                if fb_tp.exists():
                    for sf in sorted(fb_tp.glob("*.safetensors")):
                        fb.update(self.ctx.load_weights(str(sf)))
                    if remap_fn:
                        fb = remap_fn(fb)
                    fallback_weights = list(fb.items())

        model.load_weights(list(w.items()), strict=False,
                           fallback_weights=fallback_weights,
                           ctx=self.ctx)
        self.ctx.eval(*[p for _, p in model.parameters()])
        return model

    def _resolve_fallback_fp16(self, bundle_root: Path) -> Path | None:
        """Infer fp16 path (z-image-fp16) from quantized path (e.g. z-image-int8)."""
        # Pattern: .../z-image-int8 → .../z-image-fp16
        name = bundle_root.name
        parent = bundle_root.parent
        if "-int8" in name or "-int4" in name or "-4bit" in name:
            fp16_name = name.replace("-int8", "-fp16").replace("-int4", "-fp16").replace("-4bit", "-fp16")
            fp16_path = parent / fp16_name
            if fp16_path.exists():
                return fp16_path
        return None

    def _vae_preprocess_special(self, latents, vae_weights, scaling_factor, shift_factor):
        """Special VAE preprocessing — flux2 style (triggered by weight detection, not family-hardcoded)."""
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

    def _vae_decode(
        self,
        latents,
        entry,
        version_key,
        *,
        on_log: Callable | None = None,
    ):
        """VAE decode latent → PIL Image."""
        ctx = self.ctx
        from PIL import Image
        from backend.engine.common._vae import VAEDecoder, vae_output_to_uint8_hwc

        bundle_root = self._local_bundle_root(entry, version_key)
        vae_dir = (bundle_root / "vae") if bundle_root else None

        scaling_factor = 1.0
        shift_factor = 0.0
        vae_cfg: dict[str, Any] = {}
        if vae_dir and (vae_dir / "config.json").exists():
            import json
            with open(vae_dir / "config.json") as f:
                vae_cfg = json.load(f)
            scaling_factor = float(vae_cfg.get("scaling_factor", 1.0))
            shift_factor = float(vae_cfg.get("shift_factor", 0.0))

        if latents.ndim == 3:
            B, seq_len, channels = latents.shape
            latent_h = int(seq_len ** 0.5)
            latent_w = seq_len // latent_h
            latents = latents.reshape(B, latent_h, latent_w, channels).transpose(0, 3, 1, 2)

        vae_weights: dict[str, Any] = {}
        if vae_dir and vae_dir.exists():
            saf_paths = sorted(vae_dir.glob("*.safetensors"))
            if saf_paths:
                for sf in saf_paths:
                    vae_weights.update(ctx.load_weights(str(sf)))
            elif (vae_dir / "config.json").is_file():
                raise RuntimeError(
                    f"VAE directory has config but no *.safetensors under {vae_dir}; "
                    "cannot decode (install model weights)."
                )

        # Flux2-style BN/post_quant path — only when config enables quant/post_quant paths.
        # LongCat / standard AutoencoderKL has use_quant_conv=false; stray key names in files
        # must NOT trigger this branch or latents are corrupted before decode.
        use_quant_path = bool(vae_cfg.get("use_quant_conv", False))
        use_post_quant_path = bool(vae_cfg.get("use_post_quant_conv", False))
        if (use_quant_path or use_post_quant_path) and (
            "bn.running_mean" in vae_weights or "post_quant_conv.weight" in vae_weights
        ):
            latents = self._vae_preprocess_special(latents, vae_weights, scaling_factor, shift_factor)
            scaling_factor = 1.0
            shift_factor = 0.0

        C = latents.shape[1] if latents.ndim >= 4 else 16
        vae = VAEDecoder(latent_channels=C, ctx=ctx, scaling_factor=scaling_factor, shift_factor=shift_factor)

        if not vae_weights:
            raise RuntimeError(
                f"No VAE weights loaded for decode (bundle_root={bundle_root}, vae_dir={vae_dir}). "
                "Ensure models/.../vae/*.safetensors exists."
            )

        from backend.engine.common.weights import remap_vae_weights

        decoder_w = remap_vae_weights(vae_weights)
        if not decoder_w:
            raise RuntimeError(
                f"VAE weights under {vae_dir} produced no decoder tensors after remap; check bundle."
            )
        loaded, skipped = vae.load_weights(list(decoder_w.items()), strict=False)
        if on_log:
            on_log(
                "info",
                " ".join(
                    [
                        f"vae_decode latent_shape={tuple(latents.shape)}",
                        f"scaling_factor={scaling_factor}",
                        f"shift_factor={shift_factor}",
                        f"decoder_tensors={len(decoder_w)}",
                        f"loaded_params={len(loaded)}",
                        f"skipped_params={len(skipped)}",
                    ]
                ),
            )
        if not any(k.startswith("conv_in.") for k in loaded):
            raise RuntimeError(
                "VAE decoder failed to load conv_in weights; decode would be garbage. "
                f"skipped_sample={skipped[:8]}"
            )
        image = vae.forward(latents)
        pixels = vae_output_to_uint8_hwc(image, self.ctx)
        return Image.fromarray(pixels)
