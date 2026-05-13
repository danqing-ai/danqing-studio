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
    ImageEditRequest,
    LogEvent, parse_model_version, parse_size,
)
from backend.engine.common.cache import ModelCache
from backend.engine.common.pipeline_registry import (
    local_bundle_root as _local_bundle_root_fn,
    registry_scalar_default as _registry_scalar_default_fn,
    resolve_project_path as _resolve_project_path_fn,
    resolve_version_block as _resolve_version_block_fn,
)
from backend.engine.common.schedulers import get_scheduler
from backend.engine.common.text_encoders import T5Encoder
from backend.engine._transformer_registry import (
    encode_prompt_with_image_text_encoder as _encode_prompt_image_text,
    get_transformer_class as _get_transformer_class,
    get_weight_remap as _get_weight_remap,
)
from backend.engine.config.model_configs import get_config_class
from backend.engine.runtime._base import RuntimeContext

# Bar semantics: denoise uses most of the range; VAE decode + save use the tail so we do not
# sit at 100% while post-processing (queue ETA uses ``1 - progress``).
_IMAGE_PIPELINE_DENOISE_PROGRESS_SHARE = 0.88
_IMAGE_PIPELINE_POST_PROGRESS_SHARE = 0.12


def _denoise_latent_noise_dtype(ctx: RuntimeContext, config: Any):
    """Initial Gaussian noise dtype (e.g. Z-Image / mflux uses bf16 latents)."""
    spec = getattr(config, "latent_noise_dtype", None)
    if isinstance(spec, str) and spec.lower() in ("bfloat16", "bf16"):
        return ctx.bfloat16()
    return ctx.float32()


def _t5_encoder_bundle_paths(bundle_root: Path | None) -> tuple[str, str]:
    """从已安装的 Diffusers 风格 bundle 解析 T5 **权重**目录与 **tokenizer** 目录（须分离）。

    **FluxPipeline**（FLUX.1）：``text_encoder`` 为 CLIP，**T5-xxl 在 ``text_encoder_2``**，词表在 ``tokenizer_2``。
    单 T5 管线（如部分视频权重）：仅 ``text_encoder`` + ``tokenizer``。

    禁止用 ``text_encoder/`` 代替 tokenizer（FIBO 等 bundle 内另有词表格式，会触发 ``TypeError``）。
    禁止在已声明 ``local_path`` 的模型上隐式使用 ``google/t5-v1_1-xxl`` 走 Hub（首跑极慢、易超时）。
    """
    if bundle_root is None:
        raise RuntimeError(
            "T5 text encoding requires an installed model bundle (registry versions.local_path). "
            "Refusing implicit Hugging Face hub download (google/t5-v1_1-xxl)."
        )
    te2 = bundle_root / "text_encoder_2"
    te1 = bundle_root / "text_encoder"
    enc_dir: Path | None = None
    tok_candidates: list[Path] = []

    if te2.is_dir() and any(te2.iterdir()):
        enc_dir = te2
        tok_candidates = [
            bundle_root / "tokenizer_2",
            te2 / "tokenizer",
        ]
    elif te1.is_dir() and any(te1.iterdir()):
        enc_dir = te1
        tok_candidates = [
            bundle_root / "tokenizer",
            te1 / "tokenizer",
        ]

    if enc_dir is None:
        raise RuntimeError(
            f"T5 text encoder directory missing: expected ``{te2}`` or ``{te1}``. "
            "Re-install or sync the model bundle."
        )

    tok_dir: Path | None = None
    for c in tok_candidates:
        if c.is_dir() and any(c.iterdir()):
            tok_dir = c
            break
    if tok_dir is None and (bundle_root / "tokenizer_config.json").is_file():
        tok_dir = bundle_root
    if tok_dir is None:
        raise RuntimeError(
            f"T5 tokenizer not found under {bundle_root}. Tried: "
            + ", ".join(str(c) for c in tok_candidates)
            + ". Re-install the upstream tokenizer assets."
        )

    return str(enc_dir), str(tok_dir)


def _image_pipeline_emit_denoise_progress(
    on_progress: Callable[..., None] | None,
    step_1based: int,
    n_steps: int,
) -> None:
    if on_progress is None:
        return
    n = max(1, int(n_steps))
    s = min(max(1, int(step_1based)), n)
    p = _IMAGE_PIPELINE_DENOISE_PROGRESS_SHARE * (s / n)
    on_progress(p, s, n, "denoise")


def _image_pipeline_emit_post_progress(
    on_progress: Callable[..., None] | None,
    *,
    n_steps: int,
    within_post: float,
) -> None:
    """``within_post`` in [0, 1]: position inside the post-denoise segment (VAE / save)."""
    if on_progress is None:
        return
    n = max(1, int(n_steps))
    w = min(1.0, max(0.0, float(within_post)))
    p = _IMAGE_PIPELINE_DENOISE_PROGRESS_SHARE + _IMAGE_PIPELINE_POST_PROGRESS_SHARE * w
    on_progress(p, n, n, "post")


def _image_pipeline_emit_complete(on_progress: Callable[..., None] | None, n_steps: int) -> None:
    if on_progress is None:
        return
    n = max(1, int(n_steps))
    on_progress(1.0, n, n, None)


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
        return _resolve_project_path_fn(self._project_root, local_path)

    @staticmethod
    def _registry_scalar_default(entry, key: str, fallback):
        return _registry_scalar_default_fn(entry, key, fallback)

    def _resolve_version_block(self, entry, version_key: str | None) -> dict | None:
        return _resolve_version_block_fn(entry, version_key)

    def _local_bundle_root(self, entry, version_key: str | None) -> Path | None:
        return _local_bundle_root_fn(self._project_root, entry, version_key)

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

    def _load_vae_dir_cfg_weights(
        self,
        entry,
        version_key: str | None,
    ) -> tuple[Path, dict[str, Any], dict[str, Any]]:
        bundle_root = self._local_bundle_root(entry, version_key)
        vae_dir = (bundle_root / "vae") if bundle_root else None
        if vae_dir is None or not vae_dir.exists():
            raise RuntimeError(f"VAE: no vae directory under bundle {bundle_root}")

        vae_cfg: dict[str, Any] = {}
        if (vae_dir / "config.json").exists():
            import json

            with open(vae_dir / "config.json") as f:
                vae_cfg = json.load(f)

        vae_weights: dict[str, Any] = {}
        for sf in sorted(vae_dir.glob("*.safetensors")):
            vae_weights.update(self.ctx.load_weights(str(sf)))
        if not vae_weights:
            raise RuntimeError(f"VAE encode: no weights under {vae_dir}")
        return vae_dir, vae_cfg, vae_weights

    @staticmethod
    def _flux2_crop_even_hw_latent(z: Any, _ctx: RuntimeContext) -> Any:
        """Drop last row/col if odd — matches mflux img2img crop before patchify."""
        if int(z.shape[2]) % 2 != 0:
            z = z[:, :, :-1, :]
        if int(z.shape[3]) % 2 != 0:
            z = z[:, :, :, :-1]
        return z

    def _flux2_patchify_mean_latent(self, z_bchw: Any) -> Any:
        """Flux2 VAE mean [B,32,H,W] → [B,128,H/2,W/2] (mflux ``Flux2LatentCreator.patchify_latents``)."""
        ctx = self.ctx
        B, C, H, W = (int(z_bchw.shape[0]), int(z_bchw.shape[1]), int(z_bchw.shape[2]), int(z_bchw.shape[3]))
        x = ctx.reshape(z_bchw, (B, C, H // 2, 2, W // 2, 2))
        x = ctx.permute(x, (0, 1, 3, 5, 2, 4))
        return ctx.reshape(x, (B, C * 4, H // 2, W // 2))

    def _flux2_bn_normalize_editing_latent(self, latents: Any, vae_weights: dict[str, Any], bn_eps: float) -> Any:
        ctx = self.ctx
        bn_mean = vae_weights.get("bn.running_mean")
        bn_var = vae_weights.get("bn.running_var")
        if bn_mean is None or bn_var is None:
            raise RuntimeError("Flux2 img2img: bundle missing bn.running_mean / bn.running_var on VAE checkpoint.")
        bm = bn_mean.reshape(1, -1, 1, 1).astype(latents.dtype)
        bv = bn_var.reshape(1, -1, 1, 1).astype(latents.dtype)
        std = ctx.sqrt(bv + float(bn_eps))
        return (latents - bm) / std

    def _vae_encode_tensor(
        self,
        image_nchw_f01: Any,
        entry,
        version_key: str | None,
        *,
        height_px: int | None = None,
        width_px: int | None = None,
        on_log: Callable | None = None,
    ) -> Any:
        """Encode ``[1,3,H,W]`` float **linear RGB in [0, 1]** → model latent (shape depends on VAE class).

        Applies **[-1, 1] pixel normalization** before ``conv_in`` (mflux / diffusers img2img).
        """
        from backend.engine.common.vae import VAEEncoder, prepare_vae_encoder_weight_items
        from backend.engine.common.vae.qwen_image import QwenVAE, apply_qwen_vae_weights_from_bundle

        image_n11 = image_nchw_f01 * 2.0 - 1.0

        _, vae_cfg, vae_weights = self._load_vae_dir_cfg_weights(entry, version_key)
        vae_cls = str(vae_cfg.get("_class_name") or "")

        if vae_cls == "AutoencoderKLQwenImage":
            if height_px is None or width_px is None:
                raise RuntimeError("Qwen VAE encode: height_px and width_px are required for latent packing.")
            vae = QwenVAE()
            br = self._local_bundle_root(entry, version_key)
            if br is None:
                raise RuntimeError("Qwen VAE encode: bundle_root missing")
            apply_qwen_vae_weights_from_bundle(vae, br, project_root=self._project_root)
            enc_out = vae.encode(image_n11)
            if getattr(enc_out, "ndim", 0) == 5 and int(enc_out.shape[2]) == 1:
                enc_out = enc_out[:, :, 0, :, :]
            packed = self._qwen_pack_latents_nchw(enc_out, height_px, width_px)
            if on_log:
                on_log("info", f"vae_encode qwen packed_shape={tuple(packed.shape)}")
            if getattr(self.ctx, "backend", None) == "mlx":
                self.ctx.eval(packed)
            return packed

        scaling_factor = float(vae_cfg.get("scaling_factor", 1.0))
        shift_factor = float(vae_cfg.get("shift_factor", 0.0))

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

        if vae_cls == "AutoencoderKLFlux2":
            h64 = enc.encode_conv_out_nchw(image_n11)
            qw = vae_weights.get("quant_conv.weight")
            qb = vae_weights.get("quant_conv.bias")
            if qw is None or qb is None:
                raise RuntimeError("Flux2 img2img: VAE checkpoint missing quant_conv.* tensors.")
            ctx = self.ctx
            t_nhwc = ctx.permute(h64, (0, 2, 3, 1))
            t_q = ctx.conv2d(t_nhwc, ctx.permute(qw, (0, 2, 3, 1)), stride=1, padding=0)
            t_q = t_q + qb.reshape(1, 1, 1, -1)
            t_q = ctx.permute(t_q, (0, 3, 1, 2))
            mean = t_q[:, :latent_c]
            z = (mean - shift_factor) * scaling_factor
            z = self._flux2_crop_even_hw_latent(z, self.ctx)
            z = self._flux2_patchify_mean_latent(z)
            bn_eps = float(vae_cfg.get("batch_norm_eps", 1e-4))
            z = self._flux2_bn_normalize_editing_latent(z, vae_weights, bn_eps)
            if on_log:
                on_log("info", f"vae_encode flux2 transformer_latent_shape={tuple(z.shape)}")
            if getattr(self.ctx, "backend", None) == "mlx":
                self.ctx.eval(z)
            return z

        latent5 = enc.encode(image_n11)
        z = latent5[:, :, 0, :, :] if getattr(latent5, "ndim", 0) == 5 else latent5
        return z

    def _qwen_pack_latents_nchw(self, encoded_b16hw: Any, height_px: int, width_px: int) -> Any:
        """[B,16,H_lat,W_lat] → [B,64,H_px/16,W_px/16] (mflux ``FluxLatentCreator.pack_latents``, NCHW)."""
        ctx = self.ctx
        B = int(encoded_b16hw.shape[0])
        Hg = height_px // 16
        Wg = width_px // 16
        x = ctx.reshape(encoded_b16hw, (B, 16, Hg, 2, Wg, 2))
        x = ctx.permute(x, (0, 2, 4, 1, 3, 5))
        x = ctx.reshape(x, (B, Hg * Wg, 64))
        x = ctx.reshape(x, (B, Hg, Wg, 64))
        return ctx.permute(x, (0, 3, 1, 2))

    def _qwen_unpack_latents_nchw(self, packed_b64hw: Any) -> Any:
        """Inverse of ``_qwen_pack_latents_nchw`` → [B,16,H_lat,W_lat]."""
        ctx = self.ctx
        B = int(packed_b64hw.shape[0])
        Hg, Wg = int(packed_b64hw.shape[2]), int(packed_b64hw.shape[3])
        x = ctx.permute(packed_b64hw, (0, 2, 3, 1))
        x = ctx.reshape(x, (B, Hg, Wg, 16, 2, 2))
        x = ctx.permute(x, (0, 3, 1, 4, 2, 5))
        return ctx.reshape(x, (B, 16, Hg * 2, Wg * 2))

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
        acts = getattr(entry, "actions", frozenset())
        if "generate" not in acts:
            raise RuntimeError(
                f"Model {model_key!r} is not registered for text-to-image (actions must include create); "
                "refusing ImagePipeline.run — see config/models_registry.json."
            )
        config_cls = get_config_class(entry.family)
        config = config_cls()
        family = getattr(entry, "family", "flux1")

        # ── Registry-driven parameter injection ──
        for param_key in (
            "text_encoder_out_layers",
            "vae_scale",
            "enable_thinking",
            "latent_noise_dtype",
            # Flux2 Klein 4B / 9B 等：架构超参由注册表声明（避免全家用 9B 默认）
            "inner_dim",
            "num_heads",
            "attn_head_dim",
            "num_layers",
            "num_single_layers",
            "joint_attention_dim",
        ):
            val = self._registry_scalar_default(entry, param_key, None)
            if val is not None and hasattr(config, param_key):
                if param_key == "text_encoder_out_layers" and isinstance(val, list):
                    setattr(config, param_key, tuple(int(x) for x in val))
                else:
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
        _meta = request.metadata or {}
        scheduler_request = request.scheduler or _meta.get("scheduler")
        scheduler_default = scheduler_request or scheduler_registry or "flow_match_euler"

        steps = int(request.steps) if request.steps is not None else int(steps_default)
        steps = max(1, steps)
        guidance = float(request.guidance) if request.guidance is not None else float(guidance_default)
        if not getattr(config, "supports_guidance", True):
            guidance = 0.0

        # 1. Text encoding (driven by config.encoder_type, zero family branching)
        txt_embeds = None
        neg_embeds = None
        txt_attn_mask = None
        neg_attn_mask = None
        encoder_type = getattr(config, "encoder_type", "t5")
        if request.prompt and encoder_type != "t5":
            if bundle_root is None:
                raise RuntimeError(
                    f"Model {model_key!r} has no installed bundle at local_path "
                    f"(version={version_key or 'default'}); cannot load text encoder."
                )
            txt_embeds, txt_attn_mask = _encode_prompt_image_text(
                self.ctx,
                request.prompt,
                encoder_type=encoder_type,
                bundle_root=bundle_root,
                config=config,
            )
            if getattr(config, "supports_guidance", False) and guidance > 1.0:
                neg_txt = (request.negative_prompt or "").strip() or " "
                neg_embeds, neg_attn_mask = _encode_prompt_image_text(
                    self.ctx,
                    neg_txt,
                    encoder_type=encoder_type,
                    bundle_root=bundle_root,
                    config=config,
                )
        elif request.prompt and config.text_dim > 0:
            t5_dir, t5_tok = _t5_encoder_bundle_paths(bundle_root)
            enc = T5Encoder(self.ctx, t5_dir, tokenizer_path=t5_tok)
            txt_embeds = enc.encode([request.prompt])

        if ctx_exec.cancel_token.is_cancelled():
            return None

        # 2. Load model
        model = self._load_model(family, config, entry, version_key or None)
        if model is None:
            raise RuntimeError(f"Failed to load model: {model_key}")

        # ── Hook ①: after weight loading (LoRA / Adapter merging) ──
        model.after_load_weights(bundle_root=str(bundle_root) if bundle_root else None)
        self._apply_image_lora_adapters(family, model, request, on_log)

        # ── Runtime quantization (auto 4-bit for fp16 versions) ──
        ver_block = _resolve_version_block_fn(entry, version_key or None)
        disk_quant = (ver_block or {}).get("quantization")
        if not disk_quant:
            if on_log:
                on_log("info", "Auto runtime quantization (4-bit)...")
            model.quantize_runtime(bits=4, ctx=self.ctx)
            self.ctx.eval(*[p for _, p in model.parameters()])

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
        _lnd = _denoise_latent_noise_dtype(self.ctx, config)
        if seed is not None:
            latents = self.ctx.seeded_randn(latent_shape, seed, dtype=_lnd)
        else:
            latents = self.ctx.randn(latent_shape, dtype=_lnd)

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
            if encoder_type == "qwen_image":
                model_kwargs["image_height"] = h
                model_kwargs["image_width"] = w
                model_kwargs["scheduler_timesteps"] = sched_ts
                if txt_attn_mask is not None:
                    model_kwargs["encoder_hidden_states_mask"] = txt_attn_mask

            noise_cond = model(latents, t, **model_kwargs)

            # CFG — default diffusers-style; optional ``combine_cfg_noise`` on the model.
            # MLX: materialize conditional output before the unconditional forward to avoid
            # lazy-graph aliasing across two ``forward`` calls sharing ``latents``.
            if neg_embeds is not None and getattr(config, "supports_guidance", False):
                self.ctx.eval(noise_cond)
                uncond_kwargs = {"txt_embeds": neg_embeds}
                uncond_kwargs.update(extra_cond)
                if sigmas is not None:
                    uncond_kwargs["sigmas"] = sigmas
                if timestep_embed_schedule is not None and i < len(timestep_embed_schedule):
                    uncond_kwargs["timestep_embed_value"] = timestep_embed_schedule[i]
                if encoder_type == "qwen_image":
                    uncond_kwargs["image_height"] = h
                    uncond_kwargs["image_width"] = w
                    uncond_kwargs["scheduler_timesteps"] = sched_ts
                    if neg_attn_mask is not None:
                        uncond_kwargs["encoder_hidden_states_mask"] = neg_attn_mask
                noise_uncond = model(latents, t, **uncond_kwargs)
                self.ctx.eval(noise_uncond)
                noise_pred = model.combine_cfg_noise(noise_cond, noise_uncond, guidance)
                if _cfg_renorm and getattr(config, "supports_guidance", False):
                    noise_pred = model.refine_cfg_noise(
                        noise_cond, noise_pred, cfg_renorm_min=_cfg_renorm_min,
                    )
            else:
                noise_pred = noise_cond

            latents = scheduler.step(noise_pred, t, latents)
            if getattr(self.ctx, "backend", None) == "mlx":
                self.ctx.eval(latents)

            # ── Hook ④: per-step callback (dynamic condition / logging) ──
            model.step_callback(i, latents, noise_pred)

            _image_pipeline_emit_denoise_progress(on_progress, i + 1, len(timesteps))
            if on_log:
                extra = ""
                if timestep_embed_schedule is not None and i < len(timestep_embed_schedule):
                    extra = f" t_embed={timestep_embed_schedule[i]:.6g}"
                on_log("info", f"Step {i + 1}/{len(timesteps)}{extra}")

        if ctx_exec.cancel_token.is_cancelled():
            return None

        # 5. VAE decode
        image = self._vae_decode(latents, entry, version_key or None, on_log=on_log)
        _image_pipeline_emit_post_progress(on_progress, n_steps=len(timesteps), within_post=0.5)

        if ctx_exec.cancel_token.is_cancelled():
            return None

        # 6. Save (task work dir, see TaskScheduler._work_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        work = Path(ctx_exec.work_dir)
        work.mkdir(parents=True, exist_ok=True)
        out_path = work / f"{model_key}_{seed}_{timestamp}.png"
        if hasattr(image, 'save'):
            image.save(str(out_path))
        _image_pipeline_emit_post_progress(on_progress, n_steps=len(timesteps), within_post=1.0)
        _image_pipeline_emit_complete(on_progress, len(timesteps))

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
        acts = getattr(entry, "actions", frozenset())
        if "edit" not in acts:
            raise RuntimeError(
                f"Model {model_key!r} is not registered for image edit (actions need rewrite/retouch/extend); "
                "refusing ImagePipeline.run_edit — see config/models_registry.json."
            )
        config_cls = get_config_class(entry.family)
        config = config_cls()
        family = getattr(entry, "family", "flux1")

        for param_key in (
            "text_encoder_out_layers",
            "vae_scale",
            "enable_thinking",
            "latent_noise_dtype",
            "inner_dim",
            "num_heads",
            "attn_head_dim",
            "num_layers",
            "num_single_layers",
            "joint_attention_dim",
        ):
            val = self._registry_scalar_default(entry, param_key, None)
            if val is not None and hasattr(config, param_key):
                if param_key == "text_encoder_out_layers" and isinstance(val, list):
                    setattr(config, param_key, tuple(int(x) for x in val))
                else:
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

        vae_cfg_pre: dict[str, Any] = {}
        vae_dir_pre = (bundle_root / "vae") if bundle_root else None
        if vae_dir_pre and (vae_dir_pre / "config.json").exists():
            import json

            with open(vae_dir_pre / "config.json") as f:
                vae_cfg_pre = json.load(f)
        vae_cls_pre = str(vae_cfg_pre.get("_class_name") or "")
        uses_encode_bridge = vae_cls_pre in ("AutoencoderKLFlux2", "AutoencoderKLQwenImage")

        if not uses_encode_bridge and _latent_hw(h, w) != _enc_hw(h, w):
            raise RuntimeError(
                f"Image edit (rewrite) is not wired for vae_scale={vae_scale} when the VAE encoder "
                f"outputs a {_enc_hw(h, w)[0]}×{_enc_hw(h, w)[1]} latent grid but the transformer expects "
                f"{_latent_hw(h, w)[0]}×{_latent_hw(h, w)[1]} (image {w}×{h}). Models with vae_scale≠8 "
                f"(e.g. Flux2) need a dedicated encode / pack bridge before img2img can run."
            )

        seed = request.seed if request.seed is not None else random.randint(0, 2 ** 32 - 1)
        img_f01 = self._pil_to_nchw_float01(pil, w, h)
        encoded = self._vae_encode_tensor(
            img_f01,
            entry,
            version_key or None,
            height_px=h,
            width_px=w,
            on_log=on_log,
        )
        if encoded.shape[1] != config.in_channels:
            raise RuntimeError(
                f"VAE encode produced {encoded.shape[1]} latent channels but model {model_key!r} "
                f"(family={family}) expects in_channels={config.in_channels}. "
                "Check bundle VAE config / model family alignment."
            )

        # source_fidelity ↔ mflux ``--image-strength`` (clamped [0, 1]).
        fidelity = float(request.source_fidelity)
        fidelity = max(0.0, min(1.0, fidelity))

        steps_default = self._registry_scalar_default(entry, "steps", 4)
        scheduler_registry = self._registry_scalar_default(entry, "scheduler", None)
        _meta_ed = request.metadata or {}
        scheduler_request = request.scheduler or _meta_ed.get("scheduler")
        scheduler_default = scheduler_request or scheduler_registry or "flow_match_euler"
        steps = int(request.steps) if request.steps is not None else int(steps_default)
        steps = max(1, steps)
        # mflux ``Config.init_time_step``: img2img starts denoising at this index; latent noise
        # uses ``sigmas[init]`` in ``(1 - sigma) * encoded + sigma * noise`` (not linear f·x+(1-f)·ε).
        init_timestep = 0
        if fidelity > 0.0:
            init_timestep = max(1, int(steps * fidelity))
        guidance_default = self._registry_scalar_default(entry, "guidance", 0.0)
        if request.guidance is not None:
            guidance = float(request.guidance)
        else:
            guidance = float(guidance_default)
        if not getattr(config, "supports_guidance", True):
            guidance = 0.0

        txt_embeds = None
        neg_embeds = None
        txt_attn_mask = None
        neg_attn_mask = None
        encoder_type = getattr(config, "encoder_type", "t5")
        if request.prompt and encoder_type != "t5":
            if bundle_root is None:
                raise RuntimeError(
                    f"Model {model_key!r} has no installed bundle at local_path "
                    f"(version={version_key or 'default'}); cannot load text encoder."
                )
            txt_embeds, txt_attn_mask = _encode_prompt_image_text(
                self.ctx,
                request.prompt,
                encoder_type=encoder_type,
                bundle_root=bundle_root,
                config=config,
            )
            if getattr(config, "supports_guidance", False) and guidance > 1.0:
                neg_txt = (request.negative_prompt or "").strip() or " "
                neg_embeds, neg_attn_mask = _encode_prompt_image_text(
                    self.ctx,
                    neg_txt,
                    encoder_type=encoder_type,
                    bundle_root=bundle_root,
                    config=config,
                )
        elif request.prompt and config.text_dim > 0:
            t5_dir, t5_tok = _t5_encoder_bundle_paths(bundle_root)
            enc = T5Encoder(self.ctx, t5_dir, tokenizer_path=t5_tok)
            txt_embeds = enc.encode([request.prompt])

        if ctx_exec.cancel_token.is_cancelled():
            return None

        model = self._load_model(family, config, entry, version_key or None)
        if model is None:
            raise RuntimeError(f"Failed to load model: {model_key}")
        model.after_load_weights(bundle_root=str(bundle_root) if bundle_root else None)
        self._apply_image_lora_adapters(family, model, request, on_log)
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
            init_timestep=init_timestep,
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

        if init_timestep > 0 and (not timesteps or int(timesteps[0]) != int(init_timestep)):
            raise RuntimeError(
                f"Image edit (rewrite): scheduler {scheduler_default!r} did not honor "
                f"init_timestep={init_timestep} (got timesteps={timesteps!r}). "
                "mflux-compatible img2img requires FlowMatchEuler / Linear / flow_match_euler_flux_dynamic."
            )
        if init_timestep >= steps:
            raise RuntimeError(
                f"Image edit (rewrite): source_fidelity={fidelity} implies init_timestep={init_timestep} "
                f">= steps={steps}; no denoising steps remain (mflux would also skip the loop)."
            )

        noise = self.ctx.seeded_randn(encoded.shape, seed, dtype=_denoise_latent_noise_dtype(self.ctx, config))
        if init_timestep == 0:
            latents = noise
        else:
            if sigmas is None:
                raise RuntimeError(
                    "Image edit (rewrite): scheduler produced no sigmas; cannot build mflux-style img2img latents."
                )
            sig_blend = sigmas[init_timestep]
            latents = (1.0 - sig_blend) * encoded + sig_blend * noise
        if getattr(self.ctx, "backend", None) == "mlx":
            self.ctx.eval(latents)

        if on_log:
            on_log(
                "info",
                f"edit rewrite model={model_key} family={family} size={w}x{h} seed={seed} "
                f"steps={steps} init_timestep={init_timestep} scheduler={scheduler_default} "
                f"source_fidelity={fidelity}",
            )

        latents, extra_cond = model.before_denoise(latents, timesteps, sigmas, **extra_cond)

        for i, t in enumerate(timesteps):
            if ctx_exec.cancel_token.is_cancelled():
                return None
            te_idx = init_timestep + i
            model_kwargs = {"txt_embeds": txt_embeds} if txt_embeds is not None else {}
            model_kwargs.update(extra_cond)
            if sigmas is not None:
                model_kwargs["sigmas"] = sigmas
            if timestep_embed_schedule is not None and te_idx < len(timestep_embed_schedule):
                model_kwargs["timestep_embed_value"] = timestep_embed_schedule[te_idx]
            if encoder_type == "qwen_image":
                model_kwargs["image_height"] = h
                model_kwargs["image_width"] = w
                model_kwargs["scheduler_timesteps"] = sched_ts
                if txt_attn_mask is not None:
                    model_kwargs["encoder_hidden_states_mask"] = txt_attn_mask

            noise_cond = model(latents, t, **model_kwargs)

            if neg_embeds is not None and getattr(config, "supports_guidance", False):
                self.ctx.eval(noise_cond)
                uncond_kwargs = {"txt_embeds": neg_embeds}
                uncond_kwargs.update(extra_cond)
                if sigmas is not None:
                    uncond_kwargs["sigmas"] = sigmas
                if timestep_embed_schedule is not None and te_idx < len(timestep_embed_schedule):
                    uncond_kwargs["timestep_embed_value"] = timestep_embed_schedule[te_idx]
                if encoder_type == "qwen_image":
                    uncond_kwargs["image_height"] = h
                    uncond_kwargs["image_width"] = w
                    uncond_kwargs["scheduler_timesteps"] = sched_ts
                    if neg_attn_mask is not None:
                        uncond_kwargs["encoder_hidden_states_mask"] = neg_attn_mask
                noise_uncond = model(latents, t, **uncond_kwargs)
                self.ctx.eval(noise_uncond)
                noise_pred = model.combine_cfg_noise(noise_cond, noise_uncond, guidance)
                if _cfg_renorm and getattr(config, "supports_guidance", False):
                    noise_pred = model.refine_cfg_noise(
                        noise_cond, noise_pred, cfg_renorm_min=_cfg_renorm_min,
                    )
            else:
                noise_pred = noise_cond

            latents = scheduler.step(noise_pred, t, latents)
            if getattr(self.ctx, "backend", None) == "mlx":
                self.ctx.eval(latents)
            model.step_callback(i, latents, noise_pred)
            _image_pipeline_emit_denoise_progress(on_progress, i + 1, len(timesteps))
            if on_log:
                extra = ""
                if timestep_embed_schedule is not None and te_idx < len(timestep_embed_schedule):
                    extra = f" t_embed={timestep_embed_schedule[te_idx]:.6g}"
                on_log("info", f"Step {i + 1}/{len(timesteps)}{extra}")

        if ctx_exec.cancel_token.is_cancelled():
            return None

        image = self._vae_decode(latents, entry, version_key or None, on_log=on_log)
        _image_pipeline_emit_post_progress(on_progress, n_steps=len(timesteps), within_post=0.5)
        if ctx_exec.cancel_token.is_cancelled():
            return None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        work = Path(ctx_exec.work_dir)
        work.mkdir(parents=True, exist_ok=True)
        out_path = work / f"{model_key}_edit_{seed}_{timestamp}.png"
        if hasattr(image, "save"):
            image.save(str(out_path))
        _image_pipeline_emit_post_progress(on_progress, n_steps=len(timesteps), within_post=1.0)
        _image_pipeline_emit_complete(on_progress, len(timesteps))

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

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _apply_image_lora_adapters(
        self,
        family: str,
        model: Any,
        request: ImageGenerationRequest | ImageEditRequest,
        on_log: Callable[..., None] | None,
    ) -> None:
        adapters = getattr(request, "adapters", None) or []
        if not adapters:
            return
        if family not in ("flux2", "z_image", "qwen_image"):
            raise RuntimeError(
                "LoRA adapters require in-engine merging; supported image families on MLX are "
                f"flux2, z_image, and qwen_image (this model is family={family!r}). Remove adapters or switch model."
            )
        from backend.engine.runtime.mlx import MLXContext

        if not isinstance(self.ctx, MLXContext):
            raise RuntimeError(
                "LoRA merging for Flux2 / Z-Image / Qwen Image is only implemented on the MLX runtime; "
                f"current runtime is {type(self.ctx).__name__}."
            )
        base_model_id, _ = parse_model_version(request.model)
        if family == "flux2":
            from backend.engine.families.flux2.lora_mlx import merge_flux2_lora_adapters

            merge_flux2_lora_adapters(
                model,
                adapters,
                base_model_id=base_model_id,
                project_root=self._project_root,
                registry=self._registry,
                ctx=self.ctx,
                on_log=on_log,
            )
        elif family == "z_image":
            from backend.engine.families.z_image.lora_mlx import merge_z_image_lora_adapters

            patch_size = int(getattr(getattr(model, "config", None), "patch_size", 2) or 2)
            merge_z_image_lora_adapters(
                model,
                adapters,
                base_model_id=base_model_id,
                project_root=self._project_root,
                registry=self._registry,
                ctx=self.ctx,
                patch_size=patch_size,
                on_log=on_log,
            )
        else:
            from backend.engine.families.qwen.lora_mlx import merge_qwen_image_lora_adapters

            merge_qwen_image_lora_adapters(
                model,
                adapters,
                base_model_id=base_model_id,
                project_root=self._project_root,
                registry=self._registry,
                ctx=self.ctx,
                on_log=on_log,
            )

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

        from backend.engine.common.safetensors_affine_quant import read_bundle_affine_bits_if_quantized

        bundle_affine_bits = read_bundle_affine_bits_if_quantized(w, tp)

        model.load_weights(
            list(w.items()),
            strict=False,
            ctx=self.ctx,
            bundle_affine_bits=bundle_affine_bits,
        )
        self.ctx.eval(*[p for _, p in model.parameters()])
        return model

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
        from backend.engine.common.vae import VAEDecoder, remap_vae_weights, vae_output_to_uint8_hwc

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

        if str(vae_cfg.get("_class_name") or "") == "AutoencoderKLQwenImage":
            from backend.engine.common.vae.qwen_image import QwenVAE, apply_qwen_vae_weights_from_bundle

            if bundle_root is None:
                raise RuntimeError("Qwen VAE decode: missing bundle_root")
            z = self._qwen_unpack_latents_nchw(latents)
            vae_q = QwenVAE()
            apply_qwen_vae_weights_from_bundle(vae_q, bundle_root, project_root=self._project_root)
            decoded = vae_q.decode(z)
            if getattr(decoded, "ndim", 0) == 5 and int(decoded.shape[2]) == 1:
                decoded = decoded[:, :, 0, :, :]
            if on_log:
                on_log("info", f"vae_decode qwen unpacked_z_shape={tuple(z.shape)} decoded_shape={tuple(decoded.shape)}")
            pixels = vae_output_to_uint8_hwc(decoded, self.ctx)
            return Image.fromarray(pixels)

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
