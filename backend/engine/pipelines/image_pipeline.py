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
    AdapterRef,
    EngineResult,
    ExecutionContext,
    ImageGenerationRequest,
    ImageEditRequest,
    LogEvent,
    parse_model_version,
    parse_size,
    work_title_metadata,
)
from backend.core.registry_format import registry_declares_action
from backend.engine.common.bundle_layout import t5_encoder_bundle_paths
from backend.engine.common.cache import ModelCache
from backend.engine.common.pipeline_registry import (
    local_bundle_root as _local_bundle_root_fn,
    registry_scalar_default as _registry_scalar_default_fn,
    resolve_project_path as _resolve_project_path_fn,
    resolve_version_block as _resolve_version_block_fn,
)
from backend.engine.common.runtime_contracts import (
    FamilyRuntimeContract,
    SchedulerSemanticsResolver,
)
from backend.engine.common.schedulers import get_scheduler
from backend.engine.common.text_encoders import T5Encoder
from backend.engine._transformer_registry import (
    encode_prompt_with_image_text_encoder as _encode_prompt_image_text,
    get_transformer_class as _get_transformer_class,
    get_weight_remap as _get_weight_remap,
    merge_image_lora_adapters as _merge_image_lora_adapters,
)
from backend.engine.config.model_configs import (
    apply_image_bundle_config_merger,
    assert_image_family_contract,
    get_config_class,
)
from backend.engine.runtime._base import RuntimeContext
from backend.engine.vae_codec_registry import (
    get_vae_decode_handler,
    get_vae_encode_handler,
    qwen_pack_latents_nchw,
    qwen_unpack_latents_nchw,
)
from backend.engine.pipelines.pipeline_progress import (
    DENOISE_PROGRESS_SHARE,
    emit_complete,
    emit_denoise_progress,
    emit_phase,
    emit_post_progress,
    pipeline_graph_step,
    timestep_embed_schedule_from_scheduler,
    validate_bundle_graph_step,
)


def _resolve_image_preview_settings(entry: Any) -> tuple[str, int, int]:
    """Return (preview_mode, interval_steps, max_edge_px)."""
    mode = _registry_scalar_default_fn(entry, "preview_mode", None)
    if mode is None:
        family = str(getattr(entry, "family", "") or "")
        raw = getattr(entry, "raw", None) or {}
        model_type = str(raw.get("type", "") if isinstance(raw, dict) else "")
        if model_type != "diffusion" or family in ("seedvr2",):
            mode = "none"
        else:
            mode = "stream"
    mode = str(mode).strip().lower()
    if mode not in ("stream", "none"):
        mode = "none"
    interval = int(_registry_scalar_default_fn(entry, "preview_interval_steps", 2) or 2)
    max_edge = int(_registry_scalar_default_fn(entry, "preview_max_edge", 512) or 512)
    return mode, max(1, interval), max(64, min(2048, max_edge))


def _image_pipeline_cfg_noise_pred(
    ctx: RuntimeContext,
    model: Any,
    config: Any,
    latents: Any,
    t: Any,
    guidance: float,
    txt_embeds: Any,
    neg_embeds: Any,
    model_kwargs: dict[str, Any],
    uncond_overrides: dict[str, Any],
    *,
    cfg_renorm: bool,
    cfg_renorm_min: float,
) -> Any:
    """Classifier-free guidance merge — polymorphic ``forward_cfg`` when available (MLX Z-Image)."""
    forward_cfg = getattr(model, "forward_cfg", None)
    fibo_batched_cfg = (
        neg_embeds is None
        and txt_embeds is not None
        and getattr(config, "structured_prompt", False)
        and getattr(txt_embeds, "shape", None) is not None
        and int(txt_embeds.shape[0]) == 2
    )
    if (
        (neg_embeds is not None or fibo_batched_cfg)
        and getattr(config, "supports_guidance", False)
        and guidance > 0.0
        and callable(forward_cfg)
        and getattr(config, "use_mlx_cfg_fusion", True)
        and getattr(ctx, "backend", None) == "mlx"
    ):
        cfg_kwargs = {
            k: v for k, v in model_kwargs.items()
            if k not in ("txt_embeds", "neg_embeds", "redux_txt_embeds", "fill_static_packed")
        }
        return forward_cfg(
            latents,
            t,
            txt_embeds,
            neg_embeds,
            guidance,
            cfg_renorm=cfg_renorm,
            cfg_renorm_min=cfg_renorm_min,
            **cfg_kwargs,
        )

    noise_cond = model(latents, t, **model_kwargs)
    if neg_embeds is not None and getattr(config, "supports_guidance", False):
        if getattr(ctx, "backend", None) == "mlx":
            ctx.eval(noise_cond)
        uncond_kwargs = {
            k: v for k, v in model_kwargs.items()
            if k not in ("txt_embeds", "redux_txt_embeds", "fill_static_packed")
        }
        uncond_kwargs["txt_embeds"] = neg_embeds
        uncond_kwargs.update(uncond_overrides)
        noise_uncond = model(latents, t, **uncond_kwargs)
        if getattr(ctx, "backend", None) == "mlx":
            ctx.eval(noise_uncond)
        noise_pred = model.combine_cfg_noise(noise_cond, noise_uncond, guidance)
        if cfg_renorm and getattr(config, "supports_guidance", False):
            noise_pred = model.refine_cfg_noise(
                noise_cond, noise_pred, cfg_renorm_min=cfg_renorm_min,
            )
        return noise_pred
    return noise_cond


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
        self._scheduler_semantics_resolver = SchedulerSemanticsResolver()

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

    def _load_vae_dir_cfg_weights(
        self,
        entry,
        version_key: str | None,
    ) -> tuple[Path, dict[str, Any], dict[str, Any]]:
        from backend.engine.common.vae import load_vae_weight_dict, read_vae_dir_config

        bundle_root = _local_bundle_root_fn(self._project_root, entry, version_key)
        vae_dir = (bundle_root / "vae") if bundle_root else None
        if vae_dir is None or not vae_dir.exists():
            raise RuntimeError(f"VAE: no vae directory under bundle {bundle_root}")

        vae_cfg, _, _ = read_vae_dir_config(vae_dir)
        vae_weights = load_vae_weight_dict(self.ctx, vae_dir)
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

    def _vae_encode_flux2(
        self,
        *,
        image_n11: Any,
        enc: Any,
        vae_weights: dict[str, Any],
        vae_cfg: dict[str, Any],
        latent_c: int,
        scaling_factor: float,
        shift_factor: float,
        on_log: Callable | None,
    ) -> Any:
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

        image_n11 = image_nchw_f01 * 2.0 - 1.0

        _, vae_cfg, vae_weights = self._load_vae_dir_cfg_weights(entry, version_key)
        vae_cls = str(vae_cfg.get("_class_name") or "")
        entry_family = str(getattr(entry, "family", "") or "")

        encode_handler = get_vae_encode_handler(vae_cls, entry_family=entry_family)
        if encode_handler is not None:
            bundle_root = _local_bundle_root_fn(self._project_root, entry, version_key)
            return encode_handler(
                ctx=self.ctx,
                image_n11=image_n11,
                bundle_root=bundle_root,
                project_root=self._project_root,
                height_px=height_px,
                width_px=width_px,
                on_log=on_log,
            )

        from backend.engine.common.vae import infer_latent_channels

        scaling_factor = float(vae_cfg.get("scaling_factor", 1.0))
        shift_factor = float(vae_cfg.get("shift_factor", 0.0))

        latent_c = infer_latent_channels(vae_cfg, vae_weights)
        enc = VAEEncoder(
            latent_channels=latent_c,
            ctx=self.ctx,
            scaling_factor=scaling_factor,
            shift_factor=shift_factor,
        )
        enc_items = prepare_vae_encoder_weight_items(vae_weights)
        loaded, skipped = enc.load_weights(enc_items, strict=False)
        if (
            getattr(self.ctx, "backend", None) == "mlx"
            and str(getattr(entry, "family", "")).lower() == "flux1"
        ):
            enc.cast_floating_params(self.ctx.bfloat16())
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
            return self._vae_encode_flux2(
                image_n11=image_n11,
                enc=enc,
                vae_weights=vae_weights,
                vae_cfg=vae_cfg,
                latent_c=latent_c,
                scaling_factor=scaling_factor,
                shift_factor=shift_factor,
                on_log=on_log,
            )

        latent5 = enc.encode(image_n11)
        z = latent5[:, :, 0, :, :] if getattr(latent5, "ndim", 0) == 5 else latent5
        return z

    def _qwen_pack_latents_nchw(self, encoded_b16hw: Any, height_px: int, width_px: int) -> Any:
        return qwen_pack_latents_nchw(self.ctx, encoded_b16hw, height_px, width_px)

    def _qwen_unpack_latents_nchw(self, packed_b64hw: Any) -> Any:
        return qwen_unpack_latents_nchw(self.ctx, packed_b64hw)

    def _apply_registry_config_overrides(self, entry: Any, config: Any) -> None:
        for param_key in (
            "text_encoder_out_layers",
            "vae_scale",
            "enable_thinking",
            "latent_noise_dtype",
            "max_seq_len",
            "inner_dim",
            "num_heads",
            "attn_head_dim",
            "num_layers",
            "num_single_layers",
            "joint_attention_dim",
            "edit_conditioning_concat",
            "edit_rmbg_composite_output",
            "edit_use_vl_vision",
            "edit_conditioning_latent_concat",
            "patch_token_dim",
        ):
            val = _registry_scalar_default_fn(entry, param_key, None)
            if val is not None and hasattr(config, param_key):
                if param_key == "text_encoder_out_layers" and isinstance(val, list):
                    setattr(config, param_key, tuple(int(x) for x in val))
                else:
                    setattr(config, param_key, val)
        sg = _registry_scalar_default_fn(entry, "supports_guidance", None)
        if sg is not None:
            config.supports_guidance = bool(sg)

    def _encode_image_text_conditioning(
        self,
        *,
        prompt: str,
        negative_prompt: str | None,
        bundle_root: Path | None,
        config: Any,
        guidance: float,
        runtime_contract: FamilyRuntimeContract,
    ) -> tuple[Any, Any, Any, Any, Any, Any, str]:
        txt_embeds = None
        neg_embeds = None
        txt_attn_mask = None
        neg_attn_mask = None
        pooled_embeds = None
        neg_pooled_embeds = None
        encoder_type = getattr(config, "encoder_type", "t5")
        if prompt and encoder_type != "t5":
            if bundle_root is None:
                raise RuntimeError("Cannot load text encoder: model bundle is not installed at local_path.")
            if (
                encoder_type == "fibo"
                and getattr(config, "structured_prompt", False)
                and float(guidance) > 1.0
            ):
                from backend.engine._transformer_registry import get_text_encoder

                enc_cls = get_text_encoder(encoder_type)
                enc_dir = bundle_root / "text_encoder"
                tok_dir = bundle_root / "tokenizer"
                if not tok_dir.exists():
                    tok_dir = enc_dir
                enc = enc_cls(
                    self.ctx,
                    str(enc_dir),
                    tokenizer_path=str(tok_dir),
                    max_seq_len=getattr(config, "max_seq_len", 2048),
                )
                txt_embeds, txt_attn_mask = enc.encode_prompt_cfg(
                    prompt,
                    negative_prompt,
                    guidance=float(guidance),
                )
                pooled_embeds = None
            else:
                txt_embeds, txt_attn_mask, pooled_embeds = _encode_prompt_image_text(
                    self.ctx,
                    prompt,
                    encoder_type=encoder_type,
                    bundle_root=bundle_root,
                    config=config,
                )
            if runtime_contract.should_encode_negative_prompt(guidance):
                neg_txt = (negative_prompt or "").strip() or " "
                neg_embeds, neg_attn_mask, neg_pooled_embeds = _encode_prompt_image_text(
                    self.ctx,
                    neg_txt,
                    encoder_type=encoder_type,
                    bundle_root=bundle_root,
                    config=config,
                )
        elif prompt and config.text_dim > 0:
            t5_dir, t5_tok = t5_encoder_bundle_paths(bundle_root)
            enc = T5Encoder(self.ctx, t5_dir, tokenizer_path=t5_tok)
            txt_embeds = enc.encode([prompt])
        return (
            txt_embeds,
            neg_embeds,
            txt_attn_mask,
            neg_attn_mask,
            pooled_embeds,
            neg_pooled_embeds,
            encoder_type,
        )

    def _image_encode_load_for_inference(
        self,
        *,
        request: ImageGenerationRequest | ImageEditRequest,
        bundle_root: Path | None,
        config: Any,
        guidance: float,
        runtime_contract: FamilyRuntimeContract,
        family: str,
        entry: Any,
        version_key: str | None,
        model_key: str,
        steps: int,
        ctx_exec: ExecutionContext,
        on_progress: Callable | None,
        on_log: Callable | None,
    ) -> tuple[Any, dict[str, Any], Any, Any, Any, Any, Any, Any, str] | None:
        emit_phase(on_progress, phase="encoding", progress=0.02, n_steps=steps)
        pipeline_graph_step("encode_prompt", on_log)
        (
            txt_embeds,
            neg_embeds,
            txt_attn_mask,
            neg_attn_mask,
            pooled_embeds,
            neg_pooled_embeds,
            encoder_type,
        ) = self._encode_image_text_conditioning(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            bundle_root=bundle_root,
            config=config,
            guidance=guidance,
            runtime_contract=runtime_contract,
        )
        if ctx_exec.cancel_token.is_cancelled():
            return None

        emit_phase(on_progress, phase="loading_model", progress=0.08, n_steps=steps)
        pipeline_graph_step("load_transformer", on_log)
        allow_cache = not (getattr(request, "adapters", None) or [])
        model = self._load_model(
            family, config, entry, version_key or None, allow_cache=allow_cache
        )
        if model is None:
            raise RuntimeError(f"Failed to load model: {model_key}")

        model.after_load_weights(bundle_root=str(bundle_root) if bundle_root else None)
        self._apply_image_lora_adapters(family, model, request, on_log)
        extra_cond = model.prepare_conditioning(
            request, bundle_root=str(bundle_root) if bundle_root else None
        )
        return (
            model,
            extra_cond,
            txt_embeds,
            neg_embeds,
            txt_attn_mask,
            neg_attn_mask,
            pooled_embeds,
            neg_pooled_embeds,
            encoder_type,
        )

    def _finalize_image_from_latents(
        self,
        *,
        latents: Any,
        timesteps: Any,
        entry: Any,
        version_key: str | None,
        model_key: str,
        seed: int,
        request: ImageGenerationRequest | ImageEditRequest,
        ctx_exec: ExecutionContext,
        steps: int,
        guidance: float,
        w: int,
        h: int,
        on_progress: Callable | None,
        on_log: Callable | None,
        name_infix: str = "",
        post_decode: Callable[[Any], Any] | None = None,
        extra_meta: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, Any]] | None:
        pipeline_graph_step("decode_vae", on_log)
        image = self._vae_decode(latents, entry, version_key or None, on_log=on_log)
        if post_decode is not None:
            image = post_decode(image)
        emit_post_progress(on_progress, n_steps=len(timesteps), within_post=0.5)
        if ctx_exec.cancel_token.is_cancelled():
            return None

        pipeline_graph_step("save_asset", on_log)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        work = Path(ctx_exec.work_dir)
        work.mkdir(parents=True, exist_ok=True)
        out_path = work / f"{model_key}{name_infix}_{seed}_{timestamp}.png"
        if hasattr(image, "save"):
            image.save(str(out_path))
        emit_post_progress(on_progress, n_steps=len(timesteps), within_post=1.0)
        emit_complete(on_progress, len(timesteps))

        meta: dict[str, Any] = {
            "model": request.model,
            "seed": seed,
            "prompt": request.prompt,
            "steps": steps,
            "guidance": guidance,
            "width": w,
            "height": h,
            "mime_type": "image/png",
        }
        if extra_meta:
            meta.update(extra_meta)
        meta.update(work_title_metadata(request.title))
        return str(out_path), meta

    def _build_vae_preview_session(
        self,
        entry: Any,
        version_key: str | None,
        *,
        on_log: Callable | None = None,
    ) -> dict[str, Any] | None:
        from backend.engine.common.vae import build_standard_vae_preview_session, read_vae_dir_config

        bundle_root = _local_bundle_root_fn(self._project_root, entry, version_key)
        vae_dir = (bundle_root / "vae") if bundle_root else None
        vae_cfg, _, _ = read_vae_dir_config(vae_dir)
        vae_cls = str(vae_cfg.get("_class_name") or "")
        entry_family = str(getattr(entry, "family", "") or "")
        if get_vae_decode_handler(vae_cls, entry_family=entry_family) is not None:
            return None
        return build_standard_vae_preview_session(self.ctx, vae_dir, on_log=on_log)

    def _vae_decode_with_preview_session(
        self,
        latents: Any,
        entry: Any,
        version_key: str | None,
        preview_state: dict[str, Any],
        *,
        on_log: Callable | None = None,
    ) -> Any:
        from backend.engine.common.vae import (
            apply_flux2_latent_preprocess_if_enabled,
            reshape_packed_latents_to_nchw,
            vae_forward_to_pil,
        )

        session = preview_state.get("vae_session")
        if session is None:
            try:
                session = self._build_vae_preview_session(
                    entry, version_key, on_log=on_log
                )
            except Exception as exc:
                preview_state["vae_session"] = False
                if on_log:
                    on_log("warning", f"preview VAE session build failed: {exc}")
                return self._vae_decode(
                    latents, entry, version_key, on_log=on_log
                )
            preview_state["vae_session"] = session if session else False

        if not session or session is False:
            return self._vae_decode(latents, entry, version_key, on_log=on_log)

        from backend.engine.common.vae import reshape_packed_latents_to_nchw

        z = reshape_packed_latents_to_nchw(self._latents_for_vae_preview(latents))

        if session.get("use_special_preprocess"):
            from backend.engine.common.vae import apply_flux2_latent_preprocess_if_enabled

            z, _, _ = apply_flux2_latent_preprocess_if_enabled(
                self.ctx,
                z,
                session["vae_cfg"],
                session["vae_weights"],
                session["orig_scaling"],
                session["orig_shift"],
            )

        return vae_forward_to_pil(self.ctx, session["vae"], z)

    @staticmethod
    def _latents_for_vae_preview(latents: Any) -> Any:
        z = latents
        if getattr(z, "ndim", None) == 5 and int(z.shape[2]) == 1:
            z = z[:, :, 0, :, :]
        return z

    def _warm_step_preview_decoders(
        self,
        entry: Any,
        version_key: str | None,
        preview_state: dict[str, Any],
        *,
        config: Any = None,
        on_log: Callable | None = None,
    ) -> None:
        bundle_root = _local_bundle_root_fn(self._project_root, entry, version_key)
        if bool(getattr(config, "vae_preview_warmup", False)) and bundle_root and preview_state.get("flux2_vae") is None:
            from backend.engine.families.flux2.vae_mlx import load_flux2_vae_decoder

            try:
                preview_state["flux2_vae"] = load_flux2_vae_decoder(
                    self.ctx, bundle_root, on_log=on_log
                )
            except Exception as exc:
                preview_state["flux2_vae"] = False
                if on_log:
                    on_log("warning", f"flux2 preview VAE warmup failed: {exc}")

    def _decode_latents_for_step_preview(
        self,
        latents: Any,
        entry: Any,
        version_key: str | None,
        preview_state: dict[str, Any],
        *,
        packed_denoise: bool,
        flux_unpack: Callable[..., Any] | None,
        latent_h: int,
        latent_w: int,
        on_log: Callable | None = None,
    ) -> Any:
        """Same decode semantics as final frame; prefer warmed VAE session when available."""
        decode_latents = self._latents_for_vae_preview(latents)
        if packed_denoise and flux_unpack is not None:
            decode_latents = flux_unpack(self.ctx, latents, latent_h, latent_w)
            decode_latents = self._latents_for_vae_preview(decode_latents)

        flux2_vae = preview_state.get("flux2_vae")
        if flux2_vae not in (None, False):
            from backend.engine.families.flux2.vae_mlx import decode_flux2_latents_with_model

            return decode_flux2_latents_with_model(
                self.ctx, flux2_vae, decode_latents, on_log=on_log
            )

        session = preview_state.get("vae_session")
        if session and session is not False:
            return self._vae_decode_with_preview_session(
                decode_latents,
                entry,
                version_key,
                preview_state,
                on_log=on_log,
            )

        return self._vae_decode(decode_latents, entry, version_key, on_log=on_log)

    def _maybe_emit_step_preview(
        self,
        *,
        step_index_0based: int,
        n_steps: int,
        latents: Any,
        entry: Any,
        version_key: str | None,
        ctx_exec: ExecutionContext,
        on_progress: Callable[..., None] | None,
        preview_interval: int,
        preview_max_edge: int,
        preview_state: dict[str, Any],
        packed_denoise: bool,
        flux_unpack: Callable[..., Any] | None,
        latent_h: int,
        latent_w: int,
    ) -> None:
        interval = max(1, int(preview_interval))
        step_1 = step_index_0based + 1
        is_last = step_1 >= n_steps
        if step_1 > 1 and step_1 % interval != 0 and not is_last:
            return
        step_log = preview_state.get("on_log")
        try:
            from PIL import Image

            image = self._decode_latents_for_step_preview(
                latents,
                entry,
                version_key,
                preview_state,
                packed_denoise=packed_denoise,
                flux_unpack=flux_unpack,
                latent_h=latent_h,
                latent_w=latent_w,
                on_log=step_log,
            )
            if image is None:
                return
            if not hasattr(image, "save"):
                return
            pil = image
            if max(pil.size) > preview_max_edge:
                pil = pil.copy()
                pil.thumbnail(
                    (preview_max_edge, preview_max_edge),
                    Image.Resampling.BILINEAR,
                )
            work = Path(ctx_exec.work_dir)
            work.mkdir(parents=True, exist_ok=True)
            out_path = work / "preview_latest.png"
            pil.save(str(out_path), format="PNG", optimize=True)
            nbytes = out_path.stat().st_size if out_path.is_file() else 0
            n = max(1, int(n_steps))
            p = DENOISE_PROGRESS_SHARE * (step_1 / n)
            # Step preview image: frontend polls GET /api/tasks/{task_id}/preview (not SSE).
            if on_progress is not None:
                on_progress(p, step_1, n, None, "denoising")
            if preview_state.get("on_log"):
                preview_state["on_log"](
                    "info",
                    f"preview step {step_1}/{n_steps} saved {nbytes} bytes -> {out_path.name}",
                )
            if getattr(self.ctx, "backend", None) == "mlx":
                self.ctx.eval()
        except Exception as exc:
            shape = getattr(latents, "shape", None)
            if preview_state.get("on_log"):
                preview_state["on_log"](
                    "error",
                    f"step preview failed at {step_1}/{n_steps}: {exc} latent_shape={shape}",
                )

    def _denoise_steps(
        self,
        *,
        model: Any,
        scheduler: Any,
        timesteps: list[Any],
        latents: Any,
        config: Any,
        runtime_contract: FamilyRuntimeContract,
        guidance: float,
        txt_embeds: Any,
        neg_embeds: Any,
        pooled_embeds: Any,
        neg_pooled_embeds: Any,
        txt_attn_mask: Any,
        neg_attn_mask: Any,
        encoder_type: str,
        width: int,
        height: int,
        sched_ts: Any,
        sigmas: Any,
        timestep_embed_schedule: list[float] | None,
        extra_cond: dict[str, Any],
        semantics: Any,
        ctx_exec: ExecutionContext,
        on_progress: Callable[..., None] | None,
        on_log: Callable[..., None] | None,
        preview_mode: str = "none",
        preview_interval: int = 2,
        preview_max_edge: int = 512,
        preview_state: dict[str, Any] | None = None,
        entry: Any = None,
        version_key: str | None = None,
        timestep_offset: int = 0,
        packed_denoise: bool = False,
        flux_pack: Callable[..., Any] | None = None,
        flux_unpack: Callable[..., Any] | None = None,
        latent_h: int = 0,
        latent_w: int = 0,
    ) -> Any | None:
        for i, t in enumerate(timesteps):
            if ctx_exec.cancel_token.is_cancelled():
                return None
            te_idx = timestep_offset + i
            t_embed = (
                timestep_embed_schedule[te_idx]
                if timestep_embed_schedule is not None and te_idx < len(timestep_embed_schedule)
                else None
            )
            model_kwargs = runtime_contract.compose_step_kwargs(
                txt_embeds=txt_embeds,
                pooled_embeds=pooled_embeds,
                extra_cond=extra_cond,
                guidance=guidance,
                sigmas=sigmas,
                timestep_embed_value=t_embed,
                encoder_type=encoder_type,
                image_height=height,
                image_width=width,
                scheduler_timesteps=sched_ts,
                txt_attn_mask=txt_attn_mask,
            )
            latents_model = (
                flux_unpack(self.ctx, latents, latent_h, latent_w)  # type: ignore[misc]
                if packed_denoise and flux_unpack is not None
                else latents
            )
            fibo_batched_cfg = (
                neg_embeds is None
                and txt_embeds is not None
                and getattr(config, "structured_prompt", False)
                and getattr(txt_embeds, "shape", None) is not None
                and int(txt_embeds.shape[0]) == 2
            )
            if (
                getattr(config, "supports_guidance", False)
                and (neg_embeds is not None or fibo_batched_cfg)
            ):
                uncond_overrides = runtime_contract.compose_uncond_overrides(
                    pooled_embeds=neg_pooled_embeds,
                    guidance=guidance,
                    encoder_type=encoder_type,
                    image_height=height,
                    image_width=width,
                    scheduler_timesteps=sched_ts,
                    neg_attn_mask=neg_attn_mask,
                )
                noise_pred = _image_pipeline_cfg_noise_pred(
                    self.ctx,
                    model,
                    config,
                    latents_model,
                    t,
                    guidance,
                    txt_embeds,
                    neg_embeds,
                    model_kwargs,
                    uncond_overrides,
                    cfg_renorm=semantics.cfg_renorm,
                    cfg_renorm_min=semantics.cfg_renorm_min,
                )
            else:
                noise_pred = model(latents_model, t, **model_kwargs)
            if packed_denoise and flux_pack is not None:
                noise_pred = flux_pack(self.ctx, noise_pred)
            latents = scheduler.step(noise_pred, t, latents)
            if getattr(self.ctx, "backend", None) == "mlx":
                self.ctx.eval(latents)
            model.step_callback(i, latents, noise_pred)
            emit_denoise_progress(on_progress, i + 1, len(timesteps))
            if (
                preview_mode == "stream"
                and preview_state is not None
                and entry is not None
            ):
                self._maybe_emit_step_preview(
                    step_index_0based=i,
                    n_steps=len(timesteps),
                    latents=latents,
                    entry=entry,
                    version_key=version_key,
                    ctx_exec=ctx_exec,
                    on_progress=on_progress,
                    preview_interval=preview_interval,
                    preview_max_edge=preview_max_edge,
                    preview_state=preview_state,
                    packed_denoise=packed_denoise,
                    flux_unpack=flux_unpack,
                    latent_h=latent_h,
                    latent_w=latent_w,
                )
            if on_log and (i == 0 or (i + 1) % max(1, preview_interval) == 0 or i + 1 == len(timesteps)):
                extra = f" t_embed={t_embed:.6g}" if t_embed is not None else ""
                on_log("info", f"Step {i + 1}/{len(timesteps)}{extra}")
        return latents

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

        Returns: ``list[(output_path, metadata_dict)]`` or ``None`` (cancelled)
        """
        request = self._augment_request_for_structural_guide(request)
        model_key, version_key = parse_model_version(request.model)
        w, h = parse_size(request.size)
        base_seed = request.seed if request.seed is not None else random.randint(0, 2 ** 32 - 1)
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

        self._apply_registry_config_overrides(entry, config)
        assert_image_family_contract(family, config)
        runtime_contract = FamilyRuntimeContract(family=family, config=config)

        if ctx_exec.cancel_token.is_cancelled():
            return None

        bundle_root = _local_bundle_root_fn(self._project_root, entry, version_key or None)
        validate_bundle_graph_step(
            bundle_root, family=family, model_id=model_key, on_log=on_log
        )

        steps_default = _registry_scalar_default_fn(entry, "steps", 4)
        guidance_default = _registry_scalar_default_fn(entry, "guidance", 0.0)
        _meta = request.metadata or {}

        steps = int(request.steps) if request.steps is not None else int(steps_default)
        steps = max(1, steps)
        guidance = float(request.guidance) if request.guidance is not None else float(guidance_default)
        guidance = runtime_contract.resolve_guidance_scalar(guidance)
        preview_mode, preview_interval, preview_max_edge = _resolve_image_preview_settings(entry)
        preview_state: dict[str, Any] = {}

        enc_loaded = self._image_encode_load_for_inference(
            request=request,
            bundle_root=bundle_root,
            config=config,
            guidance=guidance,
            runtime_contract=runtime_contract,
            family=family,
            entry=entry,
            version_key=version_key,
            model_key=model_key,
            steps=steps,
            ctx_exec=ctx_exec,
            on_progress=on_progress,
            on_log=on_log,
        )
        if enc_loaded is None:
            return None
        (
            model,
            extra_cond,
            txt_embeds,
            neg_embeds,
            txt_attn_mask,
            neg_attn_mask,
            pooled_embeds,
            neg_pooled_embeds,
            encoder_type,
        ) = enc_loaded

        structural_cleanup: Callable[[], None] | None = None
        try:
            extra_cond, structural_cleanup = self._attach_structural_conditioning(
                request=request,
                family=family,
                model=model,
                entry=entry,
                version_key=version_key,
                extra_cond=extra_cond,
                width=w,
                height=h,
                ctx_exec=ctx_exec,
                on_log=on_log,
            )
        except Exception:
            if structural_cleanup is not None:
                structural_cleanup()
            raise

        # 3. Scheduler (registry default + request params, zero family branching)
        semantics = self._scheduler_semantics_resolver.resolve(
            entry=entry,
            config=config,
            request_scheduler=request.scheduler,
            request_metadata=_meta,
            steps=steps,
            width=w,
            height=h,
        )
        scheduler_default = semantics.scheduler_name
        scheduler = get_scheduler(scheduler_default, ctx=self.ctx)
        vae_scale = getattr(config, "vae_scale", 8)
        image_seq_len = (h // 16) * (w // 16)
        timesteps = scheduler.set_timesteps(**semantics.set_timesteps_kwargs)
        sigmas = getattr(scheduler, 'sigmas', None)
        sched_ts = getattr(scheduler, "timesteps", None)
        timestep_embed_schedule = timestep_embed_schedule_from_scheduler(scheduler)

        if on_log:
            parts = [
                f"infer model={model_key}",
                f"family={family}",
                f"version={version_key or 'default'}",
                f"size={w}x{h}",
                f"base_seed={base_seed}",
                f"steps={steps}",
                f"guidance={guidance}",
                f"scheduler={scheduler_default}",
                f"supports_guidance={getattr(config, 'supports_guidance', False)}",
                f"cfg_on={bool(neg_embeds is not None)}",
                f"image_seq_len={image_seq_len}",
                f"vae_scale={vae_scale}",
            ]
            if semantics.sigma_schedule is not None:
                parts.append(f"sigma_schedule={semantics.sigma_schedule}")
            parts.append(f"use_empirical_mu={semantics.use_empirical_mu}")
            parts.append(f"requires_sigma_shift={semantics.requires_sigma_shift}")
            if semantics.sched_extra.get("mu") is not None:
                parts.append(f"scheduler_mu={semantics.sched_extra['mu']}")
            if timestep_embed_schedule and len(timestep_embed_schedule) >= 2:
                parts.append(
                    f"t_embed_ends=[{timestep_embed_schedule[0]:.6g},{timestep_embed_schedule[-1]:.6g}]"
                )
            elif timestep_embed_schedule and len(timestep_embed_schedule) == 1:
                parts.append(f"t_embed=[{timestep_embed_schedule[0]:.6g}]")
            if semantics.cfg_renorm:
                parts.append(f"cfg_renorm=True cfg_renorm_min={semantics.cfg_renorm_min}")
            on_log("info", " ".join(parts))

        # Shared latent config (compute once)
        _lnd = runtime_contract.denoise_latent_noise_dtype(self.ctx)
        _noise_sample_dtype = runtime_contract.noise_sample_dtype(self.ctx, _lnd)
        _packed_denoise = getattr(config, "latent_noise_packed", False)
        _flux_pack = _flux_unpack = None
        _lh = _lw = 0
        seq_len = 0
        packed_shape = None
        if _packed_denoise:
            from backend.engine.families.flux1.transformer_mlx import (
                _pack_flux1_latents,
                _unpack_flux1_latents,
            )

            _flux_pack = _pack_flux1_latents
            _flux_unpack = _unpack_flux1_latents
            seq_len = (h // 16) * (w // 16)
            _lh, _lw = h // vae_scale, w // vae_scale
            packed_shape = (1, seq_len, 64)

        n = max(getattr(request, 'n', 1), 1)

        structural_output_meta: dict[str, Any] | None = None
        guide = getattr(request, "structural_guide", None)
        if guide is not None:
            structural_output_meta = {
                "structural_guide_model": (getattr(guide, "model_id", None) or "").strip(),
                "structural_guide_type": getattr(guide, "type", None) or "",
                "structural_guide_weight": float(guide.weight),
                "structural_guide_asset_id": guide.asset_id,
            }

        def _scale_progress(cb, batch_idx, total):
            if cb is None or total <= 1:
                return cb
            def wrapped(p, s, t, msg=None, phase=None):
                overall_p = (batch_idx + float(p)) / total
                prefix = f"[{batch_idx+1}/{total}]"
                msg_out = f"{prefix} {msg}" if msg else prefix
                cb(overall_p, s, t, msg_out, phase)
            return wrapped

        def _generate_one(batch_seed: int, batch_on_progress: Callable | None, batch_idx: int = 0) -> tuple[str, dict] | None:
            # Create deterministic latent using seed
            if _packed_denoise:
                if batch_seed is not None:
                    latents = self.ctx.seeded_randn(packed_shape, batch_seed, dtype=_noise_sample_dtype)
                else:
                    latents = self.ctx.randn(packed_shape, dtype=_noise_sample_dtype)
                if _noise_sample_dtype != _lnd:
                    latents = latents.astype(_lnd)
            elif getattr(config, "encoder_step_kwargs", None) == "qwen_image":
                lh, lw = h // vae_scale, w // vae_scale
                q_seq = lh * lw
                if batch_seed is not None:
                    packed_noise = self.ctx.seeded_randn(
                        (1, q_seq, 64), batch_seed, dtype=_noise_sample_dtype
                    )
                else:
                    packed_noise = self.ctx.randn((1, q_seq, 64), dtype=_noise_sample_dtype)
                if _noise_sample_dtype != _lnd:
                    packed_noise = packed_noise.astype(_lnd)
                packed_noise = self.ctx.reshape(packed_noise, (1, lh, lw, 64))
                latents = self.ctx.permute(packed_noise, (0, 3, 1, 2))
            else:
                latent_shape = (1, config.in_channels, h // vae_scale, w // vae_scale)
                latents = runtime_contract.sample_txt2img_noise(
                    self.ctx,
                    latent_shape=latent_shape,
                    seed=batch_seed,
                    sample_dtype=_noise_sample_dtype,
                    target_dtype=_lnd,
                )

            # ── Hook ③: before denoise (ControlNet signal injection / latent modification) ──
            _local_extra_cond = dict(extra_cond)
            if _packed_denoise:
                latents_nchw = _flux_unpack(self.ctx, latents, _lh, _lw)
                latents_nchw, _local_extra_cond = model.before_denoise(
                    latents_nchw,
                    timesteps,
                    sigmas,
                    txt_embeds=txt_embeds,
                    neg_embeds=neg_embeds,
                    **_local_extra_cond,
                )
                latents = _flux_pack(self.ctx, latents_nchw)
            else:
                latents, _local_extra_cond = model.before_denoise(
                    latents,
                    timesteps,
                    sigmas,
                    txt_embeds=txt_embeds,
                    neg_embeds=neg_embeds,
                    **_local_extra_cond,
                )

            batch_preview_state: dict[str, Any] = {"on_log": on_log}
            if preview_mode == "stream":
                self._warm_step_preview_decoders(
                    entry, version_key or None, batch_preview_state, config=config, on_log=on_log
                )
                try:
                    batch_preview_state["vae_session"] = self._build_vae_preview_session(
                        entry, version_key or None, on_log=on_log
                    )
                except Exception as exc:
                    batch_preview_state["vae_session"] = False
                    if on_log:
                        on_log("warning", f"preview VAE warmup skipped: {exc}")
            pipeline_graph_step("denoise", on_log)
            latents = self._denoise_steps(
                model=model,
                scheduler=scheduler,
                timesteps=timesteps,
                latents=latents,
                config=config,
                runtime_contract=runtime_contract,
                guidance=guidance,
                txt_embeds=txt_embeds,
                neg_embeds=neg_embeds,
                pooled_embeds=pooled_embeds,
                neg_pooled_embeds=neg_pooled_embeds,
                txt_attn_mask=txt_attn_mask,
                neg_attn_mask=neg_attn_mask,
                encoder_type=encoder_type,
                width=w,
                height=h,
                sched_ts=sched_ts,
                sigmas=sigmas,
                timestep_embed_schedule=timestep_embed_schedule,
                extra_cond=_local_extra_cond,
                semantics=semantics,
                ctx_exec=ctx_exec,
                on_progress=batch_on_progress,
                on_log=on_log,
                preview_mode=preview_mode,
                preview_interval=preview_interval,
                preview_max_edge=preview_max_edge,
                preview_state=batch_preview_state,
                entry=entry,
                version_key=version_key or None,
                packed_denoise=_packed_denoise,
                flux_pack=_flux_pack,
                flux_unpack=_flux_unpack,
                latent_h=_lh,
                latent_w=_lw,
            )
            if latents is None:
                return None

            if ctx_exec.cancel_token.is_cancelled():
                return None

            if _packed_denoise:
                latents = _flux_unpack(self.ctx, latents, _lh, _lw)

            _name_infix = f"_b{batch_idx + 1}" if n > 1 else ""
            return self._finalize_image_from_latents(
                latents=latents,
                timesteps=timesteps,
                entry=entry,
                version_key=version_key,
                model_key=model_key,
                seed=batch_seed,
                request=request,
                ctx_exec=ctx_exec,
                steps=steps,
                guidance=guidance,
                w=w,
                h=h,
                on_progress=batch_on_progress,
                on_log=on_log,
                name_infix=_name_infix,
                extra_meta=structural_output_meta,
            )

        results: list[tuple[str, dict]] = []
        try:
            for i in range(n):
                if ctx_exec.cancel_token.is_cancelled():
                    return results if results else None
                batch_seed = base_seed + i
                batch_on_progress = _scale_progress(on_progress, i, n) if n > 1 else on_progress
                if on_log:
                    on_log("info", f"batch {i+1}/{n} seed={batch_seed}")
                result = _generate_one(batch_seed, batch_on_progress, i)
                if result is None:
                    return results if results else None
                results.append(result)
            return results
        finally:
            if structural_cleanup is not None:
                structural_cleanup()

    def _run_flux1_fill_edit(
        self,
        request: ImageEditRequest,
        ctx_exec: ExecutionContext,
        *,
        on_progress: Callable | None = None,
        on_log: Callable | None = None,
    ):
        """FLUX.1 Fill inpainting / outpainting (mflux ``Flux1Fill`` 384-dim patch concat)."""
        from PIL import Image

        from backend.engine.common.controlnet_runtime import require_controlnet_runtime
        from backend.engine.common.structural_guide import is_fill_controlnet

        require_controlnet_runtime(self.ctx, feature="fill_edit")
        from backend.engine.families.flux1.fill_mask import (
            FILL_PATCH_TOKEN_DIM,
            apply_inpaint_mask_rgb,
            build_outpaint_image_and_mask,
            create_fill_static_packed,
            mask_pil_to_weight,
        )
        from backend.engine.families.flux1.transformer_mlx import (
            _pack_flux1_fill_mask_latents,
            _pack_flux1_latents,
            _unpack_flux1_latents,
        )

        model_key, version_key = parse_model_version(request.model)
        if not is_fill_controlnet(model_key):
            raise RuntimeError(
                f"operation {request.operation!r} requires FLUX.1 Fill (flux-fill-controlnet); "
                f"got model {model_key!r}"
            )
        entry = self._registry.require(model_key)
        acts_block = entry.raw.get("actions") if hasattr(entry, "raw") else {}
        if not registry_declares_action(acts_block, request.operation):
            raise RuntimeError(
                f"Model {model_key!r} does not declare action {request.operation!r}; "
                "see config/models_registry.json."
            )

        config_cls = get_config_class(entry.family)
        config = config_cls()
        config.patch_token_dim = FILL_PATCH_TOKEN_DIM
        self._apply_registry_config_overrides(entry, config)
        family = getattr(entry, "family", "flux1")
        assert_image_family_contract(family, config)
        runtime_contract = FamilyRuntimeContract(family=family, config=config)

        if ctx_exec.cancel_token.is_cancelled():
            return None

        bundle_root = _local_bundle_root_fn(self._project_root, entry, version_key or None)
        validate_bundle_graph_step(
            bundle_root, family=family, model_id=model_key, on_log=on_log
        )

        src_path = ctx_exec.asset_store.get_file_path(request.source_asset_id)
        pil = Image.open(str(src_path)).convert("RGB")

        if request.operation == "retouch":
            if not request.mask_asset_id:
                raise RuntimeError("retouch requires mask_asset_id")
            mask_path = ctx_exec.asset_store.get_file_path(request.mask_asset_id)
            mask_pil = Image.open(str(mask_path))
        else:
            if not request.extend:
                raise RuntimeError("extend requires extend.directions and extend.pixels")
            pil, mask_pil = build_outpaint_image_and_mask(
                pil,
                list(request.extend.directions),
                int(request.extend.pixels),
            )

        w0, h0 = pil.size
        w, h = self._align_hw_multiples(w0, h0, align=16)
        pil = self._center_crop_pil(pil, w, h)
        mask_pil = mask_pil.convert("RGB").resize((w, h), Image.Resampling.NEAREST)

        rgb = np.asarray(pil, dtype=np.float32) / 255.0
        mask_hw = mask_pil_to_weight(mask_pil)
        masked_rgb = apply_inpaint_mask_rgb(rgb, mask_hw)
        masked_pil = Image.fromarray(
            (np.clip(masked_rgb, 0.0, 1.0) * 255.0).astype(np.uint8),
            mode="RGB",
        )
        masked_nchw = self._pil_to_nchw_float01(masked_pil, w, h)

        seed = request.seed if request.seed is not None else random.randint(0, 2 ** 32 - 1)
        steps_default = _registry_scalar_default_fn(entry, "steps", 28)
        guidance_default = _registry_scalar_default_fn(entry, "guidance", 30.0)
        steps = int(request.steps) if request.steps is not None else int(steps_default)
        steps = max(1, steps)
        guidance = float(request.guidance) if request.guidance is not None else float(guidance_default)
        guidance = runtime_contract.resolve_guidance_scalar(guidance)
        preview_mode, preview_interval, preview_max_edge = _resolve_image_preview_settings(entry)
        preview_state: dict[str, Any] = {}

        enc_loaded = self._image_encode_load_for_inference(
            request=request,
            bundle_root=bundle_root,
            config=config,
            guidance=guidance,
            runtime_contract=runtime_contract,
            family=family,
            entry=entry,
            version_key=version_key,
            model_key=model_key,
            steps=steps,
            ctx_exec=ctx_exec,
            on_progress=on_progress,
            on_log=on_log,
        )
        if enc_loaded is None:
            return None
        (
            model,
            extra_cond,
            txt_embeds,
            neg_embeds,
            txt_attn_mask,
            neg_attn_mask,
            pooled_embeds,
            neg_pooled_embeds,
            encoder_type,
        ) = enc_loaded

        masked_latents = self._vae_encode_tensor(
            masked_nchw,
            entry,
            version_key or None,
            height_px=h,
            width_px=w,
            on_log=on_log,
        )
        if getattr(self.ctx, "backend", None) == "mlx":
            self.ctx.eval(masked_latents)

        fill_static = create_fill_static_packed(
            self.ctx,
            masked_latents_nchw=masked_latents,
            mask_hw=mask_hw,
            height=h,
            width=w,
            pack_latents_fn=_pack_flux1_latents,
            pack_mask_latents_fn=_pack_flux1_fill_mask_latents,
        )
        if getattr(self.ctx, "backend", None) == "mlx":
            self.ctx.eval(fill_static)

        extra_cond = dict(extra_cond)
        extra_cond["fill_static_packed"] = fill_static

        _meta_ed = request.metadata or {}
        semantics = self._scheduler_semantics_resolver.resolve(
            entry=entry,
            config=config,
            request_scheduler=request.scheduler,
            request_metadata=_meta_ed,
            steps=steps,
            width=w,
            height=h,
        )
        scheduler_default = semantics.scheduler_name
        scheduler = get_scheduler(scheduler_default, ctx=self.ctx)
        timesteps = scheduler.set_timesteps(**semantics.set_timesteps_kwargs)
        sigmas = getattr(scheduler, "sigmas", None)
        sched_ts = getattr(scheduler, "timesteps", None)
        timestep_embed_schedule = timestep_embed_schedule_from_scheduler(scheduler)
        vae_scale = int(getattr(config, "vae_scale", 8))

        if on_log:
            on_log(
                "info",
                f"edit fill model={model_key} operation={request.operation} size={w}x{h} "
                f"seed={seed} steps={steps} guidance={guidance} scheduler={scheduler_default}",
            )

        _lnd = runtime_contract.denoise_latent_noise_dtype(self.ctx)
        _noise_sample_dtype = runtime_contract.noise_sample_dtype(self.ctx, _lnd)
        _lh, _lw = h // vae_scale, w // vae_scale
        seq_len = (h // 16) * (w // 16)
        packed_shape = (1, seq_len, 64)

        latents = self.ctx.seeded_randn(packed_shape, seed, dtype=_noise_sample_dtype)
        if _noise_sample_dtype != _lnd:
            latents = latents.astype(_lnd)

        latents_nchw = _unpack_flux1_latents(self.ctx, latents, _lh, _lw)
        latents_nchw, extra_cond = model.before_denoise(
            latents_nchw,
            timesteps,
            sigmas,
            txt_embeds=txt_embeds,
            neg_embeds=neg_embeds,
            **extra_cond,
        )
        latents = _pack_flux1_latents(self.ctx, latents_nchw)

        preview_state["on_log"] = on_log
        if preview_mode == "stream":
            self._warm_step_preview_decoders(
                entry, version_key or None, preview_state, config=config, on_log=on_log
            )
            try:
                preview_state["vae_session"] = self._build_vae_preview_session(
                    entry, version_key or None, on_log=on_log
                )
            except Exception as exc:
                preview_state["vae_session"] = False
                if on_log:
                    on_log("warning", f"preview VAE warmup skipped: {exc}")

        pipeline_graph_step("denoise", on_log)
        latents = self._denoise_steps(
            model=model,
            scheduler=scheduler,
            timesteps=timesteps,
            latents=latents,
            config=config,
            runtime_contract=runtime_contract,
            guidance=guidance,
            txt_embeds=txt_embeds,
            neg_embeds=neg_embeds,
            pooled_embeds=pooled_embeds,
            neg_pooled_embeds=neg_pooled_embeds,
            txt_attn_mask=txt_attn_mask,
            neg_attn_mask=neg_attn_mask,
            encoder_type=encoder_type,
            width=w,
            height=h,
            sched_ts=sched_ts,
            sigmas=sigmas,
            timestep_embed_schedule=timestep_embed_schedule,
            extra_cond=extra_cond,
            semantics=semantics,
            ctx_exec=ctx_exec,
            on_progress=on_progress,
            on_log=on_log,
            preview_mode=preview_mode,
            preview_interval=preview_interval,
            preview_max_edge=preview_max_edge,
            preview_state=preview_state,
            entry=entry,
            version_key=version_key or None,
            packed_denoise=True,
            flux_unpack=_unpack_flux1_latents,
            latent_h=_lh,
            latent_w=_lw,
        )
        if latents is None:
            return None

        if ctx_exec.cancel_token.is_cancelled():
            return None

        latents = _unpack_flux1_latents(self.ctx, latents, _lh, _lw)

        return self._finalize_image_from_latents(
            latents=latents,
            timesteps=timesteps,
            entry=entry,
            version_key=version_key,
            model_key=model_key,
            seed=seed,
            request=request,
            ctx_exec=ctx_exec,
            steps=steps,
            guidance=guidance,
            w=w,
            h=h,
            on_progress=on_progress,
            on_log=on_log,
            name_infix="_fill",
            extra_meta={
                "operation": request.operation,
                "fill_model": model_key,
            },
        )

    def run_edit(
        self,
        request: ImageEditRequest,
        ctx_exec: ExecutionContext,
        *,
        on_progress: Callable | None = None,
        on_log: Callable | None = None,
    ):
        """图像编辑：``rewrite`` img2img；``retouch``/``extend`` 走 FLUX.1 Fill；kontext instruct 未接线。"""
        if request.operation in ("retouch", "extend"):
            return self._run_flux1_fill_edit(
                request,
                ctx_exec,
                on_progress=on_progress,
                on_log=on_log,
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
        self._apply_registry_config_overrides(entry, config)
        if getattr(config, "edit_use_vl_vision", False):
            return self._run_qwen_image_edit(
                request,
                ctx_exec,
                model_key=model_key,
                version_key=version_key,
                entry=entry,
                config=config,
                on_progress=on_progress,
                on_log=on_log,
            )

        acts = getattr(entry, "actions", frozenset())
        if "edit" not in acts:
            raise RuntimeError(
                f"Model {model_key!r} is not registered for image edit (actions need rewrite/retouch/extend); "
                "refusing ImagePipeline.run_edit — see config/models_registry.json."
            )
        family = getattr(entry, "family", "flux1")

        assert_image_family_contract(family, config)
        runtime_contract = FamilyRuntimeContract(family=family, config=config)

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

        bundle_root = _local_bundle_root_fn(self._project_root, entry, version_key or None)
        validate_bundle_graph_step(
            bundle_root, family=family, model_id=model_key, on_log=on_log
        )
        from PIL import Image

        src_path = ctx_exec.asset_store.get_file_path(request.source_asset_id)
        pil = Image.open(src_path).convert("RGB")
        w0, h0 = pil.size
        w, h = self._align_hw_multiples(w0, h0, align=16)
        pil = self._center_crop_pil(pil, w, h)

        vae_dir_pre = (bundle_root / "vae") if bundle_root else None
        from backend.engine.common.vae import read_vae_dir_config

        vae_cfg_pre, _, _ = read_vae_dir_config(vae_dir_pre)
        vae_cls_pre = str(vae_cfg_pre.get("_class_name") or "")
        uses_encode_bridge = vae_cls_pre in (
            "AutoencoderKLFlux2",
            "AutoencoderKLQwenImage",
            "AutoencoderKLWan",
        )

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
        edit_conditioning_concat = bool(getattr(config, "edit_conditioning_concat", False))

        steps_default = _registry_scalar_default_fn(entry, "steps", 4)
        _meta_ed = request.metadata or {}
        steps = int(request.steps) if request.steps is not None else int(steps_default)
        steps = max(1, steps)
        # mflux ``Config.init_time_step``: img2img starts denoising at this index; latent noise
        # uses ``sigmas[init]`` in ``(1 - sigma) * encoded + sigma * noise`` (not linear f·x+(1-f)·ε).
        init_timestep = 0
        if fidelity > 0.0 and not edit_conditioning_concat:
            init_timestep = max(1, int(steps * fidelity))
        guidance_default = _registry_scalar_default_fn(entry, "guidance", 0.0)
        if request.guidance is not None:
            guidance = float(request.guidance)
        else:
            guidance = float(guidance_default)
        guidance = runtime_contract.resolve_guidance_scalar(guidance)
        preview_mode, preview_interval, preview_max_edge = _resolve_image_preview_settings(entry)
        preview_state: dict[str, Any] = {}

        enc_loaded = self._image_encode_load_for_inference(
            request=request,
            bundle_root=bundle_root,
            config=config,
            guidance=guidance,
            runtime_contract=runtime_contract,
            family=family,
            entry=entry,
            version_key=version_key,
            model_key=model_key,
            steps=steps,
            ctx_exec=ctx_exec,
            on_progress=on_progress,
            on_log=on_log,
        )
        if enc_loaded is None:
            return None
        (
            model,
            extra_cond,
            txt_embeds,
            neg_embeds,
            txt_attn_mask,
            neg_attn_mask,
            pooled_embeds,
            neg_pooled_embeds,
            encoder_type,
        ) = enc_loaded
        if edit_conditioning_concat:
            from backend.engine.families.fibo import vae_mlx as fibo_vae_mlx

            extra_cond = dict(extra_cond)
            extra_cond["conditioning_latents"] = encoded
            extra_cond["conditioning_image_ids"] = fibo_vae_mlx.create_conditioning_image_ids(h, w)

        semantics = self._scheduler_semantics_resolver.resolve(
            entry=entry,
            config=config,
            request_scheduler=request.scheduler,
            request_metadata=_meta_ed,
            steps=steps,
            width=w,
            height=h,
            init_timestep=init_timestep,
        )
        scheduler_default = semantics.scheduler_name
        scheduler = get_scheduler(scheduler_default, ctx=self.ctx)
        timesteps = scheduler.set_timesteps(**semantics.set_timesteps_kwargs)
        sigmas = getattr(scheduler, "sigmas", None)
        sched_ts = getattr(scheduler, "timesteps", None)
        timestep_embed_schedule = timestep_embed_schedule_from_scheduler(scheduler)

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

        _lnd_edit = runtime_contract.denoise_latent_noise_dtype(self.ctx)
        _noise_sample_dtype_edit = runtime_contract.noise_sample_dtype(self.ctx, _lnd_edit)
        if getattr(config, "encoder_step_kwargs", None) == "qwen_image":
            q_h = int(encoded.shape[2])
            q_w = int(encoded.shape[3])
            q_seq = q_h * q_w
            packed_noise = self.ctx.seeded_randn((1, q_seq, 64), seed, dtype=_noise_sample_dtype_edit)
            if _noise_sample_dtype_edit != _lnd_edit:
                packed_noise = packed_noise.astype(_lnd_edit)
            packed_noise = self.ctx.reshape(packed_noise, (1, q_h, q_w, 64))
            noise = self.ctx.permute(packed_noise, (0, 3, 1, 2))
        elif getattr(config, "latent_noise_packed", False):
            from backend.engine.families.flux1.transformer_mlx import _unpack_flux1_latents

            _, _, lh, lw = encoded.shape
            seq_len = (lh // 2) * (lw // 2)
            packed = self.ctx.seeded_randn((1, seq_len, 64), seed, dtype=_noise_sample_dtype_edit)
            if _noise_sample_dtype_edit != _lnd_edit:
                packed = packed.astype(_lnd_edit)
            noise = _unpack_flux1_latents(self.ctx, packed, lh, lw)
        else:
            noise = runtime_contract.sample_edit_noise(
                self.ctx,
                encoded_shape=tuple(encoded.shape),
                seed=seed,
                sample_dtype=_noise_sample_dtype_edit,
                target_dtype=_lnd_edit,
            )
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
                f"source_fidelity={fidelity}"
                + (" edit_conditioning_concat=1" if edit_conditioning_concat else ""),
            )

        latents, extra_cond = model.before_denoise(
            latents,
            timesteps,
            sigmas,
            txt_embeds=txt_embeds,
            neg_embeds=neg_embeds,
            **extra_cond,
        )

        _packed_edit = getattr(config, "latent_noise_packed", False)
        _flux_unpack_edit = None
        _lh_edit = _lw_edit = 0
        if _packed_edit:
            from backend.engine.families.flux1.transformer_mlx import _unpack_flux1_latents

            _flux_unpack_edit = _unpack_flux1_latents
            if latents.ndim == 3:
                _, seq_len, _ = latents.shape
                _lh_edit = int(seq_len ** 0.5)
                _lw_edit = seq_len // max(_lh_edit, 1)
            elif latents.ndim == 4:
                _, _, _lh_edit, _lw_edit = latents.shape

        preview_state["on_log"] = on_log
        if preview_mode == "stream":
            self._warm_step_preview_decoders(
                entry, version_key or None, preview_state, config=config, on_log=on_log
            )
            try:
                preview_state["vae_session"] = self._build_vae_preview_session(
                    entry, version_key or None, on_log=on_log
                )
            except Exception as exc:
                preview_state["vae_session"] = False
                if on_log:
                    on_log("warning", f"preview VAE warmup skipped: {exc}")
        pipeline_graph_step("denoise", on_log)
        latents = self._denoise_steps(
            model=model,
            scheduler=scheduler,
            timesteps=timesteps,
            latents=latents,
            config=config,
            runtime_contract=runtime_contract,
            guidance=guidance,
            txt_embeds=txt_embeds,
            neg_embeds=neg_embeds,
            pooled_embeds=pooled_embeds,
            neg_pooled_embeds=neg_pooled_embeds,
            txt_attn_mask=txt_attn_mask,
            neg_attn_mask=neg_attn_mask,
            encoder_type=encoder_type,
            width=w,
            height=h,
            sched_ts=sched_ts,
            sigmas=sigmas,
            timestep_embed_schedule=timestep_embed_schedule,
            extra_cond=extra_cond,
            semantics=semantics,
            ctx_exec=ctx_exec,
            on_progress=on_progress,
            on_log=on_log,
            preview_mode=preview_mode,
            preview_interval=preview_interval,
            preview_max_edge=preview_max_edge,
            preview_state=preview_state,
            entry=entry,
            version_key=version_key or None,
            timestep_offset=init_timestep,
            packed_denoise=_packed_edit and latents.ndim == 3,
            flux_unpack=_flux_unpack_edit,
            latent_h=_lh_edit,
            latent_w=_lw_edit,
        )
        if latents is None:
            return None

        if ctx_exec.cancel_token.is_cancelled():
            return None

        if _packed_edit and latents.ndim == 3 and _flux_unpack_edit is not None:
            latents = _flux_unpack_edit(self.ctx, latents, _lh_edit, _lw_edit)

        def _edit_post_decode(image: Any) -> Any:
            if getattr(config, "edit_rmbg_composite_output", False):
                matte = image.convert("L")
                composite = pil.copy()
                composite.putalpha(matte.resize(composite.size, Image.LANCZOS))
                return composite
            return image

        return self._finalize_image_from_latents(
            latents=latents,
            timesteps=timesteps,
            entry=entry,
            version_key=version_key,
            model_key=model_key,
            seed=seed,
            request=request,
            ctx_exec=ctx_exec,
            steps=steps,
            guidance=guidance,
            w=w,
            h=h,
            on_progress=on_progress,
            on_log=on_log,
            name_infix="_edit",
            post_decode=_edit_post_decode,
            extra_meta={
                "operation": request.operation,
                "source_fidelity": fidelity,
            },
        )

    def _run_qwen_image_edit(
        self,
        request: ImageEditRequest,
        ctx_exec: ExecutionContext,
        *,
        model_key: str,
        version_key: str,
        entry: Any,
        config: Any,
        on_progress: Callable | None = None,
        on_log: Callable | None = None,
    ):
        """Qwen-Image-Edit：VL 图文编码 + VAE 参考 latent 拼接（对齐 mflux ``QwenImageEdit``）。"""
        from PIL import Image

        from backend.engine.families.qwen.edit_util import (
            compute_qwen_edit_dimensions,
            create_qwen_edit_conditioning_latents,
        )

        family = getattr(entry, "family", "qwen_image")
        assert_image_family_contract(family, config)
        runtime_contract = FamilyRuntimeContract(family=family, config=config)
        vae_scale = int(getattr(config, "vae_scale", 16))

        if ctx_exec.cancel_token.is_cancelled():
            return None

        bundle_root = _local_bundle_root_fn(self._project_root, entry, version_key or None)
        validate_bundle_graph_step(
            bundle_root, family=family, model_id=model_key, on_log=on_log
        )

        src_path = ctx_exec.asset_store.get_file_path(request.source_asset_id)
        pil = Image.open(src_path).convert("RGB")
        w, h, vl_w, vl_h, vae_w, vae_h = compute_qwen_edit_dimensions(pil)

        steps_default = _registry_scalar_default_fn(entry, "steps", 20)
        guidance_default = _registry_scalar_default_fn(entry, "guidance", 4.0)
        _meta_ed = request.metadata or {}
        steps = int(request.steps) if request.steps is not None else int(steps_default)
        steps = max(1, steps)
        if request.guidance is not None:
            guidance = float(request.guidance)
        else:
            guidance = float(guidance_default)
        guidance = runtime_contract.resolve_guidance_scalar(guidance)
        seed = request.seed if request.seed is not None else random.randint(0, 2 ** 32 - 1)
        preview_mode, preview_interval, preview_max_edge = _resolve_image_preview_settings(entry)
        preview_state: dict[str, Any] = {}

        emit_phase(on_progress, phase="encoding", progress=0.02, n_steps=steps)
        pipeline_graph_step("encode_prompt", on_log)
        neg_prompt = (request.negative_prompt or "").strip()
        if getattr(self.ctx, "backend", None) == "cuda":
            from backend.engine.families.qwen.text_encoder_cuda import encode_qwen_edit_prompts_cuda

            device = getattr(self.ctx, "_device", "cuda")
            txt_embeds, txt_attn_mask, neg_embeds, neg_attn_mask = encode_qwen_edit_prompts_cuda(
                bundle_root=bundle_root,
                device=device,
                prompt=request.prompt,
                negative_prompt=neg_prompt,
                source=pil,
            )
            pooled_embeds = neg_pooled_embeds = None
        else:
            from backend.engine.common.text_encoders.qwen_edit_mlx import (
                build_qwen_edit_vl_tokenizer,
                encode_qwen_edit_prompts_mlx,
                load_qwen_edit_vl_encoder,
            )

            tok_root = bundle_root / "tokenizer"
            if not tok_root.is_dir():
                tok_root = bundle_root / "text_encoder"
            vl_encoder = load_qwen_edit_vl_encoder(bundle_root, self.ctx)
            vl_tokenizer = build_qwen_edit_vl_tokenizer(tok_root)
            txt_embeds, txt_attn_mask, neg_embeds, neg_attn_mask = encode_qwen_edit_prompts_mlx(
                vl_encoder=vl_encoder,
                vl_tokenizer=vl_tokenizer,
                ctx=self.ctx,
                prompt=request.prompt,
                negative_prompt=neg_prompt,
                source=pil,
                vl_width=vl_w,
                vl_height=vl_h,
            )
            pooled_embeds = neg_pooled_embeds = None
        encoder_type = getattr(config, "encoder_type", "qwen_image")

        if ctx_exec.cancel_token.is_cancelled():
            return None

        emit_phase(on_progress, phase="loading_model", progress=0.08, n_steps=steps)
        pipeline_graph_step("load_transformer", on_log)
        allow_cache = not (getattr(request, "adapters", None) or [])
        model = self._load_model(
            family, config, entry, version_key or None, allow_cache=allow_cache
        )
        if model is None:
            raise RuntimeError(f"Failed to load model: {model_key}")
        model.after_load_weights(bundle_root=str(bundle_root) if bundle_root else None)
        self._apply_image_lora_adapters(family, model, request, on_log)
        extra_cond = model.prepare_conditioning(
            request, bundle_root=str(bundle_root) if bundle_root else None
        )

        def _vae_enc(img_n11: Any, *, height_px: int, width_px: int) -> Any:
            return self._vae_encode_tensor(
                img_n11,
                entry,
                version_key or None,
                height_px=height_px,
                width_px=width_px,
                on_log=on_log,
            )

        cond_latents, cond_grid = create_qwen_edit_conditioning_latents(
            self.ctx,
            vae_encode_fn=lambda img, height_px, width_px: _vae_enc(
                img, height_px=height_px, width_px=width_px
            ),
            source=pil,
            vae_width=vae_w,
            vae_height=vae_h,
            on_log=on_log,
        )
        extra_cond = dict(extra_cond)
        extra_cond["edit_conditioning_latents"] = cond_latents
        extra_cond["edit_cond_image_grid"] = cond_grid

        semantics = self._scheduler_semantics_resolver.resolve(
            entry=entry,
            config=config,
            request_scheduler=request.scheduler,
            request_metadata=_meta_ed,
            steps=steps,
            width=w,
            height=h,
            init_timestep=0,
        )
        scheduler_default = semantics.scheduler_name
        scheduler = get_scheduler(scheduler_default, ctx=self.ctx)
        timesteps = scheduler.set_timesteps(**semantics.set_timesteps_kwargs)
        sigmas = getattr(scheduler, "sigmas", None)
        sched_ts = getattr(scheduler, "timesteps", None)
        timestep_embed_schedule = timestep_embed_schedule_from_scheduler(scheduler)

        _lnd = runtime_contract.denoise_latent_noise_dtype(self.ctx)
        _noise_sample_dtype = runtime_contract.noise_sample_dtype(self.ctx, _lnd)
        lh, lw = h // vae_scale, w // vae_scale
        q_seq = lh * lw
        packed_noise = self.ctx.seeded_randn((1, q_seq, 64), seed, dtype=_noise_sample_dtype)
        if _noise_sample_dtype != _lnd:
            packed_noise = packed_noise.astype(_lnd)
        latents = self.ctx.reshape(packed_noise, (1, lh, lw, 64))
        latents = self.ctx.permute(latents, (0, 3, 1, 2))
        if getattr(self.ctx, "backend", None) == "mlx":
            self.ctx.eval(latents)

        if on_log:
            on_log(
                "info",
                f"qwen_image_edit model={model_key} out={w}x{h} vae_cond={vae_w}x{vae_h} "
                f"vl={vl_w}x{vl_h} steps={steps} guidance={guidance} seed={seed}",
            )

        latents, extra_cond = model.before_denoise(
            latents,
            timesteps,
            sigmas,
            txt_embeds=txt_embeds,
            neg_embeds=neg_embeds,
            **extra_cond,
        )

        preview_state["on_log"] = on_log
        if preview_mode == "stream":
            self._warm_step_preview_decoders(
                entry, version_key or None, preview_state, config=config, on_log=on_log
            )
        pipeline_graph_step("denoise", on_log)
        latents = self._denoise_steps(
            model=model,
            scheduler=scheduler,
            timesteps=timesteps,
            latents=latents,
            config=config,
            runtime_contract=runtime_contract,
            guidance=guidance,
            txt_embeds=txt_embeds,
            neg_embeds=neg_embeds,
            pooled_embeds=pooled_embeds,
            neg_pooled_embeds=neg_pooled_embeds,
            txt_attn_mask=txt_attn_mask,
            neg_attn_mask=neg_attn_mask,
            encoder_type=encoder_type,
            width=w,
            height=h,
            sched_ts=sched_ts,
            sigmas=sigmas,
            timestep_embed_schedule=timestep_embed_schedule,
            extra_cond=extra_cond,
            semantics=semantics,
            ctx_exec=ctx_exec,
            on_progress=on_progress,
            on_log=on_log,
            preview_mode=preview_mode,
            preview_interval=preview_interval,
            preview_max_edge=preview_max_edge,
            preview_state=preview_state,
            entry=entry,
            version_key=version_key or None,
        )
        if latents is None:
            return None

        if ctx_exec.cancel_token.is_cancelled():
            return None

        return self._finalize_image_from_latents(
            latents=latents,
            timesteps=timesteps,
            entry=entry,
            version_key=version_key,
            model_key=model_key,
            seed=seed,
            request=request,
            ctx_exec=ctx_exec,
            steps=steps,
            guidance=guidance,
            w=w,
            h=h,
            on_progress=on_progress,
            on_log=on_log,
            name_infix="_edit",
            extra_meta={"operation": request.operation, "edit_model": "qwen-image-edit"},
        )

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _augment_request_for_structural_guide(
        self,
        request: ImageGenerationRequest,
    ) -> ImageGenerationRequest:
        guide = getattr(request, "structural_guide", None)
        if guide is None:
            return request
        from backend.engine.common.controlnet_runtime import require_controlnet_runtime

        require_controlnet_runtime(self.ctx, feature="structural_guide")
        from backend.engine.common.structural_guide import companion_lora_id

        model_id = (getattr(guide, "model_id", None) or "").strip()
        if not model_id:
            raise RuntimeError(
                "structural_guide.model_id is required (registry controlnet id, e.g. flux-canny-controlnet)"
            )
        lora_id = companion_lora_id(model_id)
        if not lora_id:
            return request
        adapters = list(request.adapters or [])
        if any(a.id == lora_id or a.id.startswith(f"{lora_id}:") for a in adapters):
            return request
        adapters.append(AdapterRef(id=lora_id, weight=float(guide.weight)))
        return request.model_copy(update={"adapters": adapters})

    def _attach_structural_conditioning(
        self,
        *,
        request: ImageGenerationRequest,
        family: str,
        model: Any,
        entry: Any,
        version_key: str | None,
        extra_cond: dict[str, Any],
        width: int,
        height: int,
        ctx_exec: ExecutionContext,
        on_log: Callable[..., None] | None,
    ) -> tuple[dict[str, Any], Callable[[], None] | None]:
        guide = getattr(request, "structural_guide", None)
        if guide is None:
            return extra_cond, None
        from backend.engine.common.controlnet_runtime import require_controlnet_runtime

        require_controlnet_runtime(self.ctx, feature="structural_guide")
        if family != "flux1":
            raise RuntimeError(
                f"structural_guide is only supported on flux1 base models (got family={family!r})"
            )
        from backend.engine.common.structural_guide import (
            infer_guide_type,
            is_fill_controlnet,
            is_redux_controlnet,
            load_flux1_structural_patch_embed,
            preprocess_structural_rgb,
        )

        controlnet_id = (getattr(guide, "model_id", None) or "").strip()
        if not controlnet_id:
            raise RuntimeError("structural_guide.model_id is required")
        if is_fill_controlnet(controlnet_id):
            raise RuntimeError(
                "flux-fill is an inpainting model — use image retouch/extend with a mask, "
                "not text-to-image structural_guide"
            )
        guide_type = getattr(guide, "type", None) or infer_guide_type(controlnet_id)

        from PIL import Image

        src_path = ctx_exec.asset_store.get_file_path(guide.asset_id)
        pil = Image.open(str(src_path))

        if guide_type == "redux" or is_redux_controlnet(controlnet_id):
            from backend.engine.families.flux1.redux_encode import (
                encode_redux_context_tokens,
                resolve_redux_bundle_root,
            )

            redux_root = resolve_redux_bundle_root(
                self._registry, self._project_root, controlnet_id
            )
            tokens = encode_redux_context_tokens(
                pil, redux_bundle_root=redux_root, on_log=on_log
            )
            w = float(guide.weight)
            if w <= 0.0:
                if on_log:
                    on_log("info", "structural_guide redux skipped (weight <= 0)")
                return dict(extra_cond), None
            if w != 1.0:
                tokens = tokens * np.float32(w)
            redux_embeds = self.ctx.array(tokens.astype(np.float32))
            if getattr(self.ctx, "backend", None) == "mlx":
                self.ctx.eval(redux_embeds)
            out = dict(extra_cond)
            out["redux_txt_embeds"] = redux_embeds
            if on_log:
                on_log(
                    "info",
                    f"structural_guide type=redux controlnet={controlnet_id} "
                    f"weight={float(guide.weight):.3f} asset={guide.asset_id}",
                )
            return out, None

        rgb = preprocess_structural_rgb(
            pil,
            guide_type=guide_type,
            width=width,
            height=height,
            registry=self._registry,
            project_root=self._project_root,
            on_log=on_log,
        )
        arr = rgb[None, ...]
        image_nchw = self.ctx.array(arr)
        image_nchw = self.ctx.permute(image_nchw, (0, 3, 1, 2))

        structural_latents = self._vae_encode_tensor(
            image_nchw,
            entry,
            version_key,
            height_px=height,
            width_px=width,
            on_log=on_log,
        )
        if getattr(self.ctx, "backend", None) == "mlx":
            self.ctx.eval(structural_latents)

        activate = getattr(model, "activate_structural_patch_embed", None)
        deactivate = getattr(model, "deactivate_structural_patch_embed", None)
        if not callable(activate) or not callable(deactivate):
            raise RuntimeError(
                f"structural_guide requires Flux1Transformer structural patch embed; "
                f"model={type(model).__name__}"
            )
        pw, pb = load_flux1_structural_patch_embed(
            registry=self._registry,
            project_root=self._project_root,
            controlnet_model_id=controlnet_id,
            ctx=self.ctx,
            on_log=on_log,
        )
        activate(pw, pb)

        out = dict(extra_cond)
        out["structural_latents_nchw"] = structural_latents
        if on_log:
            on_log(
                "info",
                f"structural_guide type={guide_type} controlnet={controlnet_id} "
                f"weight={float(guide.weight):.3f} asset={guide.asset_id}",
            )
        return out, deactivate

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
        base_model_id, _ = parse_model_version(request.model)
        entry = self._registry.get(base_model_id)
        if entry is not None:
            lora_support = _registry_scalar_default_fn(entry, "lora_support", False)
            if not lora_support:
                raise RuntimeError(
                    f"Model {base_model_id!r} does not declare LoRA support; "
                    "remove adapters from the request or use a LoRA-capable base model."
                )
        from backend.engine.runtime.mlx import MLXContext

        if not isinstance(self.ctx, MLXContext):
            raise RuntimeError(
                "LoRA merging for Flux.1 / Flux2 / Z-Image / Qwen Image is only implemented on the MLX runtime; "
                f"current runtime is {type(self.ctx).__name__}."
            )
        _merge_image_lora_adapters(
            family=family,
            model=model,
            adapters=list(adapters),
            base_model_id=base_model_id,
            project_root=self._project_root,
            registry=self._registry,
            ctx=self.ctx,
            on_log=on_log,
        )

    def _model_cache_key(self, entry, version_key: str | None) -> str:
        return f"image:{entry.id}:{version_key or 'default'}"

    def _load_model(
        self,
        family: str,
        config,
        entry,
        version_key: str | None,
        *,
        allow_cache: bool = True,
    ):
        cache_key = self._model_cache_key(entry, version_key)
        if allow_cache and self._cache is not None:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached

        bundle_root = _local_bundle_root_fn(self._project_root, entry, version_key)
        apply_image_bundle_config_merger(config, bundle_root)

        trans_cls = _get_transformer_class(family)
        model = trans_cls(config, self.ctx)
        remap_fn = _get_weight_remap(family)

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
        if allow_cache and self._cache is not None:
            from backend.engine.common.weights import parse_size_gb

            ver = _resolve_version_block_fn(entry, version_key)
            size_str = ""
            if ver:
                size_str = str(ver.get("size") or "")
            if not size_str:
                raw = getattr(entry, "raw", {}) or {}
                size_str = str(raw.get("size") or "10GB")
            self._cache.put(cache_key, model, parse_size_gb(size_str))
        return model

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
        from backend.engine.common.vae import (
            apply_flux2_latent_preprocess_if_enabled,
            create_loaded_vae_decoder,
            load_vae_weight_dict,
            read_vae_dir_config,
            reshape_packed_latents_to_nchw,
            vae_forward_to_pil,
            vae_output_to_uint8_hwc,
        )
        from PIL import Image

        bundle_root = _local_bundle_root_fn(self._project_root, entry, version_key)
        vae_dir = (bundle_root / "vae") if bundle_root else None
        vae_cfg, scaling_factor, shift_factor = read_vae_dir_config(vae_dir)
        latents = reshape_packed_latents_to_nchw(latents)

        vae_cls = str(vae_cfg.get("_class_name") or "")
        entry_family = str(getattr(entry, "family", "") or "") if entry is not None else ""
        decode_handler = get_vae_decode_handler(vae_cls, entry_family=entry_family)
        if decode_handler is not None:
            return decode_handler(
                ctx=self.ctx,
                latents=latents,
                bundle_root=bundle_root,
                project_root=self._project_root,
                on_log=on_log,
                vae_output_to_uint8_hwc=vae_output_to_uint8_hwc,
                image_cls=Image,
            )

        vae_weights = load_vae_weight_dict(self.ctx, vae_dir, fail_if_config_only=True)
        latents, scaling_factor, shift_factor = apply_flux2_latent_preprocess_if_enabled(
            ctx, latents, vae_cfg, vae_weights, scaling_factor, shift_factor
        )
        vae, decoder_w, loaded, skipped = create_loaded_vae_decoder(
            ctx, latents, vae_weights, scaling_factor, shift_factor
        )
        if not decoder_w:
            raise RuntimeError(
                f"VAE weights under {vae_dir} produced no decoder tensors after remap; check bundle."
            )
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
        return vae_forward_to_pil(ctx, vae, latents)
