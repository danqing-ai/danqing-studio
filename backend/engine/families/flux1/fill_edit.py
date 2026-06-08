"""FLUX.1 Fill inpainting / outpainting — mask helpers + pipeline dispatch."""
from __future__ import annotations

import random
from contextlib import AbstractContextManager, nullcontext
from typing import Any, Callable

import numpy as np
from PIL import Image

from backend.core.contracts import ExecutionContext
from backend.core.registry_format import registry_declares_action
from backend.engine.contracts.pipeline_registry import (
    registry_scalar_default as _registry_scalar_default_fn,
)
from backend.engine.contracts.runtime_contracts import FamilyRuntimeContract
from backend.engine.common.ops.schedulers import get_scheduler
from backend.engine.config.model_configs import assert_image_family_contract, get_config_class
from backend.engine.inference.image_denoise import run_image_denoise
from backend.engine.sessions._context import ResolvedRun, require_resolved_bundle
from backend.engine.pipelines.image_run_common import (
    _IMAGE_SCHEDULER_SEMANTICS,
    align_image_hw_multiples,
    apply_image_registry_config_overrides,
    build_image_vae_preview_session,
    center_crop_image_pil,
    finalize_image_from_latents,
    image_encode_load_for_inference,
    image_vae_encode_tensor,
    pil_image_to_nchw_float01,
    warm_image_step_preview_decoders,
)
from backend.engine.pipelines.pipeline_progress import (
    pipeline_graph_step,
    timestep_embed_schedule_from_scheduler,
    validate_bundle_graph_step,
)

PhaseCmFactory = Callable[[str], AbstractContextManager[Any]]

# ``ModelConfig.x_embedder_input_dim`` for Fill: 64 noise + 64 masked VAE + 256 mask pack.
FILL_PATCH_TOKEN_DIM = 384
FILL_STATIC_TOKEN_DIM = 320


def mask_pil_to_weight(mask: Image.Image) -> np.ndarray:
    """White (high) = inpaint region → weight 1.0; returns ``[H, W]`` float32."""
    if mask.mode != "RGB":
        mask = mask.convert("RGB")
    arr = np.asarray(mask, dtype=np.float32) / 255.0
    return (arr[..., 0] > 0.5).astype(np.float32)


def apply_inpaint_mask_rgb(rgb: np.ndarray, mask_hw: np.ndarray) -> np.ndarray:
    """Zero pixels to repaint before VAE encode (``image * (1 - mask)``)."""
    m = mask_hw[..., None] if mask_hw.ndim == 2 else mask_hw[:, :, :1]
    return rgb * (1.0 - m)


def reshape_mask_latent_channels(mask_hw: np.ndarray, height: int, width: int) -> np.ndarray:
    """``[H,W]`` mask → ``[1, 64, H//8, W//8]`` (``MaskUtil.reshape_mask``)."""
    if mask_hw.ndim != 2:
        raise RuntimeError(f"reshape_mask_latent_channels expects 2D mask, got {mask_hw.shape}")
    h, w = int(height), int(width)
    if mask_hw.shape != (h, w):
        raise RuntimeError(
            f"mask shape {mask_hw.shape} does not match image {h}x{w}"
        )
    m = mask_hw.astype(np.float32)
    m = np.reshape(m, (1, h // 8, 8, w // 8, 8))
    m = np.transpose(m, (0, 2, 4, 1, 3))
    return np.reshape(m, (1, 64, h // 8, w // 8)).astype(np.float32)


def build_outpaint_image_and_mask(
    pil: Image.Image,
    directions: list[str],
    pixels: int,
) -> tuple[Image.Image, Image.Image]:
    """Expand canvas; white mask on new border (regenerate), black on original."""
    w, h = pil.size
    pad = {"top": 0, "bottom": 0, "left": 0, "right": 0}
    px = max(64, min(2048, int(pixels)))
    for d in directions:
        if d in pad:
            pad[d] = px
    new_w = w + pad["left"] + pad["right"]
    new_h = h + pad["top"] + pad["bottom"]
    canvas = Image.new("RGB", (new_w, new_h), (0, 0, 0))
    mask = Image.new("RGB", (new_w, new_h), (255, 255, 255))
    canvas.paste(pil, (pad["left"], pad["top"]))
    preserve = Image.new("RGB", (w, h), (0, 0, 0))
    mask.paste(preserve, (pad["left"], pad["top"]))
    return canvas, mask


def create_fill_static_packed(
    ctx: Any,
    *,
    masked_latents_nchw: Any,
    mask_hw: np.ndarray,
    height: int,
    width: int,
    pack_latents_fn: Any,
    pack_mask_latents_fn: Any,
) -> Any:
    """Return ``[1, seq, 320]`` masked VAE pack (64) + mask pack (256)."""
    masked_packed = pack_latents_fn(ctx, masked_latents_nchw)
    mask_spatial = reshape_mask_latent_channels(mask_hw, height, width)
    mask_tensor = ctx.array(mask_spatial)
    mask_packed = pack_mask_latents_fn(ctx, mask_tensor)
    static = ctx.concat([masked_packed, mask_packed], axis=-1)
    expected = int(static.shape[-1])
    if expected != FILL_STATIC_TOKEN_DIM:
        raise RuntimeError(
            f"FLUX Fill static context last dim {expected} (expected {FILL_STATIC_TOKEN_DIM})"
        )
    return static


def build_flux1_fill_edit_context(
    pipeline: Any,
    request: Any,
    ctx_exec: ExecutionContext,
    *,
    resolved: ResolvedRun,
    on_progress: Callable | None = None,
    on_log: Callable | None = None,
    phase_cm: PhaseCmFactory | None = None,
):
    """Build ``ImageFillEditRunContext`` through encode + schedule."""
    from backend.engine.pipelines.image_fill_edit_phases import ImageFillEditRunContext

    phase_cm = phase_cm or (lambda _name: nullcontext())
    from backend.engine.families.flux1.structural import is_fill_controlnet, require_controlnet_runtime

    require_controlnet_runtime(pipeline.ctx, feature="fill_edit")
    from backend.engine.families.flux1.transformer_mlx import (
        _pack_flux1_fill_mask_latents,
        _pack_flux1_latents,
        _unpack_flux1_latents,
    )
    from backend.engine.pipelines.image_pipeline import _resolve_image_preview_settings

    model_key = resolved.model_id
    version_key = resolved.version_key
    entry = resolved.registry_entry
    if not is_fill_controlnet(model_key):
        raise RuntimeError(
            f"operation {request.operation!r} requires FLUX.1 Fill (flux-fill-controlnet); "
            f"got model {model_key!r}"
        )
    acts_block = entry.raw.get("actions") if hasattr(entry, "raw") else {}
    if not registry_declares_action(acts_block, request.operation):
        raise RuntimeError(
            f"Model {model_key!r} does not declare action {request.operation!r}; "
            "see config/models_registry.json."
        )

    config_cls = get_config_class(entry.family)
    config = config_cls()
    config.patch_token_dim = FILL_PATCH_TOKEN_DIM
    apply_image_registry_config_overrides(pipeline, entry, config)
    family = getattr(entry, "family", "flux1")
    assert_image_family_contract(family, config)
    runtime_contract = FamilyRuntimeContract(family=family, config=config)

    if ctx_exec.cancel_token.is_cancelled():
        return None

    bundle_root = require_resolved_bundle(resolved)
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
    w, h = align_image_hw_multiples(w0, h0, align=16)
    pil = center_crop_image_pil(pil, w, h)
    mask_pil = mask_pil.convert("RGB").resize((w, h), Image.Resampling.NEAREST)

    rgb = np.asarray(pil, dtype=np.float32) / 255.0
    mask_hw = mask_pil_to_weight(mask_pil)
    masked_rgb = apply_inpaint_mask_rgb(rgb, mask_hw)
    masked_pil = Image.fromarray(
        (np.clip(masked_rgb, 0.0, 1.0) * 255.0).astype(np.uint8),
        mode="RGB",
    )
    masked_nchw = pil_image_to_nchw_float01(pipeline, masked_pil, w, h)

    seed = request.seed if request.seed is not None else random.randint(0, 2 ** 32 - 1)
    steps_default = _registry_scalar_default_fn(entry, "steps", 28)
    guidance_default = _registry_scalar_default_fn(entry, "guidance", 30.0)
    steps = int(request.steps) if request.steps is not None else int(steps_default)
    steps = max(1, steps)
    guidance = float(request.guidance) if request.guidance is not None else float(guidance_default)
    guidance = runtime_contract.resolve_guidance_scalar(guidance)
    preview_mode, preview_interval, preview_max_edge = _resolve_image_preview_settings(entry)
    preview_state: dict[str, Any] = {}

    with phase_cm("encode"):
        enc_loaded = image_encode_load_for_inference(
            pipeline,
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

        masked_latents = image_vae_encode_tensor(
            pipeline,
            masked_nchw,
            entry,
            version_key or None,
            height_px=h,
            width_px=w,
            on_log=on_log,
        )
        if getattr(pipeline.ctx, "backend", None) == "mlx":
            pipeline.ctx.eval(masked_latents)

        fill_static = create_fill_static_packed(
            pipeline.ctx,
            masked_latents_nchw=masked_latents,
            mask_hw=mask_hw,
            height=h,
            width=w,
            pack_latents_fn=_pack_flux1_latents,
            pack_mask_latents_fn=_pack_flux1_fill_mask_latents,
        )
        if getattr(pipeline.ctx, "backend", None) == "mlx":
            pipeline.ctx.eval(fill_static)

        extra_cond = dict(extra_cond)
        extra_cond["fill_static_packed"] = fill_static

    with phase_cm("schedule"):
        _meta_ed = request.metadata or {}
        semantics = _IMAGE_SCHEDULER_SEMANTICS.resolve(
            entry=entry,
            config=config,
            request_scheduler=request.scheduler,
            request_metadata=_meta_ed,
            steps=steps,
            width=w,
            height=h,
        )
        scheduler_default = semantics.scheduler_name
        scheduler = get_scheduler(scheduler_default, ctx=pipeline.ctx)
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

        _lnd = runtime_contract.denoise_latent_noise_dtype(pipeline.ctx)
        _noise_sample_dtype = runtime_contract.noise_sample_dtype(pipeline.ctx, _lnd)
        _lh, _lw = h // vae_scale, w // vae_scale
        seq_len = (h // 16) * (w // 16)
        packed_shape = (1, seq_len, 64)

        latents = pipeline.ctx.seeded_randn(packed_shape, seed, dtype=_noise_sample_dtype)
        if _noise_sample_dtype != _lnd:
            latents = latents.astype(_lnd)

        latents_nchw = _unpack_flux1_latents(pipeline.ctx, latents, _lh, _lw)
        latents_nchw, extra_cond = model.before_denoise(
            latents_nchw,
            timesteps,
            sigmas,
            txt_embeds=txt_embeds,
            neg_embeds=neg_embeds,
            **extra_cond,
        )
        latents = _pack_flux1_latents(pipeline.ctx, latents_nchw)

    return ImageFillEditRunContext(
        pipeline=pipeline,
        request=request,
        exec_ctx=ctx_exec,
        entry=entry,
        config=config,
        runtime_contract=runtime_contract,
        family=family,
        model_key=model_key,
        version_key=version_key,
        bundle_root=bundle_root,
        model=model,
        extra_cond=extra_cond,
        txt_embeds=txt_embeds,
        neg_embeds=neg_embeds,
        txt_attn_mask=txt_attn_mask,
        neg_attn_mask=neg_attn_mask,
        pooled_embeds=pooled_embeds,
        neg_pooled_embeds=neg_pooled_embeds,
        encoder_type=encoder_type,
        scheduler=scheduler,
        timesteps=timesteps,
        sigmas=sigmas,
        sched_ts=sched_ts,
        timestep_embed_schedule=timestep_embed_schedule,
        semantics=semantics,
        latents=latents,
        w=w,
        h=h,
        lh=_lh,
        lw=_lw,
        seed=seed,
        steps=steps,
        guidance=guidance,
        flux_unpack=_unpack_flux1_latents,
        preview_mode=preview_mode,
        preview_interval=preview_interval,
        preview_max_edge=preview_max_edge,
        preview_state=preview_state,
        on_progress=on_progress,
        on_log=on_log,
    )


def execute_flux1_fill_edit_denoise(ctx: Any) -> Any | None:
    """Fill denoise + unpack to NCHW latents."""
    from backend.engine.families.flux1.transformer_mlx import _unpack_flux1_latents

    pipeline = ctx.pipeline
    preview_state = ctx.preview_state
    preview_state["on_log"] = ctx.on_log
    if ctx.preview_mode == "stream":
        warm_image_step_preview_decoders(
            pipeline,
            ctx.entry,
            ctx.version_key or None,
            preview_state,
            config=ctx.config,
            on_log=ctx.on_log,
        )
        try:
            preview_state["vae_session"] = build_image_vae_preview_session(
                pipeline,
                ctx.entry, ctx.version_key or None, on_log=ctx.on_log
            )
        except Exception as exc:
            preview_state["vae_session"] = False
            if ctx.on_log:
                ctx.on_log("warning", f"preview VAE warmup skipped: {exc}")

    pipeline_graph_step("denoise", ctx.on_log)
    latents = run_image_denoise(
        pipeline,
        model=ctx.model,
        scheduler=ctx.scheduler,
        timesteps=ctx.timesteps,
        latents=ctx.latents,
        config=ctx.config,
        runtime_contract=ctx.runtime_contract,
        guidance=ctx.guidance,
        txt_embeds=ctx.txt_embeds,
        neg_embeds=ctx.neg_embeds,
        pooled_embeds=ctx.pooled_embeds,
        neg_pooled_embeds=ctx.neg_pooled_embeds,
        txt_attn_mask=ctx.txt_attn_mask,
        neg_attn_mask=ctx.neg_attn_mask,
        encoder_type=ctx.encoder_type,
        width=ctx.w,
        height=ctx.h,
        sched_ts=ctx.sched_ts,
        sigmas=ctx.sigmas,
        timestep_embed_schedule=ctx.timestep_embed_schedule,
        extra_cond=ctx.extra_cond,
        semantics=ctx.semantics,
        ctx_exec=ctx.exec_ctx,
        on_progress=ctx.on_progress,
        on_log=ctx.on_log,
        preview_mode=ctx.preview_mode,
        preview_interval=ctx.preview_interval,
        preview_max_edge=ctx.preview_max_edge,
        preview_state=preview_state,
        entry=ctx.entry,
        version_key=ctx.version_key or None,
        packed_denoise=True,
        flux_unpack=ctx.flux_unpack,
        latent_h=ctx.lh,
        latent_w=ctx.lw,
    )
    if latents is None:
        return None
    if ctx.exec_ctx.cancel_token.is_cancelled():
        return None
    return ctx.flux_unpack(pipeline.ctx, latents, ctx.lh, ctx.lw)


def persist_flux1_fill_edit(ctx: Any, latents: Any) -> tuple[str, dict[str, Any]] | None:
    """Decode Fill latents and persist via pipeline finalize."""
    if ctx.exec_ctx.cancel_token.is_cancelled():
        return None
    return finalize_image_from_latents(
        ctx.pipeline,
        latents=latents,
        timesteps=ctx.timesteps,
        entry=ctx.entry,
        version_key=ctx.version_key,
        model_key=ctx.model_key,
        seed=ctx.seed,
        request=ctx.request,
        ctx_exec=ctx.exec_ctx,
        steps=ctx.steps,
        guidance=ctx.guidance,
        w=ctx.w,
        h=ctx.h,
        on_progress=ctx.on_progress,
        on_log=ctx.on_log,
        name_infix="_fill",
        extra_meta={
            "operation": ctx.request.operation,
            "fill_model": ctx.model_key,
        },
    )

