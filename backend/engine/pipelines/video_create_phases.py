"""Video create phased helpers (``VideoSession``)."""

from __future__ import annotations

import random
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from backend.core.contracts import (
    ExecutionContext,
    VideoEditRequest,
    VideoGenerationRequest,
    parse_size,
)
from backend.engine.config.model_configs import get_config_class
from backend.engine.contracts import (
    registry_scalar_default as _registry_scalar_default_fn,
    video_apply_i2v_conditioning,
    video_i2v_encode_failure_message,
    video_infer_log_extras,
    video_prepare_i2v_source_image,
    video_rotary_model_kwargs,
)
from backend.engine.families._video_backbone import plugin_video_backbone_model_if_ready
from backend.engine.inference.video_denoise import run_video_denoise
from backend.engine.inference.video_two_stage import run_family_video_generator
from backend.engine.pipelines.pipeline_progress import pipeline_graph_step
from backend.engine.sessions._context import MediaRunContext, ResolvedRun
from backend.engine.pipelines.video_run_common import (
    apply_video_registry_config_overrides,
    create_timesteps_for_video,
    finalize_video_from_latents,
    initial_video_latents,
    prepare_video_bundle_and_schedule,
    resolve_fps,
    resolve_num_frames,
    uses_family_video_generator,
    vae_encode_frame,
    video_encode_load_and_condition,
)
from backend.engine.protocols.plugin import FamilyPlugin

PhaseCmFactory = Callable[[str], AbstractContextManager[Any]]


def execute_video_denoise(
    pipeline: Any,
    *,
    latents: Any,
    timesteps: Any,
    scheduler: Any,
    model: Any,
    txt_embeds: Any,
    neg_embeds: Any,
    guidance: float,
    config: Any,
    sigmas: Any,
    timestep_embed_schedule: list[float] | None,
    extra_cond: dict[str, Any],
    rope_kw: dict[str, Any],
    cfg_renorm: bool,
    cfg_renorm_min: float,
    ctx_exec: ExecutionContext,
    on_progress: Callable | None,
    on_log: Callable | None,
) -> Any | None:
    """Video denoise — ``DiffusionParadigm`` + loaded backbone."""
    return run_video_denoise(
        pipeline,
        model=model,
        scheduler=scheduler,
        timesteps=timesteps,
        latents=latents,
        config=config,
        guidance=guidance,
        txt_embeds=txt_embeds,
        neg_embeds=neg_embeds,
        sigmas=sigmas,
        timestep_embed_schedule=timestep_embed_schedule,
        extra_cond=extra_cond,
        rope_kw=rope_kw,
        cfg_renorm=cfg_renorm,
        cfg_renorm_min=cfg_renorm_min,
        ctx_exec=ctx_exec,
        on_progress=on_progress,
        on_log=on_log,
    )


@dataclass
class VideoCreateRunContext(MediaRunContext):
    """Mutable state for one video create/edit run (encode → schedule → infer → persist)."""

    pipeline: Any
    request: VideoGenerationRequest | VideoEditRequest
    exec_ctx: ExecutionContext
    entry: Any
    config: Any
    family: str
    model_key: str
    version_key: str | None
    bundle_root: Path | None
    model: Any
    extra_cond: dict[str, Any]
    txt_embeds: Any
    neg_embeds: Any
    scheduler: Any
    timesteps: list[Any]
    sigmas: Any
    timestep_embed_schedule: list[float] | None
    latents: Any
    rope_kw: dict[str, Any]
    w: int
    h: int
    num_frames: int
    fps: int
    seed: int
    steps: int
    guidance: float
    cfg_renorm: bool
    cfg_renorm_min: float
    mode: str
    on_progress: Callable | None = None
    on_log: Callable | None = None

    def session_infer(
        self,
        *,
        pipeline: Any | None = None,
        **_ignored: Any,
    ) -> Any | None:
        if pipeline is None:
            pipeline = self.pipeline
        return execute_video_denoise(pipeline, **video_denoise_kwargs_from_ctx(self))


def video_denoise_kwargs_from_ctx(ctx: VideoCreateRunContext) -> dict[str, Any]:
    return {
        "latents": ctx.latents,
        "timesteps": ctx.timesteps,
        "scheduler": ctx.scheduler,
        "model": ctx.model,
        "txt_embeds": ctx.txt_embeds,
        "neg_embeds": ctx.neg_embeds,
        "guidance": ctx.guidance,
        "config": ctx.config,
        "sigmas": ctx.sigmas,
        "timestep_embed_schedule": ctx.timestep_embed_schedule,
        "extra_cond": ctx.extra_cond,
        "rope_kw": ctx.rope_kw,
        "cfg_renorm": ctx.cfg_renorm,
        "cfg_renorm_min": ctx.cfg_renorm_min,
        "ctx_exec": ctx.exec_ctx,
        "on_progress": ctx.on_progress,
        "on_log": ctx.on_log,
    }


def persist_video_create(
    ctx: VideoCreateRunContext,
    latents: Any,
) -> tuple[str, dict[str, Any]] | None:
    return finalize_video_from_latents(
        ctx.pipeline,
        latents=latents,
        timesteps=ctx.timesteps,
        entry=ctx.entry,
        version_key=ctx.version_key,
        config=ctx.config,
        model_key=ctx.model_key,
        seed=ctx.seed,
        request=ctx.request,
        ctx_exec=ctx.exec_ctx,
        fps=ctx.fps,
        num_frames=ctx.num_frames,
        w=ctx.w,
        h=ctx.h,
        steps=ctx.steps,
        guidance=ctx.guidance,
        on_progress=ctx.on_progress,
        on_log=ctx.on_log,
    )


def _apply_video_i2v_source(
    pipeline: Any,
    request: VideoEditRequest,
    *,
    config: Any,
    latents: Any,
    extra_cond: dict[str, Any],
    w: int,
    h: int,
    entry: Any,
    version_key: str | None,
) -> Any:
    if not request.source_asset_id:
        return latents
    src_path = pipeline._asset_store.get_file_path(request.source_asset_id)
    if not src_path or not src_path.exists():
        return latents
    from PIL import Image

    import numpy as np

    src_img = Image.open(str(src_path)).convert("RGB")
    src_img = video_prepare_i2v_source_image(config, src_img, w, h)
    src_array = np.array(src_img).astype(np.float32) / 127.5 - 1.0
    src_tensor = pipeline.ctx.array(np.expand_dims(src_array, 0))
    vae_latent = vae_encode_frame(
        pipeline,
        src_tensor,
        entry,
        version_key or None,
        config,
    )
    if vae_latent is None:
        raise RuntimeError(video_i2v_encode_failure_message(config))
    return video_apply_i2v_conditioning(
        config,
        pipeline.ctx,
        latents,
        vae_latent,
        extra_cond,
    )


def build_video_create_run_context(
    pipeline: Any,
    request: VideoGenerationRequest | VideoEditRequest,
    exec_ctx: ExecutionContext,
    *,
    resolved: ResolvedRun,
    is_edit: bool,
    on_progress: Callable | None = None,
    on_log: Callable | None = None,
    phase_cm: PhaseCmFactory | None = None,
    plugin: FamilyPlugin | None = None,
) -> VideoCreateRunContext | None:
    """Prepare encode + schedule + latent state for diffusion video create/edit."""
    phase_cm = phase_cm or (lambda _name: nullcontext())

    model_key = resolved.model_id
    version_key = resolved.version_key
    entry = resolved.registry_entry
    family = resolved.family_id
    w, h = parse_size(request.size)
    config_cls = get_config_class(family)
    config = config_cls()
    num_frames = resolve_num_frames(pipeline, request, entry)
    fps = resolve_fps(pipeline, request, entry)
    seed = request.seed if request.seed is not None else random.randint(0, 2 ** 32 - 1)

    apply_video_registry_config_overrides(pipeline, entry, config)

    if exec_ctx.cancel_token.is_cancelled():
        return None

    bundle_root, w, h, steps, guidance, step_distill, scheduler_default = (
        prepare_video_bundle_and_schedule(
            pipeline,
            entry=entry,
            config=config,
            family=family,
            model_key=model_key,
            version_key=version_key,
            w=w,
            h=h,
            num_frames=num_frames,
            request=request,
            on_log=on_log,
        )
    )

    preloaded_model = plugin_video_backbone_model_if_ready(
        plugin, config=config, num_frames=num_frames
    )
    with phase_cm("encode"):
        enc_loaded = video_encode_load_and_condition(
            pipeline,
            request=request,
            entry=entry,
            config=config,
            family=family,
            bundle_root=bundle_root,
            version_key=version_key,
            model_key=model_key,
            num_frames=num_frames,
            guidance=guidance,
            ctx_exec=exec_ctx,
            on_log=on_log,
            preloaded_model=preloaded_model,
        )
    if enc_loaded is None:
        return None
    model, extra_cond, txt_embeds, neg_embeds, latent_frames = enc_loaded

    with phase_cm("schedule"):
        scheduler, timesteps, sigmas, timestep_embed_schedule, wan_shift = (
            create_timesteps_for_video(
                pipeline,
                entry=entry,
                config=config,
                request=request,
                steps=steps,
                step_distill=step_distill,
                scheduler_default=scheduler_default,
                bundle_root=bundle_root,
                w=w,
                h=h,
                num_frames=num_frames,
                on_log=on_log,
            )
        )

    cfg_renorm = bool(_registry_scalar_default_fn(entry, "enable_cfg_renorm", False))
    cfg_renorm_min = float(_registry_scalar_default_fn(entry, "cfg_renorm_min", 0.0))
    mode = "video_edit" if is_edit else "video_generate"

    if on_log:
        vae_scale = getattr(config, "vae_scale", 8)
        parts = [
            f"infer model={model_key}",
            f"family={family}",
            f"version={version_key or 'default'}",
            f"size={w}x{h}",
            f"frames={num_frames}",
            f"fps={fps}",
            f"seed={seed}",
            f"steps={steps}",
            f"guidance={guidance}",
            f"scheduler={scheduler_default}",
            f"supports_guidance={getattr(config, 'supports_guidance', False)}",
            f"cfg_on={bool(neg_embeds is not None)}",
            f"vae_scale={vae_scale}",
            f"mode={mode}",
        ]
        if wan_shift is not None:
            parts.append(f"shift={wan_shift}")
        if cfg_renorm:
            parts.append(f"cfg_renorm=True cfg_renorm_min={cfg_renorm_min}")
        parts.extend(video_infer_log_extras(config, scheduler, extra_cond))
        on_log("info", " ".join(parts))

    vae_scale = getattr(config, "vae_scale", 8)
    latent_c = int(getattr(config, "vae_z_dim", None) or config.dim_in)
    latent_shape = (1, latent_c, latent_frames, h // vae_scale, w // vae_scale)
    latents = initial_video_latents(pipeline, config, latent_shape, seed)

    if is_edit:
        latents = _apply_video_i2v_source(
            pipeline,
            request,  # type: ignore[arg-type]
            config=config,
            latents=latents,
            extra_cond=extra_cond,
            w=w,
            h=h,
            entry=entry,
            version_key=version_key,
        )

    extra_cond["_pipeline_fps"] = float(fps)
    latents, extra_cond = model.before_denoise(latents, timesteps, sigmas, **extra_cond)
    rope_kw = video_rotary_model_kwargs(config, pipeline.ctx, h, w, latents)

    return VideoCreateRunContext(
        pipeline=pipeline,
        request=request,
        exec_ctx=exec_ctx,
        entry=entry,
        config=config,
        family=family,
        model_key=model_key,
        version_key=version_key,
        bundle_root=bundle_root,
        model=model,
        extra_cond=extra_cond,
        txt_embeds=txt_embeds,
        neg_embeds=neg_embeds,
        scheduler=scheduler,
        timesteps=timesteps,
        sigmas=sigmas,
        timestep_embed_schedule=timestep_embed_schedule,
        latents=latents,
        rope_kw=rope_kw,
        w=w,
        h=h,
        num_frames=num_frames,
        fps=fps,
        seed=seed,
        steps=steps,
        guidance=guidance,
        cfg_renorm=cfg_renorm,
        cfg_renorm_min=cfg_renorm_min,
        mode=mode,
        on_progress=on_progress,
        on_log=on_log,
    )


def _maybe_run_video_family_generator(
    pipeline: Any,
    request: VideoGenerationRequest | VideoEditRequest,
    ctx_exec: ExecutionContext,
    resolved: ResolvedRun,
    *,
    is_edit: bool,
    phase_cm: PhaseCmFactory,
    on_progress: Callable | None,
    on_log: Callable | None,
) -> Any:
    """Return family-generator result when configured; otherwise ``_NOT_GENERATOR``."""
    entry = resolved.registry_entry
    family = resolved.family_id
    config = get_config_class(family)()
    apply_video_registry_config_overrides(pipeline, entry, config)
    if not uses_family_video_generator(config):
        return _NOT_GENERATOR
    with phase_cm("infer"):
        return run_family_video_generator(
            pipeline,
            request,
            ctx_exec,
            is_edit=is_edit,
            on_progress=on_progress,
            on_log=on_log,
        )


_NOT_GENERATOR = object()
