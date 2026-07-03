"""Image N-step denoise — ``DiffusionInference`` + ``ImageStepKwargsBuilder``."""

from __future__ import annotations

from typing import Any, Callable

from backend.core.contracts import ExecutionContext
from backend.engine.contracts import FamilyRuntimeContract
from backend.engine.inference._protocols import DenoiseStepResult, InferenceBundle
from backend.engine.inference._runtime import inference_span
from backend.engine.inference.cfg_strategies import resolve_cfg_strategy
from backend.engine.inference.diffusion import DiffusionInference
from backend.engine.inference.step_kwargs_builders import ImageStepKwargsBuilder
from backend.engine.pipelines.image_run_common import maybe_emit_image_step_preview
from backend.engine.pipelines.pipeline_progress import emit_denoise_progress
from backend.engine.common.ops.step_cache import log_step_cache_summary
from backend.engine.common.ops.teacache_calibrate import (
    publish_teacache_probe_from_model,
    teacache_probe_enabled,
)
from backend.engine.inference.optimization_plan import stash_inference_run_metadata


def run_image_denoise(
    pipeline: Any,
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
    ctx = pipeline.ctx
    cfg_strategy = resolve_cfg_strategy(model, config, ctx)
    builder = ImageStepKwargsBuilder(
        runtime_contract=runtime_contract,
        txt_embeds=txt_embeds,
        neg_embeds=neg_embeds,
        pooled_embeds=pooled_embeds,
        neg_pooled_embeds=neg_pooled_embeds,
        extra_cond=extra_cond,
        guidance=guidance,
        sigmas=sigmas,
        timestep_embed_schedule=timestep_embed_schedule,
        timestep_offset=timestep_offset,
        encoder_type=encoder_type,
        width=width,
        height=height,
        sched_ts=sched_ts,
        txt_attn_mask=txt_attn_mask,
        neg_attn_mask=neg_attn_mask,
    )

    n_steps = len(timesteps)

    def _on_step_complete(result: DenoiseStepResult) -> None:
        i = result.step_idx
        emit_denoise_progress(on_progress, i + 1, n_steps)
        if preview_mode == "stream" and preview_state is not None and entry is not None:
            maybe_emit_image_step_preview(
                pipeline,
                step_index_0based=i,
                n_steps=n_steps,
                latents=result.latents,
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
        if on_log and (i == 0 or (i + 1) % max(1, preview_interval) == 0 or i + 1 == n_steps):
            sched = timestep_embed_schedule
            te_idx = timestep_offset + i
            t_embed = sched[te_idx] if sched is not None and te_idx < len(sched) else None
            extra = f" t_embed={t_embed:.6g}" if t_embed is not None else ""
            on_log("info", f"Step {i + 1}/{n_steps}{extra}")

    bundle = InferenceBundle(
        ctx=ctx,
        model=model,
        config=config,
        scheduler=scheduler,
        timesteps=timesteps,
        init_latents=latents,
        guidance=guidance,
        cancel_token=ctx_exec.cancel_token,
        step_kwargs_builder=builder,
        sigmas=sigmas,
        cfg_renorm=bool(getattr(semantics, "cfg_renorm", False)),
        cfg_renorm_min=float(getattr(semantics, "cfg_renorm_min", 0.0)),
        cfg_strategy=cfg_strategy,
        pack_fn=flux_pack if packed_denoise else None,
        unpack_fn=flux_unpack if packed_denoise else None,
        latent_h=latent_h,
        latent_w=latent_w,
        on_step_complete=_on_step_complete,
    )

    with inference_span(ctx_exec, "diffusion_paradigm"):
        result = DiffusionInference(ctx).run(bundle)
    if teacache_probe_enabled():
        publish_teacache_probe_from_model(model)
    else:
        log_step_cache_summary(model, on_log)
        stash_inference_run_metadata(model, extra_cond)
    return result
