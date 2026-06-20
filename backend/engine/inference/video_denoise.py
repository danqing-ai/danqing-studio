"""Video N-step denoise — ``DiffusionInference`` + ``VideoStepKwargsBuilder``."""

from __future__ import annotations

from typing import Any, Callable

from backend.core.contracts import ExecutionContext
from backend.engine.inference._protocols import DenoiseStepResult
from backend.engine.inference._runtime import inference_span
from backend.engine.inference.cfg_strategies import resolve_cfg_strategy
from backend.engine.inference.diffusion_bundle import run_diffusion_denoise
from backend.engine.inference.step_kwargs_builders import VideoStepKwargsBuilder
from backend.engine.pipelines.pipeline_progress import emit_denoise_progress


def run_video_denoise(
    pipeline: Any,
    *,
    model: Any,
    scheduler: Any,
    timesteps: list[Any],
    latents: Any,
    config: Any,
    guidance: float,
    txt_embeds: Any,
    neg_embeds: Any,
    sigmas: Any,
    timestep_embed_schedule: list[float] | None,
    extra_cond: dict[str, Any],
    rope_kw: dict[str, Any],
    cfg_renorm: bool,
    cfg_renorm_min: float,
    ctx_exec: ExecutionContext,
    on_progress: Callable[..., None] | None,
    on_log: Callable[..., None] | None,
) -> Any | None:
    ctx = pipeline.ctx
    cfg_strategy = resolve_cfg_strategy(model, config, ctx)
    use_meanflow = bool(getattr(config, "use_meanflow", False))
    builder = VideoStepKwargsBuilder(
        ctx=ctx,
        model=model,
        txt_embeds=txt_embeds,
        neg_embeds=neg_embeds,
        extra_cond=extra_cond,
        rope_kw=rope_kw,
        sigmas=sigmas,
        timestep_embed_schedule=timestep_embed_schedule,
        timesteps=timesteps,
        use_meanflow=use_meanflow,
    )

    n_steps = len(timesteps)

    def _on_step_complete(result: DenoiseStepResult) -> None:
        if on_progress:
            emit_denoise_progress(on_progress, result.step_idx + 1, n_steps)
        if on_log:
            on_log("info", f"Step {result.step_idx + 1}/{n_steps}")

    step_post_fns: list[Callable] = []
    if hasattr(model, "reblend_i2v_latents"):
        step_post_fns.append(model.reblend_i2v_latents)

    init_sigma = float(getattr(scheduler, "init_noise_sigma", 1.0))

    with inference_span(ctx_exec, "diffusion_paradigm"):
        latents = run_diffusion_denoise(
            ctx,
            model=model,
            config=config,
            scheduler=scheduler,
            timesteps=timesteps,
            latents=latents,
            guidance=guidance,
            cancel_token=ctx_exec.cancel_token,
            step_kwargs_builder=builder,
            sigmas=sigmas,
            cfg_renorm=cfg_renorm,
            cfg_renorm_min=cfg_renorm_min,
            cfg_strategy=cfg_strategy,
            init_noise_sigma=init_sigma,
            scale_model_input=hasattr(scheduler, "scale_model_input"),
            step_post_fns=step_post_fns,
            on_step_complete=_on_step_complete,
        )

    from backend.engine.families.wan.moe import release_wan_moe_experts_if_supported

    release_wan_moe_experts_if_supported(model, ctx)
    return latents
