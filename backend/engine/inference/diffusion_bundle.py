"""Shared helpers for building and running ``DiffusionInference`` bundles."""
from __future__ import annotations

from typing import Any, Callable

from backend.engine.inference._protocols import InferenceBundle
from backend.engine.inference.cfg_strategies import resolve_cfg_strategy
from backend.engine.inference.diffusion import DiffusionInference


def run_diffusion_denoise(
    ctx: Any,
    *,
    model: Any,
    config: Any,
    scheduler: Any,
    timesteps: Any,
    latents: Any,
    guidance: float,
    cancel_token: Any | None,
    step_kwargs_builder: Any,
    sigmas: Any | None = None,
    cfg_renorm: bool = False,
    cfg_renorm_min: float = 0.0,
    cfg_strategy: Any | None = None,
    pack_fn: Callable | None = None,
    unpack_fn: Callable | None = None,
    latent_h: int = 0,
    latent_w: int = 0,
    init_noise_sigma: float = 1.0,
    scale_model_input: bool = False,
    step_post_fns: list[Callable] | None = None,
    on_step_complete: Callable | None = None,
) -> Any | None:
    """Build ``InferenceBundle`` and run ``DiffusionInference`` (image + video)."""
    bundle = InferenceBundle(
        ctx=ctx,
        model=model,
        config=config,
        scheduler=scheduler,
        timesteps=timesteps,
        sigmas=sigmas,
        guidance=guidance,
        cfg_renorm=cfg_renorm,
        cfg_renorm_min=cfg_renorm_min,
        cancel_token=cancel_token,
        cfg_strategy=cfg_strategy or resolve_cfg_strategy(model, config, ctx),
        step_kwargs_builder=step_kwargs_builder,
        pack_fn=pack_fn,
        unpack_fn=unpack_fn,
        latent_h=latent_h,
        latent_w=latent_w,
        init_latents=latents,
        init_noise_sigma=init_noise_sigma,
        scale_model_input=scale_model_input,
        step_post_fns=step_post_fns or [],
        on_step_complete=on_step_complete,
    )
    return DiffusionInference(ctx).run(bundle)
