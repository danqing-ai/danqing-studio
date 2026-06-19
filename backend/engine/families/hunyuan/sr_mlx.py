"""HunyuanVideo-1.5 1080p video super-resolution — reuses DiT with ``use_meanflow=True``."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from backend.engine.families.hunyuan.transformer_mlx import HunyuanVideoDiTMLX
from backend.engine.families.hunyuan.weights import remap_hunyuan_weights


def load_hunyuan_sr_transformer(
    ctx: Any,
    config: Any,
    bundle_root: Path,
    *,
    model_cache: Any | None = None,
    cache_key: str | None = None,
    cache_size_gb: float = 10.0,
) -> HunyuanVideoDiTMLX:
    """Load SR DiT from ``bundle_root/transformer/`` (community 1080p-2SR layout)."""
    if model_cache is not None and cache_key:
        cached = model_cache.get(cache_key)
        if cached is not None:
            return cached

    tp = bundle_root / "transformer"
    if not tp.is_dir():
        raise RuntimeError(f"HunyuanVideo SR bundle missing transformer/ under {bundle_root}")

    shards = sorted(tp.glob("*.safetensors"))
    if not shards:
        raise RuntimeError(f"No transformer shards under {tp}")

    w: dict[str, Any] = {}
    for sf in shards:
        w.update(ctx.load_weights(str(sf)))
    w = remap_hunyuan_weights(w)

    sr_config = config
    object.__setattr__(sr_config, "use_meanflow", True)

    model = HunyuanVideoDiTMLX(sr_config, ctx)
    model.load_weights(list(w.items()), strict=False, ctx=ctx)
    ctx.eval(*[p for _, p in model.parameters()])
    if model_cache is not None and cache_key:
        model_cache.put(cache_key, model, float(cache_size_gb))
    return model


def configure_hunyuan_step_distill_timesteps(ctx: Any, sched: Any, steps: int) -> Any:
    """Match diffusers HunyuanVideo-1.5 step-distill sigma schedule (linspace 1→0)."""
    sigmas_arr = np.linspace(1.0, 0.0, steps + 1, dtype=np.float32)[:-1]
    sched.set_timesteps(steps, use_empirical_mu=False)
    sched._sigmas = ctx.concat([
        ctx.array(sigmas_arr, dtype=ctx.float32()),
        ctx.zeros((1,), dtype=ctx.float32()),
    ], axis=0)
    return ctx.array(
        sigmas_arr * float(sched.num_train_timesteps),
        dtype=ctx.float32(),
    )


def upscale_latents_to_1080p(
    ctx: Any,
    low_res_latents: Any,
    sr_model: HunyuanVideoDiTMLX,
    *,
    txt_embeds: Any,
    txt_attn_mask: Any,
    txt_embeds_2: Any,
    txt_attn_mask_2: Any,
    steps: int = 6,
) -> Any:
    """Run few-step mean-flow SR on latents (default 6 steps, step-distill sigmas)."""
    from backend.engine.common.ops.schedulers import FlowMatchEulerScheduler
    from backend.engine.inference.diffusion_bundle import run_diffusion_denoise
    from backend.engine.inference.step_kwargs_builders import FixedStepKwargsBuilder

    sched = FlowMatchEulerScheduler(ctx=ctx)
    timesteps = configure_hunyuan_step_distill_timesteps(ctx, sched, steps)

    latents = low_res_latents
    B, C, T, H, W = latents.shape
    cond = ctx.zeros((B, C, T, H, W), dtype=latents.dtype)
    mask = ctx.zeros((B, 1, T, H, W), dtype=latents.dtype)
    vision_tokens = int(getattr(sr_model.config, "vision_num_semantic_tokens", 256))
    image_dim = int(getattr(sr_model.config, "image_embed_dim", 1152))
    image_embeds = ctx.zeros((B, vision_tokens, image_dim), dtype=latents.dtype)

    latents, _ = sr_model.before_denoise(
        latents,
        timesteps,
        None,
        cond_latents=cond,
        mask_concat=mask,
        image_embeds=image_embeds,
    )

    builder = FixedStepKwargsBuilder({
        "txt_embeds": txt_embeds,
        "txt_attn_mask": txt_attn_mask,
        "txt_embeds_2": txt_embeds_2,
        "txt_attn_mask_2": txt_attn_mask_2,
        "image_embeds": image_embeds,
    })
    result = run_diffusion_denoise(
        ctx,
        model=sr_model,
        config=sr_model.config,
        scheduler=sched,
        timesteps=timesteps,
        latents=latents,
        guidance=0.0,
        cancel_token=None,
        step_kwargs_builder=builder,
    )
    if result is None:
        return latents
    if hasattr(ctx, "clear_cache"):
        ctx.clear_cache()
    return result
