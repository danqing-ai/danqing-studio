"""Latent-space refinement before VAE decode (Z-Image hires)."""
from __future__ import annotations

from typing import Any, Callable


def apply_latent_refine_if_requested(
    pipeline: Any,
    latents: Any,
    *,
    request: Any,
    entry: Any,
    version_key: str | None,
    model: Any,
    timesteps: list[Any],
    sigmas: Any,
    txt_embeds: Any,
    neg_embeds: Any | None,
    guidance: float,
    extra_cond: dict[str, Any],
    on_log: Callable[..., None] | None = None,
) -> Any:
    spec = getattr(request, "latent_refine", None)
    if spec is None:
        return latents
    scale = float(getattr(spec, "scale", 1.0) or 1.0)
    if scale <= 1.0 + 1e-6:
        return latents

    from backend.engine.config.model_configs import get_config_class

    family = getattr(getattr(entry, "runtime", None), "family", None) or ""
    if not getattr(get_config_class(family)(), "supports_latent_refine", False):
        raise RuntimeError(f"latent_refine is not supported for family={family!r}")

    from backend.engine.common.mlx_only import require_mlx_backend

    require_mlx_backend(pipeline.ctx, feature="latent_refine")
    ctx = pipeline.ctx
    denoise_strength = float(getattr(spec, "denoise_strength", 0.35) or 0.35)
    hires_steps = int(getattr(spec, "hires_steps", 0) or 0)
    if hires_steps <= 0:
        hires_steps = max(4, int(round(denoise_strength * 12)))

    _c, f, h, w = latents.shape if latents.ndim == 4 else (latents.shape[1], latents.shape[2], latents.shape[3], latents.shape[4])
    if latents.ndim == 5:
        latents = latents[0]
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    mode = str(getattr(spec, "interpolation", "linear") or "linear").lower()
    upscaled = _resize_latent_nchw(ctx, latents, new_h, new_w, mode=mode)
    if on_log:
        on_log(
            "info",
            f"latent_refine scale={scale:.2f} {h}x{w}->{new_h}x{new_w} "
            f"hires_steps={hires_steps} denoise_strength={denoise_strength:.2f}",
        )

    noise = pipeline.ctx.seeded_randn(upscaled.shape, int(getattr(request, "seed", None) or 0) + 991, dtype=upscaled.dtype)
    blend = max(0.0, min(1.0, denoise_strength))
    latents = (1.0 - blend) * upscaled + blend * noise

    cond = dict(extra_cond)
    cond["lemica_mode"] = "none"
    for step_i in range(hires_steps):
        t_idx = step_i
        if t_idx >= len(timesteps):
            break
        t = timesteps[t_idx]
        latents = model.forward(
            latents,
            t,
            txt_embeds=txt_embeds,
            neg_embeds=neg_embeds,
            sigmas=sigmas,
            guidance=guidance,
            **cond,
        )
    ctx.eval(latents)
    return latents


def _resize_latent_nchw(ctx: Any, latents: Any, new_h: int, new_w: int, *, mode: str) -> Any:
    import mlx.core as mx

    if getattr(ctx, "backend", None) != "mlx":
        raise RuntimeError("latent_refine is MLX-only today")
    # [C,F,H,W] → upsample H,W via mx resize (treat as NCHW with F=1)
    c, f, h, w = latents.shape
    flat = mx.reshape(latents, (c * f, h, w))
    flat = mx.expand_dims(flat, 0)
    if mode == "nearest":
        out = mx.image.resize(flat, (new_h, new_w), interpolation="nearest")
    elif mode == "cubic":
        out = mx.image.resize(flat, (new_h, new_w), interpolation="cubic")
    else:
        out = mx.image.resize(flat, (new_h, new_w), interpolation="linear")
    out = mx.reshape(out[0], (c, f, new_h, new_w))
    return out
