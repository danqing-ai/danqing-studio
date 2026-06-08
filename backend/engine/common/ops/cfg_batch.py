"""Batched classifier-free guidance — one DiT forward for pos + neg branches."""
from __future__ import annotations

import importlib
from typing import Any, Callable

from backend.engine.runtime._base import RuntimeContext

TEXT_KEYS_MINIMAL = frozenset({"txt_embeds"})
TEXT_KEYS_WITH_MASK = frozenset({"txt_embeds", "txt_attn_mask"})
HUNYUAN_CFG_TEXT_KEYS = frozenset(
    {"txt_embeds", "txt_attn_mask", "txt_embeds_2", "txt_attn_mask_2"}
)
FIBO_CFG_TEXT_KEYS = frozenset({"txt_embeds", "text_encoder_layers"})


def broadcast_batch(ctx: RuntimeContext, tensor: Any | None, batch_size: int) -> Any | None:
    if tensor is None:
        return None
    b = int(tensor.shape[0])
    if b == batch_size:
        return tensor
    if b == 1:
        if hasattr(ctx, "repeat"):
            return ctx.repeat(tensor, batch_size, axis=0)
        # Backward-compatible fallback for lightweight test/mocking contexts.
        mx = importlib.import_module("mlx.core")
        return mx.repeat(tensor, batch_size, axis=0)
    raise RuntimeError(
        f"CFG batch broadcast: expected batch 1 or {batch_size}, got {b}"
    )


def merge_cfg_forward_kwargs(
    ctx: RuntimeContext,
    pos: dict[str, Any],
    neg: dict[str, Any],
    *,
    text_keys: frozenset[str],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in pos:
        if key in text_keys:
            pv = pos.get(key)
            nv = neg.get(key)
            if pv is None or nv is None:
                raise RuntimeError(
                    f"Batched CFG requires both branches for {key!r}"
                )
            out[key] = ctx.concat([pv, nv], axis=0)
        else:
            out[key] = pos[key]
    return out


def predict_noise_cfg_batched(
    forward: Callable[..., Any],
    ctx: RuntimeContext,
    latents_in: Any,
    t: Any,
    *,
    guidance: float,
    pos_kwargs: dict[str, Any],
    neg_kwargs: dict[str, Any],
    text_keys: frozenset[str],
    combine_cfg_noise: Callable[[Any, Any, float], Any],
    refine_cfg_noise: Callable[..., Any] | None = None,
    cfg_renorm: bool = False,
    cfg_renorm_min: float = 0.0,
) -> Any:
    """Run ``forward`` once with latents/text batched on axis 0, then merge CFG."""
    batched_latents = ctx.concat([latents_in, latents_in], axis=0)
    batched_kwargs = merge_cfg_forward_kwargs(
        ctx, pos_kwargs, neg_kwargs, text_keys=text_keys,
    )
    noise = forward(batched_latents, t, **batched_kwargs)
    noise_cond = noise[0:1]
    noise_uncond = noise[1:2]
    noise_pred = combine_cfg_noise(noise_cond, noise_uncond, guidance)
    if cfg_renorm and refine_cfg_noise is not None:
        noise_pred = refine_cfg_noise(
            noise_cond, noise_pred, cfg_renorm_min=cfg_renorm_min,
        )
    return noise_pred
