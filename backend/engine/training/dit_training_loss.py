"""Shared flow-matching training loss (min-SNR weighting, prior preservation)."""

from __future__ import annotations

from typing import Any, Callable

import mlx.core as mx


def min_snr_weight(sigma: mx.array, gamma: float) -> mx.array:
    """Per-sample SNR weight for linear flow-matching with x_t = (1-σ)x0 + σε."""
    if gamma <= 0:
        return mx.ones_like(sigma)
    s = mx.reshape(sigma, (-1,) + (1,) * (sigma.ndim - 1 if sigma.ndim > 1 else 0))
    s = mx.clip(s, 1e-4, 1.0 - 1e-4)
    snr = mx.square((1.0 - s) / s)
    return mx.minimum(snr, gamma) / mx.maximum(snr, 1e-8)


def flow_match_mse(
    pred: mx.array,
    x0: mx.array,
    eps: mx.array,
    *,
    sigma: mx.array,
    min_snr_gamma: float = 0.0,
) -> mx.array:
    err = mx.square(pred + x0 - eps)
    if min_snr_gamma > 0:
        w = min_snr_weight(sigma, min_snr_gamma)
        while w.ndim < err.ndim:
            w = mx.expand_dims(w, axis=-1)
        return mx.mean(w * err)
    return mx.mean(err)


def sample_noisy_latent(x0: mx.array, ctx: Any) -> tuple[mx.array, mx.array, mx.array]:
    b = x0.shape[0]
    t = mx.random.uniform(shape=(b,), dtype=ctx.float32())
    eps = mx.random.normal(x0.shape, dtype=ctx.bfloat16())
    sigma = mx.reshape(t, (b,) + (1,) * (x0.ndim - 1)).astype(ctx.bfloat16())
    x_t = (1.0 - sigma) * x0 + sigma * eps
    x_t = mx.stop_gradient(x_t)
    return x_t, eps, t


def combine_instance_prior_loss(
    instance_loss: mx.array,
    prior_loss: mx.array | None,
    *,
    prior_loss_weight: float,
) -> mx.array:
    if prior_loss is None or prior_loss_weight <= 0:
        return instance_loss
    return instance_loss + float(prior_loss_weight) * prior_loss


def make_prior_latent(x0: mx.array, ctx: Any) -> mx.array:
    """DreamBooth-style prior: sample x0 from the latent prior (standard normal)."""
    return mx.random.normal(x0.shape, dtype=ctx.bfloat16())


def wrap_loss_with_prior(
    instance_loss_fn: Callable[..., mx.array],
    *,
    prior_loss_fn: Callable[..., mx.array] | None,
    prior_loss_weight: float,
) -> Callable[..., mx.array]:
    if prior_loss_fn is None or prior_loss_weight <= 0:
        return instance_loss_fn

    def _loss(*args: Any, **kwargs: Any) -> mx.array:
        inst = instance_loss_fn(*args, **kwargs)
        prior = prior_loss_fn(*args, **kwargs)
        return combine_instance_prior_loss(inst, prior, prior_loss_weight=prior_loss_weight)

    return _loss
