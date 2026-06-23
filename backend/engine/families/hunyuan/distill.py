"""HunyuanVideo-1.5 LightX2V 4-step distill scheduler (Hy1.5-Distill-Models)."""
from __future__ import annotations

from typing import Any

import numpy as np

# LightX2V Hunyuan T2V 480p 4-step — denoising_step_list (train-scale indices).
HUNYUAN_DISTILL_DEFAULT_TIMESTEPS: tuple[float, ...] = (1000.0, 750.0, 500.0, 250.0)
HUNYUAN_DISTILL_DEFAULT_SHIFT: float = 9.0


def _hunyuan_distill_full_shifted_sigmas(num_train: int, shift: float) -> np.ndarray:
    sigmas = np.linspace(1.0, 0.0, int(num_train) + 1, dtype=np.float64)[:-1]
    sh = float(shift)
    sigmas = sh * sigmas / (1.0 + (sh - 1.0) * sigmas)
    return sigmas.astype(np.float32)


def resolve_hunyuan_distill_shift(config: Any) -> float:
    for key in ("hunyuan_distill_shift", "shift"):
        val = getattr(config, key, None)
        if val is not None:
            return float(val)
    return HUNYUAN_DISTILL_DEFAULT_SHIFT


def resolve_hunyuan_distill_timesteps(config: Any, steps: int) -> tuple[float, ...]:
    custom = getattr(config, "hunyuan_distill_timesteps", None) or ()
    if custom:
        values = tuple(float(x) for x in custom)
    else:
        values = HUNYUAN_DISTILL_DEFAULT_TIMESTEPS
    if len(values) != int(steps):
        raise RuntimeError(
            f"Hunyuan step-distill requires {len(values)} inference steps (got steps={steps}). "
            f"Use the registry default or set hunyuan_distill_timesteps to match."
        )
    return values


def configure_hunyuan_lightx2v_distill_timesteps(
    ctx: Any,
    scheduler: Any,
    steps: int,
    *,
    config: Any | None = None,
) -> Any:
    """Apply LightX2V Hy1.5 4-step distilled schedule (shift + index into full sigma table)."""
    cfg = config if config is not None else getattr(scheduler, "config", None)
    values = resolve_hunyuan_distill_timesteps(cfg, steps)
    shift = resolve_hunyuan_distill_shift(cfg) if cfg is not None else HUNYUAN_DISTILL_DEFAULT_SHIFT
    n_train = int(getattr(scheduler, "num_train_timesteps", 1000))
    full_sigmas = _hunyuan_distill_full_shifted_sigmas(n_train, shift)
    indices = [int(n_train - float(t)) for t in values]
    picked = full_sigmas[indices]
    sigmas = np.append(picked, 0.0).astype(np.float32)
    timesteps = (picked * float(n_train)).astype(np.float32)

    scheduler._sigmas = sigmas
    scheduler._sigmas_float = sigmas.tolist()
    scheduler._timesteps = ctx.array(timesteps)
    scheduler._step_index = 0
    scheduler._num_steps = int(steps)
    scheduler._model_outputs = [None] * max(1, int(getattr(scheduler, "_solver_order", 1)))
    scheduler._lower_order_nums = 0
    scheduler._last_sample = None
    scheduler._this_order = 1
    return scheduler._timesteps
