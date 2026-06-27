"""Step1X-Edit flow-match schedule — numpy (no torch)."""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np


def time_shift(mu: float, sigma: float, t: np.ndarray) -> np.ndarray:
    return np.exp(mu) / (np.exp(mu) + (1.0 / t - 1.0) ** sigma)


def get_lin_function(
    x1: float = 256,
    y1: float = 0.5,
    x2: float = 4096,
    y2: float = 1.15,
) -> Callable[[float], float]:
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def get_schedule(
    num_steps: int,
    image_seq_len: int,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    shift: bool = True,
) -> list[float]:
    timesteps = np.linspace(1.0, 0.0, num_steps + 1, dtype=np.float64)
    if shift:
        mu = get_lin_function(y1=base_shift, y2=max_shift)(float(image_seq_len))
        timesteps = time_shift(mu, 1.0, timesteps)
    return timesteps.astype(np.float32).tolist()
