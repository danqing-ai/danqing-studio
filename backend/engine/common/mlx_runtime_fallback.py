"""Shared MLX fallback helpers for eval/clear_cache."""
from __future__ import annotations

import importlib
from typing import Any, Iterator

import numpy as np


def _mlx_core() -> Any:
    return importlib.import_module("mlx.core")


def run_eval(eval_fn: Any | None, *values: Any) -> None:
    if eval_fn is not None:
        eval_fn(*values)
    else:
        _mlx_core().eval(*values)


def run_clear_cache(clear_cache_fn: Any | None) -> None:
    if clear_cache_fn is not None:
        clear_cache_fn()
    else:
        _mlx_core().clear_cache()


def load_weights_dict(load_fn: Any | None, path: str) -> dict[str, Any]:
    if load_fn is not None:
        return dict(load_fn(path))
    return dict(_mlx_core().load(path))


def iter_safetensors_float32_numpy(path: str) -> Iterator[tuple[str, np.ndarray]]:
    """Yield ``(key, float32 ndarray)`` from a safetensors shard without PyTorch."""
    core = _mlx_core()
    for key, arr in dict(core.load(path)).items():
        yield key, np.asarray(arr.astype(core.float32), dtype=np.float32)


def random_normal(
    randn_fn: Any | None,
    shape: tuple[int, ...],
    *,
    dtype: Any | None = None,
) -> Any:
    if randn_fn is not None:
        if dtype is None:
            return randn_fn(shape)
        return randn_fn(shape, dtype=dtype)
    core = _mlx_core()
    if dtype is None:
        return core.random.normal(shape)
    return core.random.normal(shape, dtype=dtype)


def seeded_random_normal(
    seeded_randn_fn: Any | None,
    shape: tuple[int, ...],
    seed: int,
    *,
    dtype: Any | None = None,
) -> Any:
    if seeded_randn_fn is not None:
        if dtype is None:
            return seeded_randn_fn(shape, int(seed))
        return seeded_randn_fn(shape, int(seed), dtype=dtype)
    core = _mlx_core()
    key = core.random.key(int(seed))
    if dtype is None:
        return core.random.normal(shape, key=key)
    return core.random.normal(shape, key=key, dtype=dtype)


def set_random_seed(seed_fn: Any | None, seed: int) -> None:
    if seed_fn is not None:
        seed_fn(int(seed))
        return
    _mlx_core().random.seed(int(seed))


def random_uniform(
    uniform_fn: Any | None,
    *,
    shape: tuple[int, ...],
    low: float = 0.0,
    high: float = 1.0,
) -> Any:
    if uniform_fn is not None:
        return uniform_fn(shape=shape, low=low, high=high)
    return _mlx_core().random.uniform(shape=shape, low=low, high=high)


def random_categorical(categorical_fn: Any | None, log_probs: Any) -> Any:
    if categorical_fn is not None:
        return categorical_fn(log_probs)
    return _mlx_core().random.categorical(log_probs)
