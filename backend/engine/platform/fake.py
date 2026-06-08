"""Fake platform for paradigm/session unit tests without MLX/CUDA."""

from __future__ import annotations

from typing import Any

import numpy as np

from backend.engine.platform.session import PlatformSession


class NumpyRef:
    """Minimal opaque tensor stand-in."""

    def __init__(self, arr: np.ndarray) -> None:
        self._arr = arr

    @property
    def shape(self) -> tuple[int, ...]:
        return self._arr.shape

    def astype(self, *_a: Any, **_k: Any) -> NumpyRef:
        return self


class FakeKernels:
    backend = "fake"

    def randn(self, shape: tuple[int, ...], dtype: Any = None) -> NumpyRef:
        return NumpyRef(np.zeros(shape, dtype=np.float32))

    def seeded_randn(self, shape: tuple[int, ...], seed: int, dtype: Any = None) -> NumpyRef:
        rng = np.random.default_rng(seed)
        return NumpyRef(rng.standard_normal(shape).astype(np.float32))

    def float32(self) -> Any:
        return np.float32

    def eval(self, *_args: Any) -> None:
        return None


def fake_platform() -> PlatformSession:
    return PlatformSession(backend="fake", device="cpu").bind_kernels(FakeKernels())
