from __future__ import annotations

"""SeedVR2 超分调度器（job-local MLX 多步实现）。

``backend.engine.common.schedulers.SeedVR2EulerScheduler`` 供 ``get_scheduler()`` /
Pipeline 单步 SR 路径使用；本模块保留 ``job_mlx`` 所需的多步 ``config`` 驱动 Euler，
二者 API 不同，不可互换。
"""

from abc import ABC, abstractmethod

import mlx.core as mx


class BaseScheduler(ABC):
    @property
    @abstractmethod
    def sigmas(self) -> mx.array: ...

    @abstractmethod
    def step(self, noise: mx.array, timestep: int, latents: mx.array, **kwargs) -> mx.array: ...

    def scale_model_input(self, latents: mx.array, t: int) -> mx.array:
        return latents


class SeedVR2EulerScheduler(BaseScheduler):
    def __init__(self, config):
        self.config = config
        self.num_inference_steps = config.num_inference_steps
        self.num_train_timesteps = config.num_train_steps if config.num_train_steps is not None else 1000
        self.cfg_scale = config.guidance
        self.T = float(self.num_train_timesteps)
        self._timesteps, self._sigmas = self._compute_timesteps_and_sigmas()

    @property
    def timesteps(self) -> mx.array:
        return self._timesteps

    @property
    def sigmas(self) -> mx.array:
        return self._sigmas

    def _compute_timesteps_and_sigmas(self) -> tuple[mx.array, mx.array]:
        timesteps_arr = mx.linspace(
            self.T, 0.0, self.num_inference_steps + 1, dtype=mx.float32
        )
        sigmas_arr = timesteps_arr / self.T
        return timesteps_arr, sigmas_arr

    def step(
        self,
        noise: mx.array,
        timestep: int,
        latents: mx.array,
        **kwargs,
    ) -> mx.array:
        model_output = noise
        sample = latents
        timestep_idx = timestep
        t = self._timesteps[timestep_idx]
        s = self._timesteps[timestep_idx + 1]
        t_norm = t / self.T
        s_norm = s / self.T
        pred_x_0 = sample - t_norm * model_output
        pred_noise = sample + (1 - t_norm) * model_output
        if s > 0:
            next_sample = (1 - s_norm) * pred_x_0 + s_norm * pred_noise
        else:
            next_sample = pred_x_0
        return next_sample


SCHEDULER_REGISTRY: dict[str, type] = {
    "seedvr2_euler": SeedVR2EulerScheduler,
    "SeedVR2EulerScheduler": SeedVR2EulerScheduler,
}


def try_import_external_scheduler(scheduler_object_path: str) -> None:
    raise RuntimeError(
        f"External scheduler {scheduler_object_path!r} is not supported for SeedVR2 in DanQing."
    )


__all__ = [
    "BaseScheduler",
    "SCHEDULER_REGISTRY",
    "SeedVR2EulerScheduler",
    "try_import_external_scheduler",
]
