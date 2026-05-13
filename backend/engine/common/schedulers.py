"""
Schedulers — shared across all diffusion models.

Reference implementations: common schedulers and UniPC / DPM++ variants.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np


def flux_calculate_shift_mu(
    image_seq_len: int,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
) -> float:
    """Resolution-dependent μ for FlowMatchEulerDiscreteScheduler (same as Flux ``calculate_shift``)."""

    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return float(image_seq_len * m + b)


class Scheduler(ABC):
    """扩散调度器基类。"""

    @abstractmethod
    def set_timesteps(self, num_inference_steps: int, **kwargs) -> Any:
        """计算去噪时间步序列。"""
        ...

    @abstractmethod
    def step(self, noise_pred: Any, timestep: Any, latents: Any,
             **kwargs) -> Any:
        """单步去噪：latents → next_latents。"""
        ...

    @property
    @abstractmethod
    def num_train_timesteps(self) -> int:
        ...


class FlowMatchEulerScheduler(Scheduler):
    """Flow Matching Euler discrete scheduler.

    Used by: Flux2 Klein / Z-Image / FIBO
    Reference: FlowMatchEulerDiscreteScheduler implementation.
    """

    def __init__(self, num_train_timesteps: int = 1000,
                 shift: float = 1.0,
                 ctx: Any = None):
        self._num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.ctx = ctx
        self._timesteps: Any = None
        self._sigmas: Any = None
        self._init_timestep: int = 0
        self._step_index: int = 0

    @property
    def num_train_timesteps(self) -> int:
        return self._num_train_timesteps

    @property
    def timesteps(self) -> Any:
        return self._timesteps

    @property
    def sigmas(self) -> Any:
        return self._sigmas

    def set_timesteps(self, num_inference_steps: int,
                      init_timestep: int = 0,
                      image_seq_len: int = 256,
                      use_empirical_mu: bool = True,
                      **kwargs) -> Any:
        """Compute denoising timestep sequence.

        Default reference: FlowMatchEulerDiscreteScheduler._compute_timesteps_and_sigmas
        (mu=1.0, shift_terminal=0.02). When use_empirical_mu=True uses get_timesteps_and_sigmas.
        """
        self._init_timestep = init_timestep

        if use_empirical_mu:
            # Reference get_timesteps_and_sigmas (for set_image_seq_len)
            sigmas = self.ctx.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps, dtype=self.ctx.float32())
            mu = self._compute_empirical_mu(image_seq_len, num_inference_steps)
            sigmas = self._time_shift_exponential_array(mu, 1.0, sigmas)
            timesteps = sigmas * self._num_train_timesteps
            sigmas = self.ctx.concat([sigmas, self.ctx.zeros((1,), dtype=sigmas.dtype)], axis=0)
        else:
            # Reference _compute_timesteps_and_sigmas (default initialization)
            mu_val = kwargs.get("mu", 1.0)
            if mu_val is None:
                mu_val = 1.0
            sigma_min = 1.0 / self._num_train_timesteps
            sigma_max = 1.0
            timesteps_linear = [
                sigma_max * self._num_train_timesteps
                - i * (sigma_max - sigma_min) * self._num_train_timesteps / (num_inference_steps - 1)
                for i in range(num_inference_steps)
            ]
            sigmas_linear = [t / self._num_train_timesteps for t in timesteps_linear]
            sigmas_shifted = [self._time_shift_exponential(mu_val, 1.0, s) for s in sigmas_linear]
            sigmas_final = self._stretch_to_terminal(sigmas_shifted)
            timesteps = [s * self._num_train_timesteps for s in sigmas_final]
            sigmas_with_zero = sigmas_final + [0.0]
            sigmas = self.ctx.array(sigmas_with_zero, dtype=self.ctx.float32())
            timesteps = self.ctx.array(timesteps, dtype=self.ctx.float32())

        self._sigmas = sigmas
        self._timesteps = timesteps
        self._step_index = init_timestep
        return list(range(init_timestep, num_inference_steps))

    @staticmethod
    def _compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
        a1, b1 = 8.73809524e-05, 1.89833333
        a2, b2 = 0.00016927, 0.45666666
        if image_seq_len > 4300:
            return float(a2 * image_seq_len + b2)
        m_200 = a2 * image_seq_len + b2
        m_10 = a1 * image_seq_len + b1
        a = (m_200 - m_10) / 190.0
        b = m_200 - 200.0 * a
        return float(a * num_steps + b)

    @staticmethod
    def _time_shift_exponential(mu: float, sigma_power: float, t: float) -> float:
        import math
        return math.exp(mu) / (math.exp(mu) + ((1.0 / t - 1.0) ** sigma_power))

    def _time_shift_exponential_array(self, mu: float, sigma_power: float, t) -> Any:
        """Vectorized ``_time_shift_exponential`` on ``t`` using :class:`RuntimeContext` (MLX/CUDA)."""
        ctx = self.ctx
        if ctx is None:
            raise RuntimeError("FlowMatchEulerScheduler requires RuntimeContext (pass ctx= from pipeline)")
        ones = ctx.ones_like(t)
        exp_mu = ctx.exp(ones * mu)
        inner = ctx.div(1.0, t) - 1.0
        pow_inner = inner ** sigma_power
        return ctx.div(exp_mu, exp_mu + pow_inner)

    def _stretch_to_terminal(self, sigmas: list[float]) -> list[float]:
        shift_terminal = 0.02
        one_minus_sigmas = [1.0 - s for s in sigmas]
        scale_factor = one_minus_sigmas[-1] / (1.0 - shift_terminal)
        stretched = [1.0 - (oms / scale_factor) for oms in one_minus_sigmas]
        return stretched

    def step(self, noise_pred: Any, timestep: Any, latents: Any,
             **kwargs) -> Any:
        ctx = self.ctx
        # Use passed timestep as index into sigmas (matching reference impl)
        t_idx = int(timestep)
        if t_idx >= len(self._timesteps):
            return latents
        sigma = self._sigmas[t_idx]
        sigma_next = self._sigmas[t_idx + 1]
        # Euler: x_{t-1} = x_t + (sigma_{t-1} - sigma_t) * v
        dt = sigma_next - sigma
        return latents + dt * noise_pred


class FlowMatchEulerFluxDynamicScheduler(FlowMatchEulerScheduler):
    """Flow Match Euler + Flux ``calculate_shift`` μ + diffusers-style dynamic σ schedule.

    Intended for LongCat-Image (registry ``scheduler``: ``flow_match_euler_flux_dynamic``); **does not alter**
    :class:`FlowMatchEulerScheduler` behavior used by Z-Image / FIBO / etc.

    μ: optional ``mu`` in kwargs; otherwise computed from ``image_seq_len`` and
    ``scheduler_*`` kwargs (defaults match typical ``scheduler_config.json``).
    """

    def set_timesteps(self, num_inference_steps: int,
                      init_timestep: int = 0,
                      image_seq_len: int = 256,
                      use_empirical_mu: bool = True,
                      **kwargs) -> Any:
        del use_empirical_mu  # fixed schedule for this class
        self._init_timestep = init_timestep
        mu = kwargs.get("mu")
        if mu is None:
            mu = flux_calculate_shift_mu(
                image_seq_len,
                int(kwargs.get("scheduler_base_image_seq_len", 256)),
                int(kwargs.get("scheduler_max_image_seq_len", 4096)),
                float(kwargs.get("scheduler_base_shift", 0.5)),
                float(kwargs.get("scheduler_max_shift", 1.15)),
            )
        num_train = float(self._num_train_timesteps)
        raw_sigmas = kwargs.get("sigmas")
        if raw_sigmas is not None:
            sigmas_np = np.asarray(raw_sigmas, dtype=np.float32).reshape(-1)
            if sigmas_np.size != num_inference_steps:
                raise RuntimeError(
                    "FlowMatchEulerFluxDynamicScheduler: len(sigmas) must equal num_inference_steps "
                    f"({sigmas_np.size} != {num_inference_steps})"
                )
            sigmas_t = self.ctx.array(sigmas_np)
        else:
            sigma_max = 1.0
            sigma_min = 1.0 / num_train
            timesteps_np = np.linspace(
                sigma_max * num_train,
                sigma_min * num_train,
                num_inference_steps,
                dtype=np.float32,
            )
            sigmas_np = timesteps_np / num_train
            sigmas_t = self.ctx.array(sigmas_np)
        sigmas_shifted = self._time_shift_exponential_array(float(mu), 1.0, sigmas_t)
        sigmas = self.ctx.concat([sigmas_shifted, self.ctx.zeros((1,), dtype=sigmas_shifted.dtype)], axis=0)
        timesteps = sigmas_shifted * num_train
        self._sigmas = sigmas
        self._timesteps = timesteps
        self._step_index = init_timestep
        return list(range(init_timestep, num_inference_steps))


class LinearScheduler(Scheduler):
    """标准线性调度器 (Flux1 / Qwen-Image)。"""

    def __init__(self, num_train_timesteps: int = 1000,
                 ctx: Any = None):
        self._num_train_timesteps = num_train_timesteps
        self.ctx = ctx
        self._timesteps: Any = None
        self._init_timestep: int = 0
        self._step_index: int = 0

    @property
    def num_train_timesteps(self) -> int:
        return self._num_train_timesteps

    @property
    def sigmas(self) -> Any:
        return self._sigmas

    def set_timesteps(self, num_inference_steps: int,
                      init_timestep: int = 0,
                      image_seq_len: int = 256,
                      image_width: int = 256,
                      image_height: int = 256,
                      requires_sigma_shift: bool = False,
                      **kwargs) -> Any:
        self._init_timestep = init_timestep
        sigmas = self.ctx.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps, dtype=self.ctx.float32())

        # Matches LinearScheduler._get_sigmas
        if requires_sigma_shift:
            sigma_max_shift, sigma_base_shift = 1.15, 0.5
            sigma_max_seq_len, sigma_base_seq_len = 4096, 256
            sigma_shift_terminal = None  # z-image-turbo is None, some other models may set it
            m = (sigma_max_shift - sigma_base_shift) / (sigma_max_seq_len - sigma_base_seq_len)
            b = sigma_base_shift - m * sigma_base_seq_len
            mu = m * image_width * image_height / 256 + b
            mu = self.ctx.array(mu)
            shifted = self.ctx.exp(mu) / (self.ctx.exp(mu) + (1 / sigmas - 1))
            if sigma_shift_terminal is not None:
                one_minus = 1.0 - shifted
                scale_val = one_minus[-1] / (1.0 - sigma_shift_terminal)
                shifted = 1.0 - (one_minus / scale_val)
            sigmas = self.ctx.concat([shifted, self.ctx.zeros((1,), dtype=self.ctx.float32())], axis=0)
        else:
            sigmas = self.ctx.concat([sigmas, self.ctx.zeros((1,), dtype=self.ctx.float32())], axis=0)

        self._sigmas = sigmas
        self._timesteps = self.ctx.arange(num_inference_steps, dtype=self.ctx.float32())
        self._step_index = init_timestep
        return list(range(init_timestep, num_inference_steps))

    def step(self, noise_pred: Any, timestep: Any, latents: Any,
             **kwargs) -> Any:
        # Use sigmas array indexed by timestep (matching reference LinearScheduler)
        t_idx = int(timestep)
        sigma = self._sigmas[t_idx]
        sigma_prev = self._sigmas[t_idx + 1]
        dt = sigma_prev - sigma
        # mflux ``LinearScheduler.step``: cast dt and noise_pred to ``latents.dtype`` before add-mul.
        if hasattr(dt, "astype") and hasattr(latents, "astype") and hasattr(noise_pred, "astype"):
            dt = dt.astype(latents.dtype)
            return latents + noise_pred.astype(latents.dtype) * dt
        return latents + dt * noise_pred


class UniPCScheduler(Scheduler):
    """UniPC 高阶调度器 (Wan 默认)。

    与常见 Wan 2 UniPC 描述对齐。
    """

    def __init__(self, num_train_timesteps: int = 1000,
                 ctx: Any = None):
        self._num_train_timesteps = num_train_timesteps
        self.ctx = ctx
        self._timesteps: Any = None
        self._sigmas: Any = None
        self._step_index: int = 0
        self._noise_pred_history: list = []

    @property
    def num_train_timesteps(self) -> int:
        return self._num_train_timesteps

    def set_timesteps(self, num_inference_steps: int, **kwargs) -> Any:
        ctx = self.ctx
        if ctx is None:
            raise RuntimeError("UniPCScheduler requires RuntimeContext (pass ctx= from pipeline)")
        step_ratio = self._num_train_timesteps // num_inference_steps
        timesteps = ctx.flip(
            ctx.arange(0, num_inference_steps + 1, dtype=ctx.float32()) * step_ratio,
            axis=0,
        )
        self._timesteps = timesteps
        self._sigmas = timesteps / self._num_train_timesteps
        self._step_index = 0
        self._noise_pred_history = []
        return timesteps[:-1]

    def step(self, noise_pred: Any, timestep: Any, latents: Any,
             **kwargs) -> Any:
        ctx = self.ctx
        sigma = self._sigmas[self._step_index]
        sigma_next = self._sigmas[self._step_index + 1]
        self._step_index += 1

        self._noise_pred_history.append(noise_pred)
        if len(self._noise_pred_history) >= 2:
            prev_pred = self._noise_pred_history[-2]
            dt = sigma_next - sigma
            # 二阶修正
            correction = 0.5 * (noise_pred - prev_pred) * (sigma_next - self._sigmas[self._step_index - 2]) / (sigma - self._sigmas[self._step_index - 2])
            return latents + dt * (noise_pred + correction)

        dt = sigma_next - sigma
        return latents + dt * noise_pred


class DPMPlusPlusScheduler(Scheduler):
    """DPM++ 2M 调度器 (CogVideoX 默认, Wan 可选)。"""

    def __init__(self, num_train_timesteps: int = 1000,
                 ctx: Any = None):
        self._num_train_timesteps = num_train_timesteps
        self.ctx = ctx
        self._timesteps: Any = None
        self._sigmas: Any = None
        self._step_index: int = 0
        self._prev_pred: Any = None

    @property
    def num_train_timesteps(self) -> int:
        return self._num_train_timesteps

    def set_timesteps(self, num_inference_steps: int, **kwargs) -> Any:
        ctx = self.ctx
        if ctx is None:
            raise RuntimeError("DPMPlusPlusScheduler requires RuntimeContext (pass ctx= from pipeline)")
        step_ratio = self._num_train_timesteps // num_inference_steps
        timesteps = ctx.flip(
            ctx.arange(0, num_inference_steps + 1, dtype=ctx.float32()) * step_ratio,
            axis=0,
        )
        self._timesteps = timesteps
        self._sigmas = timesteps / self._num_train_timesteps
        self._step_index = 0
        self._prev_pred = None
        return timesteps[:-1]

    def step(self, noise_pred: Any, timestep: Any, latents: Any,
             **kwargs) -> Any:
        sigma = self._sigmas[self._step_index]
        sigma_next = self._sigmas[self._step_index + 1]
        self._step_index += 1

        if self._prev_pred is not None:
            h = sigma_next - sigma
            h_last = sigma - self._sigmas[self._step_index - 2]
            r = h / h_last
            d = noise_pred + r * (noise_pred - self._prev_pred) / 2
            result = latents + h * d
        else:
            result = latents + (sigma_next - sigma) * noise_pred

        self._prev_pred = noise_pred
        return result


class SeedVR2EulerScheduler(Scheduler):
    """SeedVR2 1-step Euler 调度器。

    单步去噪，无需多步循环。
    """

    def __init__(self, num_train_timesteps: int = 1000,
                 ctx: Any = None):
        self._num_train_timesteps = num_train_timesteps
        self.ctx = ctx
        self._timesteps: Any = None

    @property
    def num_train_timesteps(self) -> int:
        return self._num_train_timesteps

    @property
    def timesteps(self) -> Any:
        return self._timesteps

    def set_timesteps(self, num_inference_steps: int = 1, **kwargs) -> Any:
        ctx = self.ctx
        if ctx is None:
            raise RuntimeError("SeedVR2EulerScheduler requires RuntimeContext (pass ctx= from pipeline)")

        denoise = float(kwargs.get("denoise", 0.3))
        denoise = max(1e-6, min(1.0, denoise))
        # Single-step SR: one finite timestep so sigma != 0 and Euler mixes noise + model output.
        t_max = float(self._num_train_timesteps) * denoise
        self._timesteps = ctx.array([t_max], dtype=ctx.float32())
        return self._timesteps

    def step(self, noise_pred: Any, timestep: Any, latents: Any,
             **kwargs) -> Any:
        sigma = timestep / self._num_train_timesteps
        sigma_prev = 0.0
        return latents + (sigma_prev - sigma) * noise_pred


_scheduler_registry: dict[str, type[Scheduler]] = {
    "flow_match_euler": FlowMatchEulerScheduler,
    "flow_match_euler_flux_dynamic": FlowMatchEulerFluxDynamicScheduler,
    "linear": LinearScheduler,
    "unipc": UniPCScheduler,
    "dpm++": DPMPlusPlusScheduler,
    "seedvr2_euler": SeedVR2EulerScheduler,
    "euler": LinearScheduler,            # alias
    "flow_match_euler_discrete": FlowMatchEulerScheduler,  # alias
    "flow_match_euler_dynamic": FlowMatchEulerFluxDynamicScheduler,  # alias (LongCat)
}


def get_scheduler(name: str, ctx: Any = None, **kwargs) -> Scheduler:
    """根据名称创建调度器实例。"""
    cls = _scheduler_registry.get(name.lower())
    if cls is None:
        raise KeyError(f"unknown scheduler: {name}. Available: {list(_scheduler_registry.keys())}")
    return cls(ctx=ctx, **kwargs)
