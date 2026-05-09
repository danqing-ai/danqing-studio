"""
调度器 — 所有扩散模型共用。

参考 mflux 项目的常见调度器和 mlx-video 的 UniPC/DPM++。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional


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
    """Flow Matching Euler 离散调度器。

    用于: Flux2 Klein / Z-Image / FIBO
    参考 mflux FlowMatchEulerDiscreteScheduler 实现。
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
        """计算去噪时间步序列。

        默认参考 mflux FlowMatchEulerDiscreteScheduler._compute_timesteps_and_sigmas
        (mu=1.0, shift_terminal=0.02)。当 use_empirical_mu=True 时参考 get_timesteps_and_sigmas。
        """
        self._init_timestep = init_timestep
        import mlx.core as mx

        if use_empirical_mu:
            # 参考 mflux get_timesteps_and_sigmas (用于 set_image_seq_len)
            sigmas = mx.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps, dtype=mx.float32)
            mu = self._compute_empirical_mu(image_seq_len, num_inference_steps)
            sigmas = self._time_shift_exponential_array(mu, 1.0, sigmas)
            timesteps = sigmas * self._num_train_timesteps
            sigmas = mx.concatenate([sigmas, mx.zeros((1,), dtype=sigmas.dtype)], axis=0)
        else:
            # 参考 mflux _compute_timesteps_and_sigmas (默认初始化)
            mu_val = kwargs.get('mu', 1.0)
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
            sigmas = mx.array(sigmas_with_zero, dtype=mx.float32)
            timesteps = mx.array(timesteps, dtype=mx.float32)

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

    @staticmethod
    def _time_shift_exponential_array(mu: float, sigma_power: float, t) -> Any:
        import mlx.core as mx
        return mx.exp(mu) / (mx.exp(mu) + ((1.0 / t - 1.0) ** sigma_power))

    def _stretch_to_terminal(self, sigmas: list[float]) -> list[float]:
        shift_terminal = 0.02
        one_minus_sigmas = [1.0 - s for s in sigmas]
        scale_factor = one_minus_sigmas[-1] / (1.0 - shift_terminal)
        stretched = [1.0 - (oms / scale_factor) for oms in one_minus_sigmas]
        return stretched

    def step(self, noise_pred: Any, timestep: Any, latents: Any,
             **kwargs) -> Any:
        ctx = self.ctx
        # Use passed timestep as index into sigmas (matching mflux)
        t_idx = int(timestep)
        if t_idx >= len(self._timesteps):
            return latents
        sigma = self._sigmas[t_idx]
        sigma_next = self._sigmas[t_idx + 1]
        # Euler: x_{t-1} = x_t + (sigma_{t-1} - sigma_t) * v
        dt = sigma_next - sigma
        return latents + dt * noise_pred


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
        import mlx.core as mx
        sigmas = mx.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps, dtype=mx.float32)

        # 与 mflux LinearScheduler._get_sigmas 一致
        if requires_sigma_shift:
            sigma_max_shift, sigma_base_shift = 1.15, 0.5
            sigma_max_seq_len, sigma_base_seq_len = 4096, 256
            sigma_shift_terminal = None  # z-image-turbo is None, some other models may set it
            m = (sigma_max_shift - sigma_base_shift) / (sigma_max_seq_len - sigma_base_seq_len)
            b = sigma_base_shift - m * sigma_base_seq_len
            mu = m * image_width * image_height / 256 + b
            mu = mx.array(mu)
            shifted = mx.exp(mu) / (mx.exp(mu) + (1 / sigmas - 1))
            if sigma_shift_terminal is not None:
                one_minus = 1.0 - shifted
                scale_val = one_minus[-1] / (1.0 - sigma_shift_terminal)
                shifted = 1.0 - (one_minus / scale_val)
            sigmas = mx.concatenate([shifted, mx.zeros((1,), dtype=mx.float32)], axis=0)
        else:
            sigmas = mx.concatenate([sigmas, mx.zeros((1,), dtype=mx.float32)], axis=0)

        self._sigmas = sigmas
        self._timesteps = mx.arange(num_inference_steps, dtype=mx.float32)
        self._step_index = init_timestep
        return list(range(init_timestep, num_inference_steps))
        return list(range(init_timestep, num_inference_steps))

    def step(self, noise_pred: Any, timestep: Any, latents: Any,
             **kwargs) -> Any:
        ctx = self.ctx
        # Use sigmas array indexed by timestep (matching mflux LinearScheduler)
        t_idx = int(timestep)
        sigma = self._sigmas[t_idx]
        sigma_prev = self._sigmas[t_idx + 1]
        dt = sigma_prev - sigma
        return latents + dt * noise_pred


class UniPCScheduler(Scheduler):
    """UniPC 高阶调度器 (Wan 默认)。

    参考 mlx-video wan_2 实现。
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
            import mlx.core as mx
            ctx = type('Ctx', (), {
                'arange': mx.arange, 'float32': mx.float32,
                'zeros': mx.zeros, 'concat': mx.concatenate,
            })()
        step_ratio = self._num_train_timesteps // num_inference_steps
        timesteps = (ctx.arange(0, num_inference_steps + 1, dtype=ctx.float32()) * step_ratio)[::-1]
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
            import mlx.core as mx
            ctx = type('Ctx', (), {
                'arange': mx.arange, 'float32': mx.float32,
                'zeros': mx.zeros, 'concat': mx.concatenate,
            })()
        step_ratio = self._num_train_timesteps // num_inference_steps
        timesteps = (ctx.arange(0, num_inference_steps + 1, dtype=ctx.float32()) * step_ratio)[::-1]
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

    @property
    def num_train_timesteps(self) -> int:
        return self._num_train_timesteps

    def set_timesteps(self, num_inference_steps: int = 1, **kwargs) -> Any:
        ctx = self.ctx
        if ctx is None:
            import mlx.core as mx
            ctx = type('Ctx', (), {'arange': mx.arange, 'float32': mx.float32})()

        denoise = kwargs.get("denoise", 0.3)
        t = ctx.arange(0, 1, dtype=ctx.float32()) * self._num_train_timesteps * denoise
        self._timesteps = t
        return t

    def step(self, noise_pred: Any, timestep: Any, latents: Any,
             **kwargs) -> Any:
        sigma = timestep / self._num_train_timesteps
        sigma_prev = 0.0
        return latents + (sigma_prev - sigma) * noise_pred


_scheduler_registry: dict[str, type[Scheduler]] = {
    "flow_match_euler": FlowMatchEulerScheduler,
    "linear": LinearScheduler,
    "unipc": UniPCScheduler,
    "dpm++": DPMPlusPlusScheduler,
    "seedvr2_euler": SeedVR2EulerScheduler,
    "euler": LinearScheduler,            # alias
    "flow_match_euler_discrete": FlowMatchEulerScheduler,  # alias
}


def get_scheduler(name: str, ctx: Any = None, **kwargs) -> Scheduler:
    """根据名称创建调度器实例。"""
    cls = _scheduler_registry.get(name.lower())
    if cls is None:
        raise KeyError(f"unknown scheduler: {name}. Available: {list(_scheduler_registry.keys())}")
    return cls(ctx=ctx, **kwargs)
