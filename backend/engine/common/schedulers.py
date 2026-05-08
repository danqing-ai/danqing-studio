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
                      **kwargs) -> Any:
        self._init_timestep = init_timestep
        ctx = self.ctx
        if ctx is None:
            import mlx.core as mx
            ctx = type('Ctx', (), {
                'arange': mx.arange, 'float32': mx.float32,
                'zeros': mx.zeros, 'concat': mx.concatenate,
                'linspace': mx.linspace,
            })()

        # 完全参考 mflux FlowMatchEulerDiscreteScheduler.get_timesteps_and_sigmas
        import mlx.core as mx
        sigmas = mx.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps, dtype=mx.float32)
        mu = self._compute_empirical_mu(image_seq_len, num_inference_steps)
        sigmas = self._time_shift_exponential_array(mu, 1.0, sigmas)
        timesteps = sigmas * self._num_train_timesteps
        sigmas = mx.concatenate([sigmas, mx.zeros(1)], axis=0)
        self._sigmas = sigmas
        self._timesteps = timesteps
        self._step_index = init_timestep
        # 返回整数索引序列 [init, init+1, ..., num_steps-1]（与 mflux 一致）
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
    def _time_shift_exponential_array(mu: float, sigma_power: float, t) -> Any:
        import mlx.core as mx
        return mx.exp(mu) / (mx.exp(mu) + ((1.0 / t - 1.0) ** sigma_power))

    def step(self, noise_pred: Any, timestep: Any, latents: Any,
             **kwargs) -> Any:
        ctx = self.ctx
        if self._step_index >= len(self._timesteps):
            return latents
        sigma = self._sigmas[self._step_index]
        sigma_next = self._sigmas[self._step_index + 1]
        self._step_index += 1
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

    def set_timesteps(self, num_inference_steps: int,
                      init_timestep: int = 0,
                      **kwargs) -> Any:
        self._init_timestep = init_timestep
        ctx = self.ctx
        if ctx is None:
            import mlx.core as mx
            ctx = type('Ctx', (), {
                'arange': mx.arange, 'float32': mx.float32,
                'zeros': mx.zeros, 'concat': mx.concatenate,
            })()
        step_ratio = max(1, self._num_train_timesteps // max(1, num_inference_steps))
        timesteps = (ctx.arange(0, num_inference_steps, dtype=ctx.float32()) * step_ratio)[::-1]
        self._timesteps = timesteps
        self._sigmas = ctx.concat([
            timesteps,
            ctx.zeros((1,), dtype=ctx.float32()),
        ], axis=0) / self._num_train_timesteps
        self._step_index = init_timestep
        return timesteps[init_timestep:]

    def step(self, noise_pred: Any, timestep: Any, latents: Any,
             **kwargs) -> Any:
        ctx = self.ctx
        sigma = timestep / self._num_train_timesteps
        sigma_prev = self._timesteps[min(self._step_index + 1, len(self._timesteps) - 1)] / self._num_train_timesteps if self._step_index < len(self._timesteps) - 1 else 0.0
        self._step_index += 1
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
