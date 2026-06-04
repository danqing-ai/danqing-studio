"""
Schedulers — shared across all diffusion models.

Reference implementations: common schedulers and UniPC / DPM++ variants.
"""
from __future__ import annotations

import math
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


def _betas_scaled_linear(beta_start: float, beta_end: float, n: int) -> np.ndarray:
    return np.linspace(beta_start**0.5, beta_end**0.5, n, dtype=np.float64) ** 2


def _rescale_zero_terminal_snr(alphas_cumprod: np.ndarray) -> np.ndarray:
    alphas_bar_sqrt = np.sqrt(alphas_cumprod)
    alphas_bar_sqrt_0 = float(alphas_bar_sqrt[0])
    alphas_bar_sqrt_T = float(alphas_bar_sqrt[-1])
    alphas_bar_sqrt = alphas_bar_sqrt - alphas_bar_sqrt_T
    alphas_bar_sqrt = alphas_bar_sqrt * (alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T))
    return alphas_bar_sqrt ** 2


class CogVideoXDPMScheduler(Scheduler):
    """CogVideoX DPM-Solver++ — matches diffusers ``CogVideoXDPMScheduler``."""

    order = 1
    is_cogvideox_dpm = True

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        ctx: Any = None,
        *,
        beta_start: float = 0.00085,
        beta_end: float = 0.012,
        beta_schedule: str = "scaled_linear",
        prediction_type: str = "epsilon",
        timestep_spacing: str = "leading",
        steps_offset: int = 0,
        snr_shift_scale: float = 3.0,
        set_alpha_to_one: bool = True,
        rescale_betas_zero_snr: bool = False,
    ):
        self._num_train_timesteps = num_train_timesteps
        self.ctx = ctx
        if beta_schedule == "scaled_linear":
            betas = _betas_scaled_linear(beta_start, beta_end, num_train_timesteps)
        elif beta_schedule == "linear":
            betas = np.linspace(beta_start, beta_end, num_train_timesteps, dtype=np.float64)
        else:
            raise NotImplementedError(f"CogVideoXDPMScheduler beta_schedule {beta_schedule!r} not supported")

        alphas_cumprod = np.cumprod(1.0 - betas)
        alphas_cumprod = alphas_cumprod / (
            snr_shift_scale + (1.0 - snr_shift_scale) * alphas_cumprod
        )
        if rescale_betas_zero_snr:
            alphas_cumprod = _rescale_zero_terminal_snr(alphas_cumprod)

        self._alphas_cumprod = alphas_cumprod.astype(np.float64)
        self._final_alpha_cumprod = 1.0 if set_alpha_to_one else float(self._alphas_cumprod[0])
        self._prediction_type = prediction_type
        self._timestep_spacing = timestep_spacing
        self._steps_offset = int(steps_offset)
        self.init_noise_sigma = 1.0
        self._timesteps: Any = None
        self._timesteps_list: list[int] = []
        self._num_inference_steps = 0

    @property
    def num_train_timesteps(self) -> int:
        return self._num_train_timesteps

    @property
    def timesteps(self) -> Any:
        return self._timesteps

    def _alpha_at(self, t: int) -> float:
        return float(self._alphas_cumprod[int(t)])

    @staticmethod
    def _as_int_timestep(timestep: Any) -> int:
        if isinstance(timestep, (int, np.integer)):
            return int(timestep)
        try:
            return int(np.asarray(timestep).item())
        except (ValueError, TypeError):
            return int(float(timestep))

    def set_timesteps(self, num_inference_steps: int, **kwargs) -> Any:
        ctx = self.ctx
        if ctx is None:
            raise RuntimeError("CogVideoXDPMScheduler requires RuntimeContext (pass ctx= from pipeline)")
        if num_inference_steps > self._num_train_timesteps:
            raise ValueError(
                f"num_inference_steps={num_inference_steps} exceeds "
                f"num_train_timesteps={self._num_train_timesteps}"
            )

        self._num_inference_steps = int(num_inference_steps)
        n = self._num_inference_steps
        if self._timestep_spacing == "linspace":
            ts = np.linspace(0, self._num_train_timesteps - 1, n).round()[::-1].astype(np.int64)
        elif self._timestep_spacing == "leading":
            step_ratio = self._num_train_timesteps // n
            ts = (np.arange(0, n, dtype=np.float64) * step_ratio).round()[::-1].astype(np.int64)
            ts += self._steps_offset
        elif self._timestep_spacing == "trailing":
            step_ratio = self._num_train_timesteps / n
            ts = np.round(np.arange(self._num_train_timesteps, 0, -step_ratio)).astype(np.int64) - 1
        else:
            raise ValueError(f"unsupported timestep_spacing {self._timestep_spacing!r}")

        self._timesteps_list = [int(x) for x in ts.tolist()]
        self._timesteps = ctx.array(self._timesteps_list, dtype=ctx.int32())
        return self._timesteps

    def scale_model_input(self, sample: Any, timestep: Any = None) -> Any:
        return sample

    @staticmethod
    def _lambda_from_alpha(alpha: float) -> float:
        """Match PyTorch ``log((alpha/(1-alpha))**0.5)``: ``alpha→1`` yields ``+inf``, not div-by-zero."""
        a = float(alpha)
        one_minus = 1.0 - a
        if one_minus <= 0.0:
            return float("inf")
        ratio = a / one_minus
        if ratio <= 0.0:
            return float("-inf")
        return math.log(ratio ** 0.5)

    @staticmethod
    def _sqrt_ratio(num: float, den: float) -> float:
        if den <= 0.0:
            return float("inf") if num > 0.0 else 0.0
        return (num / den) ** 0.5

    def _get_variables(
        self,
        alpha_prod_t: float,
        alpha_prod_t_prev: float,
        alpha_prod_t_back: float | None,
    ) -> tuple[float, float | None, float, float]:
        lamb = self._lambda_from_alpha(alpha_prod_t)
        lamb_next = self._lambda_from_alpha(alpha_prod_t_prev)
        h = lamb_next - lamb
        if alpha_prod_t_back is not None:
            lamb_previous = self._lambda_from_alpha(alpha_prod_t_back)
            h_last = lamb - lamb_previous
            if h == 0.0 or not math.isfinite(h):
                r = 0.0
            else:
                r = h_last / h
            return h, r, lamb, lamb_next
        return h, None, lamb, lamb_next

    def _get_mult(
        self,
        h: float,
        r: float | None,
        alpha_prod_t: float,
        alpha_prod_t_prev: float,
        alpha_prod_t_back: float | None,
    ) -> tuple[float, ...]:
        mult1 = (
            self._sqrt_ratio(1.0 - alpha_prod_t_prev, 1.0 - alpha_prod_t)
            * math.exp(-h)
        )
        mult2 = math.expm1(-2.0 * h) * (max(alpha_prod_t_prev, 0.0) ** 0.5)
        if alpha_prod_t_back is not None and r not in (None, 0.0) and math.isfinite(r):
            mult3 = 1.0 + 1.0 / (2.0 * r)
            mult4 = 1.0 / (2.0 * r)
            return mult1, mult2, mult3, mult4
        return mult1, mult2

    def step(
        self,
        noise_pred: Any,
        timestep: Any,
        latents: Any,
        *,
        old_pred_original_sample: Any | None = None,
        timestep_back: Any | None = None,
        **_: Any,
    ) -> tuple[Any, Any]:
        ctx = self.ctx
        if ctx is None:
            raise RuntimeError("CogVideoXDPMScheduler requires RuntimeContext")
        if self._num_inference_steps <= 0:
            raise RuntimeError("CogVideoXDPMScheduler.set_timesteps must run before step()")

        t = self._as_int_timestep(timestep)
        prev_timestep = t - self._num_train_timesteps // self._num_inference_steps

        alpha_prod_t = self._alpha_at(t)
        alpha_prod_t_prev = (
            self._alpha_at(prev_timestep) if prev_timestep >= 0 else float(self._final_alpha_cumprod)
        )
        alpha_prod_t_back = (
            self._alpha_at(self._as_int_timestep(timestep_back))
            if timestep_back is not None
            else None
        )
        beta_prod_t = 1.0 - alpha_prod_t
        alpha_sqrt = max(alpha_prod_t, 1e-12) ** 0.5

        if self._prediction_type == "epsilon":
            pred_original_sample = (
                latents - (beta_prod_t ** 0.5) * noise_pred
            ) / alpha_sqrt
        elif self._prediction_type == "sample":
            pred_original_sample = noise_pred
        elif self._prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t ** 0.5) * latents - (beta_prod_t ** 0.5) * noise_pred
        else:
            raise ValueError(f"unsupported prediction_type {self._prediction_type!r}")

        h, r, _, _ = self._get_variables(alpha_prod_t, alpha_prod_t_prev, alpha_prod_t_back)
        mult = self._get_mult(h, r, alpha_prod_t, alpha_prod_t_prev, alpha_prod_t_back)
        if math.isfinite(h):
            mult_noise = (1.0 - alpha_prod_t_prev) ** 0.5 * max(0.0, 1.0 - math.exp(-2.0 * h)) ** 0.5
        else:
            mult_noise = 0.0

        noise = ctx.randn(latents.shape, dtype=latents.dtype)
        prev_sample = mult[0] * latents - mult[1] * pred_original_sample + mult_noise * noise

        if old_pred_original_sample is None or prev_timestep < 0 or len(mult) < 4:
            return prev_sample, pred_original_sample

        denoised_d = mult[2] * pred_original_sample - mult[3] * old_pred_original_sample
        noise2 = ctx.randn(latents.shape, dtype=latents.dtype)
        prev_sample = mult[0] * latents - mult[1] * denoised_d + mult_noise * noise2
        return prev_sample, pred_original_sample


class WanFlowUniPCScheduler(Scheduler):
    """Wan 2.x flow-matching UniPC (ported from ``wan/utils/fm_solvers_unipc.py``)."""

    order = 1

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        ctx: Any = None,
        *,
        shift: float = 1.0,
        solver_order: int = 2,
        solver_type: str = "bh2",
        predict_x0: bool = True,
    ):
        self._num_train_timesteps = num_train_timesteps
        self.ctx = ctx
        self._default_shift = float(shift)
        self._solver_order = int(solver_order)
        self._solver_type = solver_type
        self._predict_x0 = bool(predict_x0)
        self._timesteps: Any = None
        self._sigmas: np.ndarray | None = None
        self._step_index: int = 0
        self._model_outputs: list[Any | None] = []
        self._timestep_list: list[Any | None] = []
        self._lower_order_nums: int = 0
        self._last_sample: Any | None = None
        self._this_order: int = 1
        # Match official FlowUniPCMultistepScheduler: training-curve endpoints at shift=1.
        alphas = np.linspace(1, 1 / self._num_train_timesteps, self._num_train_timesteps)[::-1].copy()
        train_sigmas = 1.0 - alphas
        init_shift = 1.0
        train_sigmas = init_shift * train_sigmas / (1.0 + (init_shift - 1.0) * train_sigmas)
        self._sigma_max = float(train_sigmas[0])
        self._sigma_min = float(train_sigmas[-1])

    @property
    def num_train_timesteps(self) -> int:
        return self._num_train_timesteps

    @property
    def timesteps(self) -> Any:
        return self._timesteps

    @property
    def sigmas(self) -> Any:
        return self._sigmas

    def set_timesteps(self, num_inference_steps: int, **kwargs) -> Any:
        ctx = self.ctx
        if ctx is None:
            raise RuntimeError("WanFlowUniPCScheduler requires RuntimeContext (pass ctx= from pipeline)")

        shift = float(kwargs.get("shift", self._default_shift))
        sched_sigmas = np.linspace(self._sigma_max, self._sigma_min, num_inference_steps + 1)[:-1]
        sched_sigmas = shift * sched_sigmas / (1.0 + (shift - 1.0) * sched_sigmas)
        # Match mlx-video / official Wan: integer training timesteps (not float sigma*1000).
        timesteps = (
            sched_sigmas * self._num_train_timesteps
        ).astype(np.int64).astype(np.float32)
        sigmas_full = np.concatenate([sched_sigmas, [0.0]]).astype(np.float32)

        self._sigmas = sigmas_full
        self._timesteps = ctx.array(timesteps.astype(np.float32))
        self._step_index = 0
        self._model_outputs = [None] * self._solver_order
        self._timestep_list = [None] * self._solver_order
        self._lower_order_nums = 0
        self._last_sample = None
        return self._timesteps

    def _sigma_to_alpha_sigma_t(self, sigma: float) -> tuple[float, float]:
        return 1.0 - sigma, sigma

    def convert_model_output(self, model_output: Any, sample: Any) -> Any:
        ctx = self.ctx
        sigma = float(self._sigmas[self._step_index])
        if self._predict_x0:
            return sample - sigma * model_output
        return model_output

    def step(self, noise_pred: Any, timestep: Any, latents: Any, **kwargs) -> Any:
        ctx = self.ctx
        if self._sigmas is None:
            raise RuntimeError("WanFlowUniPCScheduler.set_timesteps must be called before step")

        converted = self.convert_model_output(noise_pred, latents)

        if (
            self._step_index > 0
            and self._last_sample is not None
            and self._model_outputs[-1] is not None
        ):
            latents = self._corrector_step(converted, latents)

        for i in range(self._solver_order - 1):
            self._model_outputs[i] = self._model_outputs[i + 1]
            self._timestep_list[i] = self._timestep_list[i + 1]
        self._model_outputs[-1] = converted
        self._timestep_list[-1] = timestep

        remaining = len(self._timesteps) - self._step_index if self._timesteps is not None else 1
        self._this_order = min(self._solver_order, max(1, int(remaining)))
        self._this_order = min(self._this_order, self._lower_order_nums + 1)

        self._last_sample = latents
        prev = self._predictor_step(latents, order=self._this_order)
        if self._lower_order_nums < self._solver_order:
            self._lower_order_nums += 1
        self._step_index += 1
        return prev

    @staticmethod
    def _flow_log_lambda(alpha: float, sigma: float) -> float:
        return float(
            np.log(max(alpha, 1e-6)) - np.log(max(sigma, 1e-6))
        )

    def _predictor_step(self, sample: Any, *, order: int) -> Any:
        """UniP-B(h) predictor (``wan/utils/fm_solvers_unipc.py`` ``multistep_uni_p_bh_update``)."""
        ctx = self.ctx
        sigma_t = float(self._sigmas[self._step_index + 1])
        sigma_s0 = float(self._sigmas[self._step_index])
        m0 = self._model_outputs[-1]
        if m0 is None:
            raise RuntimeError("WanFlowUniPCScheduler: model output history missing for predictor")
        if sigma_t <= 1e-6:
            return m0
        alpha_t, sigma_t_v = self._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s0, sigma_s0_v = self._sigma_to_alpha_sigma_t(sigma_s0)
        sigma_t_l = max(sigma_t_v, 1e-6)
        sigma_s0_l = max(sigma_s0_v, 1e-6)
        lambda_t = self._flow_log_lambda(alpha_t, sigma_t_l)
        lambda_s0 = self._flow_log_lambda(alpha_s0, sigma_s0_l)
        h = lambda_t - lambda_s0
        hh = -h
        h_phi_1 = np.expm1(hh)
        b_h = np.expm1(hh) if self._solver_type == "bh2" else hh
        x_t_ = (sigma_t / sigma_s0_l) * sample - alpha_t * h_phi_1 * m0

        pred_res = None
        use_order = max(1, int(order))
        if use_order >= 2 and self._step_index >= 1:
            mi = self._model_outputs[-2]
            if mi is not None:
                si = self._step_index - 1
                alpha_si, sigma_si_v = self._sigma_to_alpha_sigma_t(float(self._sigmas[si]))
                lambda_si = self._flow_log_lambda(alpha_si, max(sigma_si_v, 1e-6))
                rk = (lambda_si - lambda_s0) / h if abs(h) > 1e-12 else 1.0
                d1 = (mi - m0) / rk
                # Official UniPC order-2 uses ``rhos_p = [0.5]`` (single history slope).
                pred_res = 0.5 * d1
        if pred_res is None:
            return x_t_
        return x_t_ - alpha_t * b_h * pred_res

    def _corrector_step(self, model_t: Any, sample: Any) -> Any:
        ctx = self.ctx
        sigma_t = float(self._sigmas[self._step_index])
        sigma_s0 = float(self._sigmas[self._step_index - 1])
        m0 = self._model_outputs[-1]
        if sigma_t <= 1e-6:
            return m0
        alpha_t, _ = self._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s0, sigma_s0_v = self._sigma_to_alpha_sigma_t(sigma_s0)
        sigma_t_l = max(sigma_t, 1e-6)
        sigma_s0_l = max(sigma_s0_v, 1e-6)
        h = (
            np.log(max(alpha_t, 1e-6)) - np.log(sigma_t_l)
            - (np.log(max(alpha_s0, 1e-6)) - np.log(sigma_s0_l))
        )
        hh = -h
        h_phi_1 = np.expm1(hh)
        b_h = np.expm1(hh) if self._solver_type == "bh2" else hh
        x_t_ = (sigma_t / sigma_s0_l) * self._last_sample - alpha_t * h_phi_1 * m0
        d1_t = model_t - m0
        return x_t_ - alpha_t * b_h * 0.5 * d1_t


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
    "cogvideox_dpm": CogVideoXDPMScheduler,
    "wan_flow_unipc": WanFlowUniPCScheduler,
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
