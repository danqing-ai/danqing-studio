"""FlowMatchingInference — 通用 Euler ODE/SDE 积分循环 (L2)。

服务于 ACE-Step DiT (8-step turbo) 和 DiffRhythm 内层 CFM (per-block)。
Family-specific 行为通过 ``AudioInferenceBundle.flow`` 注入。
"""
from __future__ import annotations

import time
from typing import Any

from backend.engine.inference._protocols import AudioInferenceBundle
from backend.engine.inference._runtime import raise_if_cancelled
from backend.engine.inference.memory_guard import MemoryGuard


class FlowMatchingInference:
    """通用 flow-matching 推理策略 — N-step Euler 积分。"""

    def run(self, bundle: AudioInferenceBundle) -> dict[str, Any]:
        """执行 flow-matching 积分，返回 ``{"latents": ..., "time_costs": {...}}``."""
        flow = bundle.flow
        schedule = flow.timestep_schedule
        if not schedule:
            raise RuntimeError("FlowMatchingInference requires flow.timestep_schedule")

        model_forward = bundle.model_forward
        eval_fn = bundle.eval_fn or (lambda *_: None)
        guard = bundle.memory_guard or (
            MemoryGuard(bundle.ctx) if bundle.ctx is not None else None
        )
        on_step = bundle.on_step_complete
        num_steps = len(schedule)

        if flow.init_noise_fn is not None:
            xt = flow.init_noise_fn(bundle.latent_shape, bundle.seed)
        elif bundle.ctx is not None:
            xt = bundle.ctx.randn(bundle.latent_shape)
        else:
            raise RuntimeError(
                "FlowMatchingInference requires flow.init_noise_fn or bundle.ctx"
            )

        state: dict[str, Any] = {}
        if flow.cache_init_fn is not None:
            state["cache"] = flow.cache_init_fn()

        t0 = time.time()
        for step_idx in range(num_steps):
            raise_if_cancelled(bundle.cancel_token)

            if flow.before_step_fn is not None:
                flow.before_step_fn(step_idx, state)

            t_curr = schedule[step_idx]
            velocity = model_forward(xt, t_curr, state)
            eval_fn(velocity)
            if guard is not None:
                guard.step(velocity)

            if step_idx == num_steps - 1:
                if flow.euler_step_fn is not None:
                    xt = flow.euler_step_fn(xt, velocity, t_curr, None, step_idx)
                else:
                    xt = xt - velocity * t_curr
                eval_fn(xt)
                if guard is not None:
                    guard.step(xt)
                break

            t_next = schedule[step_idx + 1]
            if flow.euler_step_fn is not None:
                xt = flow.euler_step_fn(xt, velocity, t_curr, t_next, step_idx)
            else:
                dt = t_curr - t_next
                xt = xt - velocity * dt
            eval_fn(xt)
            if guard is not None:
                guard.step(xt)

            if on_step is not None:
                on_step(step_idx, xt, velocity)

        diffusion_time = time.time() - t0
        return {
            "latents": xt,
            "time_costs": {
                "diffusion_time_cost": diffusion_time,
                "diffusion_per_step_time_cost": diffusion_time / max(num_steps, 1),
            },
        }
