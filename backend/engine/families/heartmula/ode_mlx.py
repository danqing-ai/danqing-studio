"""HeartMuLa ODE solvers (euler only)."""
from __future__ import annotations

import mlx.core as mx

def euler_solve(
    velocity_fn: Callable[[float, mx.array, mx.array], mx.array],
    x0: mx.array,
    condition: mx.array,
    t_start: float = 0.0,
    t_end: float = 1.0,
    num_steps: int = 10,
    guidance_scale: float = 1.0,
    uncond: Optional[mx.array] = None,
) -> mx.array:
    """Euler method ODE solver for flow matching.

    Integrates the ODE: dx/dt = v(t, x, condition)
    from t_start to t_end using fixed-step Euler method.

    Args:
        velocity_fn: Velocity function v(t, x, condition) -> velocity.
        x0: Initial state of shape (batch, ...).
        condition: Conditioning signal.
        t_start: Start time (typically 0).
        t_end: End time (typically 1).
        num_steps: Number of integration steps.
        guidance_scale: Classifier-free guidance scale.
        uncond: Unconditional embedding for CFG. If provided with
            guidance_scale > 1, applies CFG.

    Returns:
        Final state x(t_end) with same shape as x0.
    """
    dt = (t_end - t_start) / num_steps
    x = x0
    t = t_start

    for _ in range(num_steps):
        t_tensor = mx.full((1,), t)

        if guidance_scale > 1.0 and uncond is not None:
            # Classifier-free guidance
            v_cond = velocity_fn(t_tensor, x, condition)
            v_uncond = velocity_fn(t_tensor, x, uncond)
            v = v_uncond + guidance_scale * (v_cond - v_uncond)
        else:
            v = velocity_fn(t_tensor, x, condition)

        # Euler step
        x = x + dt * v
        t = t + dt

    return x
