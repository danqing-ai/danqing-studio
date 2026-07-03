"""DiT inference runtime session — binds ``ImageInferencePlan`` to family forward paths."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from backend.engine.common.ops.lemica import lemica_compute_steps
from backend.engine.common.ops.step_cache import StepCacheSession
from backend.engine.common.ops.teacache_calibrate import teacache_probe_enabled
from backend.engine.inference.optimization_plan import (
    ImageInferencePlan,
    InferencePlan,
    VideoInferencePlan,
    plan_from_extra_cond,
    pop_inference_plan,
)


@dataclass
class DiTRuntimeSession:
    """Per-run MLX DiT runtime flags + step cache session."""

    plan: InferencePlan
    step_cache: StepCacheSession | None
    use_mlx_compile: bool
    lemica_bool_list: tuple[bool, ...] | None = None

    @classmethod
    def from_plan(cls, plan: InferencePlan) -> DiTRuntimeSession:
        probe = teacache_probe_enabled()
        if probe:
            step_cache = StepCacheSession.configure_probe(
                family=plan.family,
                num_steps=plan.num_steps,
            )
        elif plan.step_cache_enabled:
            step_cache = StepCacheSession.configure(
                family=plan.family,
                mode=plan.teacache_mode,
                num_steps=plan.num_steps,
            )
        else:
            step_cache = None
        lemica_list = None
        if isinstance(plan, ImageInferencePlan) and plan.lemica_enabled and not probe:
            lemica_list = lemica_compute_steps(plan.lemica_mode, plan.num_steps)
        return cls(
            plan=plan,
            step_cache=step_cache,
            use_mlx_compile=plan.use_mlx_compile and not probe,
            lemica_bool_list=lemica_list,
        )

    @classmethod
    def from_before_denoise_cond(
        cls,
        *,
        family: str,
        config: Any,
        entry: Any,
        ctx: Any,
        cond: dict[str, Any],
        timesteps: Any,
    ) -> tuple[DiTRuntimeSession, dict[str, Any]]:
        cond = dict(cond)
        n_steps = len(timesteps) if timesteps is not None else 0
        plan = pop_inference_plan(cond)
        if plan is None:
            plan = plan_from_extra_cond(
                cond,
                family=family,
                config=config,
                entry=entry,
                ctx=ctx,
                num_steps=n_steps,
            )
        cond.pop("teacache_thresh", None)
        return cls.from_plan(plan), cond

    def refresh_compile(self, ctx: Any, forward_fn: Callable[..., Any]) -> Callable[..., Any] | None:
        if not self.use_mlx_compile or getattr(ctx, "backend", None) != "mlx":
            return None
        try:
            return ctx.compile(forward_fn)
        except Exception:
            self.use_mlx_compile = False
            return None
