"""Wan 2.2 14B MoE dual-expert wrapper (high/low noise DiT)."""
from __future__ import annotations

from typing import Any, Callable, Literal


ExpertSide = Literal["high", "low"]


class WanMoETransformer:
    """Route denoise steps to high-noise or low-noise Wan experts.

    When ``lazy=True``, only one expert is resident at a time (swap at boundary).
    """

    def __init__(
        self,
        high: Any | None,
        low: Any | None,
        *,
        boundary_step_index: int,
        config: Any,
        lazy: bool = False,
        ctx: Any | None = None,
        load_high: Callable[[], Any] | None = None,
        load_low: Callable[[], Any] | None = None,
        release_high: Callable[[], None] | None = None,
        release_low: Callable[[], None] | None = None,
    ) -> None:
        self._high = high
        self._low = low
        self.boundary_step_index = max(0, int(boundary_step_index))
        self.config = config
        self.lazy = bool(lazy)
        self._ctx = ctx
        self._load_high = load_high
        self._load_low = load_low
        self._release_high = release_high
        self._release_low = release_low
        self._active_side: ExpertSide | None = None
        self._bundle_root: str | None = None
        if not self.lazy and (high is None or low is None):
            raise RuntimeError("Wan MoE requires both experts when lazy=False.")
        if self.lazy and (load_high is None or load_low is None):
            raise RuntimeError("Wan MoE lazy mode requires load_high and load_low.")

    @property
    def ctx(self) -> Any:
        if self._ctx is not None:
            return self._ctx
        if self._high is not None:
            return self._high.ctx
        if self._low is not None:
            return self._low.ctx
        raise RuntimeError("Wan MoE: no expert loaded yet.")

    def _release_side(self, side: ExpertSide) -> None:
        if side == "high":
            if self._high is None:
                return
            if self._release_high is not None:
                self._release_high()
            self._high = None
        else:
            if self._low is None:
                return
            if self._release_low is not None:
                self._release_low()
            self._low = None
        if self._active_side == side:
            self._active_side = None
        clear = getattr(self._ctx, "clear_cache", None)
        if callable(clear):
            clear()

    def _load_side(self, side: ExpertSide) -> Any:
        if side == "high":
            if self._high is None:
                if self._load_high is None:
                    raise RuntimeError("Wan MoE lazy mode missing load_high.")
                self._high = self._load_high()
                self._after_load_one(self._high)
            return self._high
        if self._low is None:
            if self._load_low is None:
                raise RuntimeError("Wan MoE lazy mode missing load_low.")
            self._low = self._load_low()
            self._after_load_one(self._low)
            self._sync_i2v_state()
        return self._low

    def _after_load_one(self, expert: Any) -> None:
        fn = getattr(expert, "after_load_weights", None)
        if callable(fn):
            fn(bundle_root=self._bundle_root)

    def _ensure_high(self) -> Any:
        if not self.lazy:
            return self._high
        if self._active_side == "low":
            self._release_side("low")
        self._active_side = "high"
        return self._load_side("high")

    def _ensure_expert(self, step_idx: int | None) -> Any:
        idx = 0 if step_idx is None else int(step_idx)
        want: ExpertSide = "high" if idx < self.boundary_step_index else "low"
        if not self.lazy:
            return self._high if want == "high" else self._low
        if self._active_side == want:
            return self._high if want == "high" else self._low
        other: ExpertSide = "low" if want == "high" else "high"
        self._release_side(other)
        self._active_side = want
        return self._load_side(want)

    def _sync_i2v_state(self) -> None:
        inner_h = getattr(self._high, "_inner", None)
        inner_l = getattr(self._low, "_inner", None)
        if inner_h is None or inner_l is None:
            return
        sync = getattr(inner_l, "set_i2v_state", None)
        if not callable(sync):
            return
        sync(getattr(inner_h, "_i2v_cond", None), getattr(inner_h, "_i2v_mask", None))

    def forward(self, latents: Any, timestep: Any, txt_embeds: Any | None = None, **kwargs: Any) -> Any:
        step_idx = kwargs.pop("wan_denoise_step_idx", None)
        return self._ensure_expert(step_idx).forward(latents, timestep, txt_embeds=txt_embeds, **kwargs)

    def predict_noise_cfg(
        self,
        latents: Any,
        t: Any,
        *,
        guidance: float,
        pos_kwargs: dict[str, Any] | None = None,
        neg_kwargs: dict[str, Any] | None = None,
        cfg_renorm: bool = False,
        cfg_renorm_min: float = 0.0,
    ) -> Any:
        step_idx = (pos_kwargs or {}).get("wan_denoise_step_idx", 0)
        expert = self._ensure_expert(step_idx)
        return expert.predict_noise_cfg(
            latents,
            t,
            guidance=guidance,
            pos_kwargs=pos_kwargs,
            neg_kwargs=neg_kwargs,
            cfg_renorm=cfg_renorm,
            cfg_renorm_min=cfg_renorm_min,
        )

    def prepare_conditioning(self, request: Any, bundle_root: str | None = None) -> dict[str, Any]:
        return self._ensure_high().prepare_conditioning(request, bundle_root=bundle_root)

    def before_denoise(self, latents: Any, timesteps: Any, sigmas: Any, **cond: Any) -> tuple[Any, dict[str, Any]]:
        latents, cond = self._ensure_high().before_denoise(latents, timesteps, sigmas, **cond)
        if not self.lazy and self._low is not None:
            self._sync_i2v_state()
        return latents, cond

    def after_load_weights(self, bundle_root: str | None = None) -> None:
        self._bundle_root = bundle_root
        if not self.lazy:
            for expert in (self._high, self._low):
                fn = getattr(expert, "after_load_weights", None)
                if callable(fn):
                    fn(bundle_root=bundle_root)

    def release_experts(self) -> None:
        """Drop both experts from memory (lazy MoE peak reduction before VAE decode)."""
        self._release_side("high")
        self._release_side("low")

    def build_timestep_per_token(self, scalar_t: Any, seq_len: int, mask2: Any | None = None) -> Any:
        return self._ensure_high().build_timestep_per_token(scalar_t, seq_len, mask2=mask2)

    def reblend_i2v_latents(self, latents: Any) -> Any:
        high = self._high if self._high is not None else self._ensure_high()
        inner = getattr(high, "_inner", None)
        fn = getattr(inner, "reblend_i2v_latents", None) if inner is not None else None
        if callable(fn):
            return fn(latents)
        return latents

    def step_callback(self, step_idx: int, latents: Any, noise_pred: Any) -> None:
        del step_idx, latents, noise_pred

    def parameters(self):
        for side in ("high", "low"):
            expert = self._high if side == "high" else self._low
            if expert is None:
                continue
            prefix = f"{side}."
            for name, value in expert.parameters():
                yield f"{prefix}{name}", value


def release_wan_moe_experts_if_supported(model: Any, ctx: Any | None = None) -> None:
    fn = getattr(model, "release_experts", None)
    if not callable(fn):
        return
    fn()
    runtime = ctx if ctx is not None else getattr(model, "ctx", None)
    clear = getattr(runtime, "clear_cache", None)
    if callable(clear):
        clear()
