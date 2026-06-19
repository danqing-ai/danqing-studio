"""StepKwargsBuilder 实现 — Image / Video 每步 kwargs 构建。

| Builder                   | 来源                                    | 封装内容                          |
|---------------------------|-----------------------------------------|-----------------------------------|
| ``ImageStepKwargsBuilder`` | ``FamilyRuntimeContract``               | ``compose_step_kwargs`` + overrides |
| ``VideoStepKwargsBuilder`` | ``_denoise_video`` 内联 kwargs 构建     | RoPE, Wan seq_len, dual text, …   |
"""
from __future__ import annotations

from typing import Any

from backend.engine.contracts.runtime_contracts import FamilyRuntimeContract


def _resolve_timestep_embed_value(
    schedule: list[float] | None,
    offset: int,
    step_idx: int,
    fallback: float | None,
) -> float | None:
    """从 schedule 获取 timestep_embed_value，fallback 到传入值。"""
    if schedule is not None:
        idx = offset + step_idx
        if idx < len(schedule):
            return schedule[idx]
    return fallback


# ---------------------------------------------------------------------------
# ImageStepKwargsBuilder
# ---------------------------------------------------------------------------

class ImageStepKwargsBuilder:
    """Image pipeline kwargs builder — 委托给 ``FamilyRuntimeContract``。"""

    def __init__(
        self,
        *,
        runtime_contract: FamilyRuntimeContract,
        txt_embeds: Any,
        neg_embeds: Any | None,
        pooled_embeds: Any | None,
        neg_pooled_embeds: Any | None,
        extra_cond: dict[str, Any],
        guidance: float,
        sigmas: Any | None,
        timestep_embed_schedule: list[float] | None,
        timestep_offset: int = 0,
        encoder_type: str,
        width: int,
        height: int,
        sched_ts: Any,
        txt_attn_mask: Any | None,
        neg_attn_mask: Any | None,
    ) -> None:
        self._rc = runtime_contract
        self._txt_embeds = txt_embeds
        self._neg_embeds = neg_embeds
        self._pooled_embeds = pooled_embeds
        self._neg_pooled_embeds = neg_pooled_embeds
        self._extra_cond = extra_cond
        self._guidance = guidance
        self._sigmas = sigmas
        self._timestep_embed_schedule = timestep_embed_schedule
        self._timestep_offset = timestep_offset
        self._encoder_type = encoder_type
        self._width = width
        self._height = height
        self._sched_ts = sched_ts
        self._txt_attn_mask = txt_attn_mask
        self._neg_attn_mask = neg_attn_mask

    def _resolve_timestep_embed_value(self, step_idx: int, fallback: float | None) -> float | None:
        """从 schedule 获取 timestep_embed_value，fallback 到传入值。"""
        return _resolve_timestep_embed_value(
            self._timestep_embed_schedule, self._timestep_offset, step_idx, fallback
        )

    def build_cond_kwargs(
        self, t: Any, *, step_idx: int, sigmas: Any,
        timestep_embed_value: float | None,
    ) -> dict[str, Any]:
        t_embed = self._resolve_timestep_embed_value(step_idx, timestep_embed_value)
        return self._rc.compose_step_kwargs(
            txt_embeds=self._txt_embeds,
            pooled_embeds=self._pooled_embeds,
            extra_cond=self._extra_cond,
            guidance=self._guidance,
            sigmas=self._sigmas if sigmas is None else sigmas,
            timestep_embed_value=t_embed,
            encoder_type=self._encoder_type,
            image_height=self._height,
            image_width=self._width,
            scheduler_timesteps=self._sched_ts,
            txt_attn_mask=self._txt_attn_mask,
        )

    def build_uncond_kwargs(
        self, t: Any, *, step_idx: int, sigmas: Any,
        timestep_embed_value: float | None,
    ) -> dict[str, Any] | None:
        if self._neg_embeds is None:
            return None
        t_embed = self._resolve_timestep_embed_value(step_idx, timestep_embed_value)
        # 构建 cond kwargs (不含 text) + 替换 neg text
        cond_base = self._rc.compose_step_kwargs(
            txt_embeds=self._txt_embeds,
            pooled_embeds=self._pooled_embeds,
            extra_cond=self._extra_cond,
            guidance=self._guidance,
            sigmas=self._sigmas if sigmas is None else sigmas,
            timestep_embed_value=t_embed,
            encoder_type=self._encoder_type,
            image_height=self._height,
            image_width=self._width,
            scheduler_timesteps=self._sched_ts,
            txt_attn_mask=self._txt_attn_mask,
        )
        overrides = self._rc.compose_uncond_overrides(
            pooled_embeds=self._neg_pooled_embeds,
            guidance=self._guidance,
            encoder_type=self._encoder_type,
            image_height=self._height,
            image_width=self._width,
            scheduler_timesteps=self._sched_ts,
            neg_attn_mask=self._neg_attn_mask,
        )
        from backend.engine.inference.cfg_strategies import build_uncond_kwargs
        return build_uncond_kwargs(cond_base, self._neg_embeds, overrides)


# ---------------------------------------------------------------------------
# VideoStepKwargsBuilder
# ---------------------------------------------------------------------------

# _denoise_video 中不需要在 cond/uncond kwargs 之间切换的 key
_VIDEO_SKIP_KEYS = frozenset({
    "txt_attn_mask", "txt_embeds_2", "txt_attn_mask_2",
    "neg_txt_attn_mask", "neg_txt_embeds_2", "neg_txt_attn_mask_2",
    "cond_latents", "mask_concat", "i2v_mode",
    "wan_i2v", "wan_cond_latent", "wan_i2v_mask", "wan_seq_len",
    "wan_expand_timesteps", "wan_bundle_root", "wan_size",
})


class VideoStepKwargsBuilder:
    """Video pipeline kwargs builder — 内联构建 (与 ``_denoise_video`` 对齐)。"""

    def __init__(
        self,
        *,
        ctx: Any,
        model: Any,
        txt_embeds: Any,
        neg_embeds: Any | None,
        extra_cond: dict[str, Any],
        rope_kw: dict[str, Any],
        sigmas: Any | None,
        timestep_embed_schedule: list[float] | None,
        timestep_offset: int = 0,
        timesteps: Any | None = None,
        use_meanflow: bool = False,
    ) -> None:
        self._ctx = ctx
        self._model = model
        self._txt_embeds = txt_embeds
        self._neg_embeds = neg_embeds
        self._extra_cond = extra_cond
        self._rope_kw = rope_kw
        self._sigmas = sigmas
        self._timestep_embed_schedule = timestep_embed_schedule
        self._timestep_offset = timestep_offset
        self._timesteps = timesteps
        self._use_meanflow = use_meanflow

    def _resolve_timestep_embed_value(self, step_idx: int, fallback: float | None) -> float | None:
        return _resolve_timestep_embed_value(
            self._timestep_embed_schedule, self._timestep_offset, step_idx, fallback
        )

    # -- shared helpers --------------------------------------------------

    def _build_base_kwargs(
        self, t: Any, *, step_idx: int, timestep_embed_value: float | None,
        txt_embeds: Any, neg_prefix: str = "",
    ) -> dict[str, Any]:
        """Build kwargs for one branch (cond or uncond)."""
        ec = self._extra_cond
        kw: dict[str, Any] = {"txt_embeds": txt_embeds} if txt_embeds is not None else {}
        # text attention mask variants
        for pos_key, neg_key in (
            ("txt_attn_mask", "neg_txt_attn_mask"),
            ("txt_embeds_2", "neg_txt_embeds_2"),
            ("txt_attn_mask_2", "neg_txt_attn_mask_2"),
        ):
            use_key = neg_key if neg_prefix else pos_key
            val = ec.get(use_key)
            if val is not None:
                kw[pos_key] = val
        # RoPE + shared extra
        kw.update(self._rope_kw)
        for k, v in ec.items():
            if k not in _VIDEO_SKIP_KEYS:
                kw[k] = v
        if self._sigmas is not None:
            kw["sigmas"] = self._sigmas
        if timestep_embed_value is not None:
            kw["timestep_embed_value"] = timestep_embed_value
        # Wan-specific
        wan_seq = int(ec.get("wan_seq_len", 0))
        if wan_seq > 0:
            kw["seq_len"] = wan_seq
        if ec.get("wan_expand_timesteps"):
            seq_len = int(ec.get("wan_seq_len", 0))
            if seq_len > 0 and hasattr(self._model, "build_timestep_per_token"):
                kw["timestep_per_token"] = self._model.build_timestep_per_token(
                    t if getattr(t, "ndim", 0) > 0 else self._ctx.array([float(t)]),
                    seq_len,
                    ec.get("wan_i2v_mask"),
                )
        if self._use_meanflow and self._timesteps is not None:
            n = len(self._timesteps)
            if step_idx >= n - 1:
                t_r = 0.0
            else:
                t_r = float(self._timesteps[step_idx + 1])
            kw["timestep_r"] = self._ctx.array([t_r], dtype=self._ctx.float32())
        return kw

    # -- public API -------------------------------------------------------

    def build_cond_kwargs(
        self, t: Any, *, step_idx: int, sigmas: Any,
        timestep_embed_value: float | None,
    ) -> dict[str, Any]:
        t_embed = self._resolve_timestep_embed_value(step_idx, timestep_embed_value)
        return self._build_base_kwargs(
            t, step_idx=step_idx, timestep_embed_value=t_embed,
            txt_embeds=self._txt_embeds,
        )

    def build_uncond_kwargs(
        self, t: Any, *, step_idx: int, sigmas: Any,
        timestep_embed_value: float | None,
    ) -> dict[str, Any] | None:
        if self._neg_embeds is None:
            return None
        t_embed = self._resolve_timestep_embed_value(step_idx, timestep_embed_value)
        return self._build_base_kwargs(
            t, step_idx=step_idx, timestep_embed_value=t_embed,
            txt_embeds=self._neg_embeds, neg_prefix="neg_",
        )


# ---------------------------------------------------------------------------
# FixedStepKwargsBuilder — 固定 cond kwargs（SR / 单路径 forward）
# ---------------------------------------------------------------------------

class FixedStepKwargsBuilder:
    """每步返回相同的 cond kwargs；无 CFG 时 ``build_uncond_kwargs`` 为 None。"""

    def __init__(self, cond_kwargs: dict[str, Any]) -> None:
        self._cond_kwargs = dict(cond_kwargs)

    def build_cond_kwargs(
        self, t: Any, *, step_idx: int, sigmas: Any,
        timestep_embed_value: float | None,
    ) -> dict[str, Any]:
        del t, step_idx, timestep_embed_value
        kw = dict(self._cond_kwargs)
        if sigmas is not None:
            kw["sigmas"] = sigmas
        return kw

    def build_uncond_kwargs(
        self, t: Any, *, step_idx: int, sigmas: Any,
        timestep_embed_value: float | None,
    ) -> dict[str, Any] | None:
        del t, step_idx, sigmas, timestep_embed_value
        return None
