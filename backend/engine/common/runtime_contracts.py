"""Runtime contract entrypoints for image-family semantics.

This module intentionally keeps behavior identical to ``ImagePipeline``'s
historical inline logic while providing a single place to review/extend family
runtime and scheduler semantics.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _registry_scalar_default(entry: Any, key: str, fallback: Any) -> Any:
    spec = (entry.parameters or {}).get(key)
    if spec is None:
        return fallback
    if isinstance(spec, dict):
        return spec.get("default", fallback)
    return spec


@dataclass(frozen=True)
class FamilyRuntimeContract:
    """Minimal runtime contract for family-specific denoise semantics."""

    family: str
    config: Any

    def denoise_latent_noise_dtype(self, ctx: Any) -> Any:
        spec = getattr(self.config, "latent_noise_dtype", None)
        if isinstance(spec, str) and spec.lower() in ("bfloat16", "bf16"):
            return ctx.bfloat16()
        return ctx.float32()

    def noise_sample_dtype(self, ctx: Any, denoise_dtype: Any) -> Any:
        # mflux aligns gaussian sampling in fp32, then casts to latent precision.
        dtype_name = str(denoise_dtype).split(".")[-1].lower()
        if self.family in ("flux2", "z_image") and dtype_name in ("bfloat16", "float16"):
            return ctx.float32()
        return denoise_dtype

    def resolve_guidance_scalar(self, guidance: float) -> float:
        if getattr(self.config, "supports_guidance", True):
            return float(guidance)
        if self.family == "flux1":
            return float(guidance)
        return 0.0

    def should_encode_negative_prompt(self, guidance: float) -> bool:
        return (
            self.family != "flux1"
            and bool(getattr(self.config, "supports_guidance", False))
            and float(guidance) > 1.0
            and not bool(getattr(self.config, "structured_prompt", False))
        )

    def base_model_kwargs(
        self,
        *,
        txt_embeds: Any,
        pooled_embeds: Any | None,
        extra_cond: dict[str, Any],
        guidance: float,
    ) -> dict[str, Any]:
        kwargs = {"txt_embeds": txt_embeds} if txt_embeds is not None else {}
        if pooled_embeds is not None:
            kwargs["pooled_embeds"] = pooled_embeds
        if self.family == "flux1":
            kwargs["guidance_scale"] = float(guidance)
        kwargs.update(extra_cond)
        return kwargs

    @staticmethod
    def _apply_qwen_image_kwargs(
        kwargs: dict[str, Any],
        *,
        encoder_type: str,
        image_height: int,
        image_width: int,
        scheduler_timesteps: Any,
        encoder_hidden_states_mask: Any | None,
    ) -> None:
        if encoder_type != "qwen_image":
            return
        kwargs["image_height"] = int(image_height)
        kwargs["image_width"] = int(image_width)
        kwargs["scheduler_timesteps"] = scheduler_timesteps
        if encoder_hidden_states_mask is not None:
            kwargs["encoder_hidden_states_mask"] = encoder_hidden_states_mask

    def compose_step_kwargs(
        self,
        *,
        txt_embeds: Any,
        pooled_embeds: Any | None,
        extra_cond: dict[str, Any],
        guidance: float,
        sigmas: Any | None,
        timestep_embed_value: float | None,
        encoder_type: str,
        image_height: int,
        image_width: int,
        scheduler_timesteps: Any,
        txt_attn_mask: Any | None,
    ) -> dict[str, Any]:
        kwargs = self.base_model_kwargs(
            txt_embeds=txt_embeds,
            pooled_embeds=pooled_embeds,
            extra_cond=extra_cond,
            guidance=guidance,
        )
        if sigmas is not None:
            kwargs["sigmas"] = sigmas
        if timestep_embed_value is not None:
            kwargs["timestep_embed_value"] = float(timestep_embed_value)
        self._apply_qwen_image_kwargs(
            kwargs,
            encoder_type=encoder_type,
            image_height=image_height,
            image_width=image_width,
            scheduler_timesteps=scheduler_timesteps,
            encoder_hidden_states_mask=txt_attn_mask,
        )
        return kwargs

    def compose_uncond_overrides(
        self,
        *,
        pooled_embeds: Any | None,
        guidance: float,
        encoder_type: str,
        image_height: int,
        image_width: int,
        scheduler_timesteps: Any,
        neg_attn_mask: Any | None,
    ) -> dict[str, Any]:
        overrides: dict[str, Any] = {}
        if pooled_embeds is not None:
            overrides["pooled_embeds"] = pooled_embeds
        if self.family == "flux1":
            overrides["guidance_scale"] = float(guidance)
        self._apply_qwen_image_kwargs(
            overrides,
            encoder_type=encoder_type,
            image_height=image_height,
            image_width=image_width,
            scheduler_timesteps=scheduler_timesteps,
            encoder_hidden_states_mask=neg_attn_mask,
        )
        return overrides

    def sample_txt2img_noise(
        self,
        ctx: Any,
        *,
        latent_shape: tuple[int, int, int, int],
        seed: int | None,
        sample_dtype: Any,
        target_dtype: Any,
    ) -> Any:
        _, channels, height, width = latent_shape
        if self.family == "z_image":
            z_shape = (channels, 1, height, width)
            z_noise = (
                ctx.seeded_randn(z_shape, seed, dtype=sample_dtype)
                if seed is not None
                else ctx.randn(z_shape, dtype=sample_dtype)
            )
            noise = ctx.squeeze(ctx.expand_dims(z_noise, axis=0), axis=2)
        else:
            noise = (
                ctx.seeded_randn(latent_shape, seed, dtype=sample_dtype)
                if seed is not None
                else ctx.randn(latent_shape, dtype=sample_dtype)
            )
        if sample_dtype != target_dtype:
            noise = noise.astype(target_dtype)
        return noise

    def sample_edit_noise(
        self,
        ctx: Any,
        *,
        encoded_shape: tuple[int, ...],
        seed: int,
        sample_dtype: Any,
        target_dtype: Any,
    ) -> Any:
        if self.family == "z_image":
            z_shape = (int(encoded_shape[1]), 1, int(encoded_shape[2]), int(encoded_shape[3]))
            z_noise = ctx.seeded_randn(z_shape, seed, dtype=sample_dtype)
            noise = ctx.squeeze(ctx.expand_dims(z_noise, axis=0), axis=2)
        else:
            noise = ctx.seeded_randn(encoded_shape, seed, dtype=sample_dtype)
        if sample_dtype != target_dtype:
            noise = noise.astype(target_dtype)
        return noise


@dataclass(frozen=True)
class SchedulerSemantics:
    scheduler_name: str
    cfg_renorm: bool
    cfg_renorm_min: float
    use_empirical_mu: bool
    requires_sigma_shift: bool
    set_timesteps_kwargs: dict[str, Any]
    sigma_schedule: str | None
    sched_extra: dict[str, Any]


class SchedulerSemanticsResolver:
    """Resolve scheduler defaults and semantic flags from registry+config."""

    def resolve(
        self,
        *,
        entry: Any,
        config: Any,
        request_scheduler: str | None,
        request_metadata: dict[str, Any] | None,
        steps: int,
        width: int,
        height: int,
        init_timestep: int | None = None,
    ) -> SchedulerSemantics:
        scheduler_registry = _registry_scalar_default(entry, "scheduler", None)
        scheduler_meta = (request_metadata or {}).get("scheduler")
        scheduler_name = request_scheduler or scheduler_meta or scheduler_registry or "flow_match_euler"

        image_seq_len = (height // 16) * (width // 16)
        sched_extra: dict[str, Any] = {}
        mu = _registry_scalar_default(entry, "scheduler_mu", None)
        if mu is not None:
            sched_extra["mu"] = float(mu)
        for key in (
            "scheduler_base_image_seq_len",
            "scheduler_max_image_seq_len",
            "scheduler_base_shift",
            "scheduler_max_shift",
        ):
            value = _registry_scalar_default(entry, key, None)
            if value is not None:
                sched_extra[key] = value

        sigma_schedule = _registry_scalar_default(entry, "scheduler_sigma_schedule", None)
        if sigma_schedule == "linspace_1_to_inv_steps":
            sched_extra["sigmas"] = np.linspace(
                1.0, 1.0 / float(steps), steps, dtype=np.float64
            ).tolist()

        req_sigma_reg = _registry_scalar_default(entry, "requires_sigma_shift", None)
        requires_sigma_shift = (
            bool(req_sigma_reg)
            if req_sigma_reg is not None
            else bool(getattr(config, "requires_sigma_shift", False))
        )
        use_emu_reg = _registry_scalar_default(entry, "use_empirical_mu", None)
        use_empirical_mu = bool(use_emu_reg) if use_emu_reg is not None else requires_sigma_shift

        cfg_renorm = bool(_registry_scalar_default(entry, "enable_cfg_renorm", False))
        cfg_renorm_min = float(_registry_scalar_default(entry, "cfg_renorm_min", 0.0))

        kwargs: dict[str, Any] = {
            "num_inference_steps": int(steps),
            "image_seq_len": image_seq_len,
            "image_width": int(width),
            "image_height": int(height),
            "use_empirical_mu": use_empirical_mu,
            "requires_sigma_shift": requires_sigma_shift,
            **sched_extra,
        }
        if init_timestep is not None:
            kwargs["init_timestep"] = int(init_timestep)

        return SchedulerSemantics(
            scheduler_name=scheduler_name,
            cfg_renorm=cfg_renorm,
            cfg_renorm_min=cfg_renorm_min,
            use_empirical_mu=use_empirical_mu,
            requires_sigma_shift=requires_sigma_shift,
            set_timesteps_kwargs=kwargs,
            sigma_schedule=sigma_schedule,
            sched_extra=sched_extra,
        )
