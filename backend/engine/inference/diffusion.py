"""DiffusionInference — 标准 N 步扩散推理策略 (Layer 2)。

同时服务 ImagePipeline 和 VideoPipeline — 4D vs 5D tensor 差异由 Model (L3) 处理。

流程:
    init noise → pack → denoise loop → (caller unpacks final)

注意: ``before_denoise`` hook 由 Pipeline (L1) 在构建 bundle 前调用，
DiffusionInference 不重复调用（避免 fill_edit / edit_util 双重触发）。

Denoise loop 每步:
    scale_model_input → build kwargs → unpack → CfgStrategy.predict_noise →
    pack → scheduler.step → MemoryGuard → step_callbacks → on_step_complete
"""
from __future__ import annotations

from typing import Any

from backend.engine.inference._protocols import (
    DenoiseStepResult,
    InferenceBundle,
)
from backend.engine.inference._runtime import is_cancelled
from backend.engine.inference.cfg_strategies import resolve_cfg_strategy
from backend.engine.inference.memory_guard import MemoryGuard


class DiffusionInference:
    """标准 N 步扩散推理 — image + video 共用。

    Pipeline (L1) 构建 ``InferenceBundle`` 后调用 ``run(bundle)`` 获取 denoised latents。
    """

    def __init__(self, ctx: Any) -> None:
        self._ctx = ctx

    def run(self, bundle: InferenceBundle) -> Any | None:
        """执行完整扩散推理，返回 denoised latents（取消时返回 None）。"""
        # ── 1. 初始 latents ──────────────────────────────────────────
        latents = self._init_latents(bundle)

        # ── 2. Pack ──────────────────────────────────────────────────
        # Create/edit paths may already pass packed tokens (B, seq, C); only pack 4D NCHW.
        if bundle.pack_fn is not None and getattr(latents, "ndim", None) != 3:
            latents = bundle.pack_fn(bundle.ctx, latents)

        # ── 3. 解析策略 ──────────────────────────────────────────────
        cfg_strategy = bundle.cfg_strategy or resolve_cfg_strategy(
            bundle.model, bundle.config, bundle.ctx,
        )
        guard = bundle.memory_guard or MemoryGuard(bundle.ctx)
        builder = bundle.step_kwargs_builder

        # ── 4. Denoise loop ──────────────────────────────────────────
        n_steps = len(bundle.timesteps)
        for i, t in enumerate(bundle.timesteps):
            # 4a. Cancel check
            if is_cancelled(bundle.cancel_token):
                return None

            # 4b. scale_model_input (部分 scheduler 需要, 如 video)
            latents_in = latents
            if bundle.scale_model_input and hasattr(bundle.scheduler, "scale_model_input"):
                latents_in = bundle.scheduler.scale_model_input(latents, t)

            # 4c. Build kwargs (builder 内部处理 timestep_embed_value)
            sigmas = bundle.sigmas
            if builder is not None:
                cond_kwargs = builder.build_cond_kwargs(
                    t, step_idx=i, sigmas=sigmas, timestep_embed_value=None,
                )
                uncond_kwargs = builder.build_uncond_kwargs(
                    t, step_idx=i, sigmas=sigmas, timestep_embed_value=None,
                )
            else:
                # Fallback: 无 builder 时使用 minimal kwargs
                cond_kwargs = {"txt_embeds": bundle.txt_embeds} if bundle.txt_embeds is not None else {}
                if sigmas is not None:
                    cond_kwargs["sigmas"] = sigmas
                uncond_kwargs = (
                    {"txt_embeds": bundle.neg_embeds}
                    if bundle.neg_embeds is not None
                    else None
                )

            # 4d. Unpack for model forward (packed denoise)
            latents_model = latents_in
            if bundle.unpack_fn is not None:
                latents_model = bundle.unpack_fn(
                    bundle.ctx, latents_in, bundle.latent_h, bundle.latent_w,
                )

            # 4e. CfgStrategy predict noise
            noise_pred = cfg_strategy.predict_noise(
                bundle.model, latents_model, t,
                cond_kwargs=cond_kwargs,
                uncond_kwargs=uncond_kwargs,
                guidance=bundle.guidance,
                ctx=bundle.ctx,
                cfg_renorm=bundle.cfg_renorm,
                cfg_renorm_min=bundle.cfg_renorm_min,
            )

            # 4f. Pack noise prediction back
            if bundle.pack_fn is not None:
                noise_pred = bundle.pack_fn(bundle.ctx, noise_pred)

            # 4g. Scheduler step
            latents = bundle.scheduler.step(noise_pred, t, latents)

            # 4h. MemoryGuard (统一 eval + clear_cache)
            cleared = guard.step(latents)

            # 4i. Step post-functions (e.g. video reblend_i2v_latents)
            for fn in bundle.step_post_fns:
                latents = fn(latents)
                if getattr(bundle.ctx, "backend", None) == "mlx":
                    bundle.ctx.eval(latents)

            # 4j. Model step callback (observation, logging)
            bundle.model.step_callback(i, latents, noise_pred)

            # 4k. Pipeline step complete (progress, preview, …)
            if bundle.on_step_complete is not None:
                bundle.on_step_complete(DenoiseStepResult(
                    step_idx=i,
                    total_steps=n_steps,
                    latents=latents,
                    noise_pred=noise_pred,
                    memory_cleared=cleared,
                ))

        # 返回 packed-form latents; 最终 unpack 由 Pipeline (L1) 处理
        return latents

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _init_latents(bundle: InferenceBundle) -> Any:
        """获取初始 latents — 来自 bundle.init_latents 或生成随机噪声。"""
        if bundle.init_latents is not None:
            latents = bundle.init_latents
            if bundle.init_noise_sigma != 1.0:
                latents = latents * bundle.init_noise_sigma
            return latents
        # 随机噪声
        ctx = bundle.ctx
        shape = bundle.latent_shape
        if bundle.seed:
            return ctx.seeded_randn(shape, bundle.seed, dtype=ctx.float32())
        return ctx.randn(shape, dtype=ctx.float32())
