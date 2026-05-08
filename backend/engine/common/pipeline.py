"""
通用去噪循环 — 所有扩散模型共用。

唯一与后端相关的逻辑是 CFG（无条件 + 条件 双 B 前向），
其余（调度器步进、进展回调、取消）完全通用。
"""
from __future__ import annotations

from typing import Any, Callable, Optional

from backend.core.contracts import CancelToken

from .schedulers import Scheduler


class GenerationCancelled(Exception):
    """生成被取消。"""
    pass


StepCallback = Callable[[int, int, Any], None]
"""每步回调: (step, total_steps, latents) → None"""


class DenoisingPipeline:
    """通用去噪循环。

    支持:
    - 无条件生成 (context=None)
    - 无条件 CFG (context 不为 None 但 guidance_scale <= 1.0)
    - 标准 CFG (双 B 前向：条件 + 无条件)
    """

    def __init__(self, ctx: Any = None):
        self.ctx = ctx

    def run(
        self,
        model: Any,
        scheduler: Scheduler,
        latents: Any,
        timesteps: Any,
        context: dict | None = None,
        negative_context: dict | None = None,
        guidance_scale: float = 1.0,
        on_step: Optional[StepCallback] = None,
        cancel_token: Optional[CancelToken] = None,
    ) -> Any:
        """执行去噪循环。

        Args:
            model: 模型 (Flux1Transformer / LTXTransformer / ...)
            scheduler: 调度器
            latents: 初始噪声 [B, C, H, W] 或 [B, C, T, H, W]
            timesteps: 去噪时间步序列
            context: 条件字典 (txt_embeds, clip_embeds, image_embeds, ...)
            negative_context: 负向条件字典（可选，用于 CFG）
            guidance_scale: CFG 指导强度 (<=1 跳过 CFG)
            on_step: 每步回调
            cancel_token: 取消令牌

        Returns:
            去噪后的 latent
        """
        for i, t in enumerate(timesteps):
            if cancel_token is not None and cancel_token.is_cancelled():
                raise GenerationCancelled()

            # 准备时间步张量
            t_batch = t
            if hasattr(t, 'reshape'):
                pass  # already batched
            elif isinstance(t, (int, float)):
                pass  # scalar

            # 为需要 sigmas 的模型（如 Z-Image）传入调度器 sigma 序列
            model_kwargs = dict(context or {})
            if hasattr(scheduler, 'sigmas') and scheduler.sigmas is not None:
                model_kwargs['sigmas'] = scheduler.sigmas

            if guidance_scale > 1.0 and (context is not None or negative_context is not None):
                # CFG 模式
                neg_ctx = negative_context or self._build_empty_context(context)
                noise_pred = self._cfg_forward(
                    model, latents, t_batch, context, neg_ctx, guidance_scale,
                    sigmas=model_kwargs.get('sigmas'),
                )
            else:
                # 无 CFG / 无条件
                noise_pred = model(latents, t_batch, **model_kwargs)

            latents = scheduler.step(noise_pred, t_batch, latents)

            if on_step is not None:
                on_step(i + 1, len(timesteps), latents)

        return latents

    def _cfg_forward(self, model, latents, t, pos_ctx, neg_ctx, guidance_scale, sigmas=None):
        """双 B CFG 前向：通过 RuntimeContext 操作张量。"""
        ctx = self.ctx
        if ctx is None:
            raise RuntimeError("DenoisingPipeline._cfg_forward requires RuntimeContext (pass ctx= to constructor)")

        latents_combined = ctx.concat([latents, latents], axis=0)
        combined_ctx = {}
        for key in pos_ctx:
            pos = pos_ctx[key]
            neg = neg_ctx.get(key, pos)
            if isinstance(pos, (list, tuple)):
                combined_ctx[key] = pos + neg
            else:
                combined_ctx[key] = ctx.concat([neg, pos], axis=0)

        if sigmas is not None:
            combined_ctx['sigmas'] = sigmas
        noise_pred = model(latents_combined, t, **combined_ctx)

        # 拆分：neg | pos
        B_half = latents.shape[0]
        noise_pred_uncond = noise_pred[:B_half]
        noise_pred_text = noise_pred[B_half:]

        return noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

    @staticmethod
    def _build_empty_context(context: dict) -> dict:
        empty = {}
        for key, val in context.items():
            if isinstance(val, (list, tuple)):
                empty[key] = [v * 0.0 for v in val]
            else:
                empty[key] = val * 0.0
        return empty
