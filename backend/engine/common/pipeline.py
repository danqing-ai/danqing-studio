"""
General denoising loop — shared by all diffusion models.

The only backend-dependent logic is CFG (unconditional + conditional dual-B forward),
everything else (scheduler stepping, progress callback, cancellation) is fully generic.
"""
from __future__ import annotations

from typing import Any, Callable, Optional

from backend.core.contracts import CancelToken

from .schedulers import Scheduler


class GenerationCancelled(Exception):
    """Generation cancelled."""
    pass


StepCallback = Callable[[int, int, Any], None]
"""Per-step callback: (step, total_steps, latents) → None"""


class DenoisingPipeline:
    """General-purpose denoising loop.

    Supports:
    - Unconditional generation (context=None)
    - Unconditional CFG (context not None but guidance_scale <= 1.0)
    - Standard CFG (dual-B forward: conditional + unconditional)
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
        """Execute the denoising loop.

        Args:
            model: Model (Flux1Transformer / LTXTransformer / ...)
            scheduler: Scheduler
            latents: Initial noise [B, C, H, W] or [B, C, T, H, W]
            timesteps: Denoising timestep sequence
            context: Conditional dictionary (txt_embeds, clip_embeds, image_embeds, ...)
            negative_context: Negative conditional dictionary (optional, for CFG)
            guidance_scale: CFG guidance strength (<=1 skips CFG)
            on_step: Per-step callback
            cancel_token: Cancel token

        Returns:
            Denoised latent
        """
        for i, t in enumerate(timesteps):
            if cancel_token is not None and cancel_token.is_cancelled():
                raise GenerationCancelled()

            # Prepare timestep tensor
            t_batch = t
            if hasattr(t, 'reshape'):
                pass  # already batched
            elif isinstance(t, (int, float)):
                pass  # scalar

            # Pass scheduler sigma sequence for models that need sigmas (e.g. Z-Image)
            model_kwargs = dict(context or {})
            if hasattr(scheduler, 'sigmas') and scheduler.sigmas is not None:
                model_kwargs['sigmas'] = scheduler.sigmas

            if guidance_scale > 1.0 and (context is not None or negative_context is not None):
                # CFG mode
                neg_ctx = negative_context or self._build_empty_context(context)
                noise_pred = self._cfg_forward(
                    model, latents, t_batch, context, neg_ctx, guidance_scale,
                    sigmas=model_kwargs.get('sigmas'),
                )
            else:
                # No CFG / unconditional
                noise_pred = model(latents, t_batch, **model_kwargs)

            latents = scheduler.step(noise_pred, t_batch, latents)

            if on_step is not None:
                on_step(i + 1, len(timesteps), latents)

        return latents

    def _cfg_forward(self, model, latents, t, pos_ctx, neg_ctx, guidance_scale, sigmas=None):
        """Dual-B CFG forward: operates on tensors via RuntimeContext."""
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

        # Split: neg | pos
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
