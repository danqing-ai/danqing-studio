"""Wan video Transformer — public entry wrapping MLX core."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from backend.engine.common._base import TransformerBase
from backend.engine.common.cfg_batch import TEXT_KEYS_MINIMAL, predict_noise_cfg_batched
from backend.engine.config.model_configs import WanConfig
from backend.engine.runtime._base import RuntimeContext

from .conditioning import expand_wan_cond_latent, masks_like, prepare_ti2v_i2v_latents, wan_seq_len
from .transformer_mlx import WanModelMLX


class WanTransformer(TransformerBase):
    """Wan TI2V / T2V / I2V DiT — delegates to ``WanModelMLX``."""

    def __init__(self, config: WanConfig, ctx: RuntimeContext, num_frames: int = 81):
        backend = getattr(ctx, "backend", "mlx")
        if backend == "cuda":
            from backend.engine.common.dit_cuda_unavailable import raise_cuda_dit_unavailable

            raise_cuda_dit_unavailable("Wan")
        if backend != "mlx":
            raise RuntimeError(f"Unsupported backend for Wan: {backend!r}")
        self.config = config
        self.ctx = ctx
        self._core = WanModelMLX(config, ctx, num_frames=num_frames)
        self._seq_len: int | None = None
        self._output_size: tuple[int, int] | None = None

    def forward(self, latents: Any, timestep: Any, txt_embeds: Any | None = None, **kwargs: Any) -> Any:
        if self._seq_len is None:
            _, _, t, h, w = latents.shape
            self._seq_len = wan_seq_len(int(t), int(h), int(w), self.config.patch_size)
        kwargs.setdefault("seq_len", self._seq_len)
        return self._core.forward(latents, timestep, txt_embeds=txt_embeds, **kwargs)

    def parameters(self):
        return self._core.parameters()

    def load_weights(self, weights, strict=False, ctx=None, **kw):
        return self._core.load_weights(weights, strict=strict, ctx=ctx or self.ctx, **kw)

    def after_load_weights(self, bundle_root=None) -> None:
        super().after_load_weights(bundle_root)
        self._core.after_load_weights(bundle_root)

    def _build_param_map(self):
        self._core._build_param_map()

    def prepare_conditioning(self, request: Any, bundle_root: str | None = None) -> dict[str, Any]:
        cond: dict[str, Any] = {}
        if getattr(request, "source_asset_id", None):
            cond["wan_i2v"] = True
            cond["wan_bundle_root"] = bundle_root
            size = getattr(request, "size", "480x720")
            cond["wan_size"] = size
        return cond

    def before_denoise(self, latents: Any, timesteps: Any, sigmas: Any, **cond: Any) -> tuple[Any, dict[str, Any]]:
        ctx = self.ctx
        self._core.invalidate_text_cache()
        _, _, t, h, w = latents.shape
        self._seq_len = wan_seq_len(int(t), int(h), int(w), self.config.patch_size)
        cond["wan_seq_len"] = self._seq_len

        if cond.get("wan_i2v") and cond.get("wan_cond_latent") is not None:
            z = cond["wan_cond_latent"]
            _, _, t, _, _ = latents.shape
            z = expand_wan_cond_latent(ctx, z, int(t))
            mask2 = cond.get("wan_i2v_mask")
            if mask2 is None:
                _, mask2_list = masks_like(ctx, [ctx.squeeze(latents, 0)], zero=True)
                mask2 = mask2_list[0]
            self._core.set_i2v_state(z, mask2)
            latents = prepare_ti2v_i2v_latents(ctx, latents, z, mask2)

        if getattr(self.config, "expand_timesteps", False) and cond.get("wan_i2v"):
            # Per-token timesteps are required for I2V (first-frame tokens → 0).
            # T2V uses the scalar timestep path; flat per-token schedules through
            # adaLN + mx.compile corrupt denoise output (full-frame noise).
            cond["wan_expand_timesteps"] = True
            if cond.get("wan_i2v_mask") is None:
                _, mask2_list = masks_like(ctx, [ctx.squeeze(latents, 0)], zero=True)
                cond["wan_i2v_mask"] = mask2_list[0]
        return latents, cond

    def step_callback(self, step_idx: int, latents: Any, noise_pred: Any) -> None:
        del step_idx, noise_pred

    def reblend_i2v_latents(self, latents: Any) -> Any:
        return self._core.reblend_i2v_latents(latents)

    def predict_noise_cfg(
        self,
        latents_in: Any,
        t: Any,
        *,
        guidance: float,
        pos_kwargs: dict[str, Any],
        neg_kwargs: dict[str, Any],
        cfg_renorm: bool = False,
        cfg_renorm_min: float = 0.0,
    ) -> Any:
        return predict_noise_cfg_batched(
            self.forward,
            self.ctx,
            latents_in,
            t,
            guidance=guidance,
            pos_kwargs=pos_kwargs,
            neg_kwargs=neg_kwargs,
            text_keys=TEXT_KEYS_MINIMAL,
            combine_cfg_noise=self.combine_cfg_noise,
            refine_cfg_noise=self.refine_cfg_noise,
            cfg_renorm=cfg_renorm,
            cfg_renorm_min=cfg_renorm_min,
        )

    def build_timestep_per_token(self, scalar_t: Any, seq_len: int, mask2: Any | None = None) -> Any:
        """Per-token timesteps for ``expand_timesteps`` (first-frame tokens → 0 when masked)."""
        ctx = self.ctx
        cfg = self.config
        ph, pw = int(cfg.patch_size[1]), int(cfg.patch_size[2])

        if hasattr(scalar_t, "item"):
            t_scalar = float(scalar_t.item())
        else:
            try:
                t_scalar = float(scalar_t.reshape(-1)[0])
            except Exception:
                t_scalar = float(scalar_t)
        t_arr = ctx.array(t_scalar, dtype=ctx.float32())

        if mask2 is None or not getattr(cfg, "expand_timesteps", False):
            return ctx.broadcast_to(ctx.reshape(t_arr, (1, 1)), (1, seq_len))

        m = mask2
        if m.ndim == 5:
            m = ctx.squeeze(m, 0)
        if m.ndim == 4:
            m0 = m[0]
        else:
            m0 = m

        m_sub = m0[:, ::ph, ::pw] if ph > 1 or pw > 1 else m0
        temp_ts = (m_sub * t_arr).reshape(-1)
        n_tok = int(temp_ts.shape[0])
        flat = temp_ts.reshape(-1)
        if n_tok < seq_len:
            pad = ctx.ones((seq_len - n_tok,), dtype=ctx.float32()) * t_arr
            flat = ctx.concat([flat, pad], axis=0)
        elif n_tok > seq_len:
            flat = flat[:seq_len]
        return ctx.reshape(flat, (1, seq_len))
