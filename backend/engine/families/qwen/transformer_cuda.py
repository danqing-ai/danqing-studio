"""Qwen-Image DiT — PyTorch / CUDA via diffusers ``QwenImageTransformer2DModel``."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from backend.engine.common.model.base import TransformerBase
from backend.engine.config.model_configs import QwenImageConfig


def _to_torch(x: Any, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(x, device=device, dtype=dtype)


def _scalar_float(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().reshape(-1)[0].item())
    if hasattr(value, "item"):
        try:
            return float(value.item())
        except Exception:
            pass
    import numpy as np

    return float(np.asarray(value, dtype=np.float64).reshape(-1)[0])


class QwenImageDiTCuda(TransformerBase):
    """Pipeline 入口：NCHW latent ↔ diffusers Qwen DiT（权重自 bundle ``transformer/`` 加载）。"""

    def __init__(self, config: QwenImageConfig, ctx: Any):
        super().__init__()
        self.config = config
        self.ctx = ctx
        self._device = torch.device(getattr(ctx, "_device", "cuda"))
        self._model: Any = None
        self._bundle_root: Path | None = None
        self._param_map: dict[str, Any] = {}

    def _ensure_model(self) -> Any:
        if self._model is not None:
            return self._model
        if self._bundle_root is None:
            raise RuntimeError(
                "Qwen-Image CUDA transformer is not loaded; "
                "call after_load_weights(bundle_root=...) before denoise."
            )
        te_path = self._bundle_root / "transformer"
        if not te_path.is_dir():
            raise RuntimeError(f"Qwen-Image bundle missing transformer directory: {te_path}")
        from diffusers import QwenImageTransformer2DModel

        self._model = QwenImageTransformer2DModel.from_pretrained(
            str(te_path),
            torch_dtype=torch.bfloat16,
        ).to(self._device)
        self._model.eval()
        self._build_param_map()
        return self._model

    def _build_param_map(self) -> None:
        self._param_map = {k: v for k, v in self._ensure_model().named_parameters()}

    def parameters(self):
        if not self._param_map:
            self._build_param_map()
        return list(self._param_map.items())

    def load_weights(
        self,
        weights: list[tuple[str, Any]],
        strict: bool = False,
        ctx: Any = None,
        *,
        bundle_affine_bits: int | None = None,
    ):
        del weights, strict, ctx, bundle_affine_bits
        # Weights load from HF bundle in ``after_load_weights`` (Pipeline passes mlx-remapped keys).

    def after_load_weights(self, bundle_root: str | None = None) -> None:
        if bundle_root is None:
            raise RuntimeError("Qwen-Image CUDA transformer requires bundle_root in after_load_weights")
        self._bundle_root = Path(bundle_root)
        self._model = None
        self._ensure_model()

    def forward(
        self,
        latents: Any,
        timestep: Any,
        txt_embeds: Any = None,
        sigmas: Any = None,
        timestep_embed_value: Any = None,
        encoder_hidden_states_mask: Any = None,
        scheduler_timesteps: Any = None,
        image_height: int | None = None,
        image_width: int | None = None,
        **conditioning: Any,
    ) -> torch.Tensor:
        del timestep_embed_value
        if txt_embeds is None:
            raise RuntimeError("Qwen Image requires txt_embeds.")
        if encoder_hidden_states_mask is None:
            raise RuntimeError("Qwen Image requires encoder_hidden_states_mask.")
        if image_height is None or image_width is None:
            raise RuntimeError("Qwen Image requires image_height / image_width.")

        model = self._ensure_model()
        device = self._device
        dtype = torch.bfloat16

        lat = _to_torch(latents, device=device, dtype=dtype)
        b, c, h_lat, w_lat = lat.shape
        edit_cond = conditioning.get("edit_conditioning_latents")
        cond_image_grid = conditioning.get("edit_cond_image_grid")
        target_seq_len: int | None = None

        if edit_cond is not None:
            cond = _to_torch(edit_cond, device=device, dtype=dtype)
            noise = lat.permute(0, 2, 3, 1).reshape(b, h_lat * w_lat, c)
            cond_packed = cond.permute(0, 2, 3, 1).reshape(b, cond.shape[2] * cond.shape[3], c)
            target_seq_len = int(noise.shape[1])
            hidden = torch.cat([noise, cond_packed], dim=1)
        else:
            hidden = lat.permute(0, 2, 3, 1).reshape(b, h_lat * w_lat, c)

        enc = _to_torch(txt_embeds, device=device, dtype=dtype)
        mask = _to_torch(encoder_hidden_states_mask, device=device, dtype=torch.float32)
        if mask.dtype != torch.bool:
            mask = mask >= 0.5

        t_val = _resolve_timestep_value(
            timestep,
            sigmas=sigmas,
            scheduler_timesteps=scheduler_timesteps,
        )
        t_tensor = torch.full((b,), int(round(t_val * 1000)), device=device, dtype=torch.long)

        latent_h = int(image_height) // 16
        latent_w = int(image_width) // 16
        img_shapes = [(1, latent_h, latent_w)]
        if cond_image_grid is not None:
            if isinstance(cond_image_grid, (list, tuple)) and len(cond_image_grid) == 3:
                img_shapes.append(tuple(int(x) for x in cond_image_grid))
            else:
                raise RuntimeError(f"Qwen edit cond_image_grid must be (f,h,w); got {cond_image_grid!r}")

        out = model(
            hidden_states=hidden,
            encoder_hidden_states=enc,
            encoder_hidden_states_mask=mask,
            timestep=t_tensor,
            img_shapes=img_shapes,
            return_dict=False,
        )
        sample = out[0] if isinstance(out, tuple) else out
        if hasattr(sample, "sample"):
            sample = sample.sample

        if target_seq_len is not None:
            sample = sample[:, :target_seq_len, :]

        seq_len = sample.shape[1]
        expected_seq_len = latent_h * latent_w
        if seq_len != expected_seq_len:
            raise RuntimeError(
                f"Qwen CUDA DiT output seq_len={seq_len} does not match latent grid "
                f"{latent_w}x{latent_h} ({expected_seq_len}) for {image_width}x{image_height}."
            )
        out_nhwc = sample.reshape(b, latent_h, latent_w, sample.shape[-1])
        return out_nhwc.permute(0, 3, 1, 2).to(dtype=lat.dtype)


def _resolve_timestep_value(
    timestep: Any,
    *,
    sigmas: Any,
    scheduler_timesteps: Any,
) -> float:
    if isinstance(timestep, (float, int)) and not isinstance(timestep, bool):
        t_int = int(timestep)
        if sigmas is not None:
            try:
                sig_list = list(sigmas)
                if 0 <= t_int < len(sig_list):
                    return _scalar_float(sig_list[t_int])
            except TypeError:
                pass
        if scheduler_timesteps is not None:
            try:
                ts_list = list(scheduler_timesteps)
                for idx, ts in enumerate(ts_list):
                    if abs(int(_scalar_float(ts)) - t_int) < 1 and sigmas is not None:
                        sig_list = list(sigmas)
                        if idx < len(sig_list):
                            return _scalar_float(sig_list[idx])
            except TypeError:
                pass
        return float(t_int) / 1000.0
    return _scalar_float(timestep)
