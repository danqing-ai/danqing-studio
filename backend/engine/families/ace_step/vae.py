"""
ACE-Step VAE — common interface dispatching to MLX or CUDA backend.
"""
from __future__ import annotations

from typing import Any


class AceStepVAE:
    """Audio VAE wrapper that dispatches to the appropriate backend."""

    def __init__(self, ctx: Any, **kwargs):
        backend = getattr(ctx, "backend", "mlx")
        vae_dir = kwargs.get("vae_dir", "")
        self._backend = backend

        if backend == "mlx":
            from .vae_mlx import AceStepVAEMLX, load_vae_weights_from_pytorch
            mlx_kwargs = {k: v for k, v in kwargs.items()
                          if k in ("encoder_hidden_size", "downsampling_ratios",
                                   "channel_multiples", "decoder_channels",
                                   "decoder_input_channels", "audio_channels")}
            self._vae = AceStepVAEMLX(**mlx_kwargs)
            # Load weights from PyTorch VAE checkpoint
            if vae_dir:
                from diffusers.models import AutoencoderOobleck
                pt_vae = AutoencoderOobleck.from_pretrained(vae_dir)
                pt_vae.eval()
                load_vae_weights_from_pytorch(pt_vae, self._vae)
        elif backend == "cuda":
            from .vae_cuda import AceStepVAECuda
            self._vae = AceStepVAECuda(vae_dir)
        else:
            raise RuntimeError(f"Unsupported backend: {backend}")

    def encode(self, audio) -> Any:
        return self._vae.encode(audio)

    def decode(self, latents) -> Any:
        return self._vae.decode(latents)
