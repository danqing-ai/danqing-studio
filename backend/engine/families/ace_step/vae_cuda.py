"""
ACE-Step audio VAE — CUDA (PyTorch) thin wrapper around ``diffusers.AutoencoderOobleck``.

The ACE-Step VAE uses the Stable Audio / Oobleck autoencoder.  This wrapper
provides a unified API compatible with ``AceStepVAEMLX``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch


class AceStepVAECuda:
    """Thin wrapper around ``diffusers.AutoencoderOobleck``."""

    def __init__(self, vae_dir: str):
        from diffusers.models import AutoencoderOobleck

        vae_path = Path(vae_dir)
        if not vae_path.exists():
            vae_path = Path(vae_dir) / "vae"
        self._vae = AutoencoderOobleck.from_pretrained(str(vae_path))
        self._vae.eval()
        self._device = next(self._vae.parameters()).device

    def to(self, device: str) -> "AceStepVAECuda":
        self._vae = self._vae.to(device)
        self._device = device
        return self

    def half(self) -> "AceStepVAECuda":
        self._vae = self._vae.half()
        return self

    def float(self) -> "AceStepVAECuda":
        self._vae = self._vae.float()
        return self

    def encode(self, audio_nlc: torch.Tensor) -> torch.Tensor:
        """Encode audio waveform → latent.

        Args:
            audio_nlc: [B, L, C] waveform in NLC.

        Returns:
            latent: [B, L_latent, C_latent] in NLC.
        """
        x = audio_nlc.permute(0, 2, 1)  # NLC → NCL
        with torch.inference_mode():
            latent_dist = self._vae.encode(x.to(self._device))
            if hasattr(latent_dist, "latent_dist"):
                z = latent_dist.latent_dist.sample()
            else:
                z = latent_dist.sample()
        z = z.permute(0, 2, 1)  # NCL → NLC
        return z

    def decode(self, latents_nlc: torch.Tensor) -> torch.Tensor:
        """Decode latent → audio waveform.

        Args:
            latents_nlc: [B, L_latent, C_latent] in NLC.

        Returns:
            audio: [B, L_audio, C_audio] in NLC.
        """
        x = latents_nlc.permute(0, 2, 1)  # NLC → NCL
        with torch.inference_mode():
            audio = self._vae.decode(x.to(self._device)).sample
        audio = audio.permute(0, 2, 1)  # NCL → NLC
        return audio
