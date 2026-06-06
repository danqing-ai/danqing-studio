"""
DiffRhythm 2 decoder — public entry; MLX / CUDA in ``vae_mlx`` / ``vae_cuda``.

DiffRhythm 2 uses a Music VAE latent (5 Hz, mel_dim=64) decoded by BigVGAN to 48 kHz.
Legacy conv VAE stubs remain until BigVGAN integration lands.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


class DiffRhythmVAE:
    """DiffRhythm VAE — dual-backend dispatcher.

    Provides encode() and decode() for latent audio compression.
    """

    def __init__(self, ctx: Any, vae_dir: str):
        self._ctx = ctx
        self._vae_dir = Path(vae_dir)
        backend = getattr(ctx, "backend", "mlx")

        if backend == "mlx":
            from .vae_mlx import DiffRhythmVAEMLX

            self._vae = DiffRhythmVAEMLX(ctx, vae_dir=str(vae_dir))
        elif backend == "cuda":
            from .vae_cuda import DiffRhythmVAECuda

            self._vae = DiffRhythmVAECuda(ctx, vae_dir=str(vae_dir))
        else:
            raise RuntimeError(f"Unsupported backend: {backend}")

        self._backend = backend

    def encode(self, audio: Any) -> Any:
        """Encode audio [B, T, C] to latents [B, L, latent_dim]."""
        return self._vae.encode(audio)

    def encode_mean(self, audio: Any) -> Any:
        """Encode audio to latents (mean, no sampling)."""
        return self._vae.encode_mean(audio)

    def decode(self, latents: Any) -> Any:
        """Decode latents [B, L, latent_dim] to audio [B, T, C]."""
        return self._vae.decode(latents)
