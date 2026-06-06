"""
DiffRhythm 2 BigVGAN decoder — PyTorch / CUDA (integration pending).

Bundle artifacts: ``decoder.bin`` + ``decoder.json`` from ASLP-lab/DiffRhythm2.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class DiffRhythmVAECuda:
    """BigVGAN decoder — PyTorch / CUDA (integration pending)."""

    def __init__(self, ctx: Any, vae_dir: str):
        self._ctx = ctx
        self._vae_dir = Path(vae_dir)
        self._model: Any = None

    def _ensure_model(self):
        if self._model is not None:
            return
        try:
            import torch
        except ImportError:
            raise RuntimeError("DiffRhythm CUDA VAE requires PyTorch")
        # Placeholder: load upstream VAE when available
        logger.warning("DiffRhythm CUDA VAE: using placeholder (upstream not yet integrated)")
        self._model = None

    def encode(self, audio: Any) -> Any:
        self._ensure_model()
        raise NotImplementedError("DiffRhythm CUDA VAE encode not yet implemented")

    def encode_mean(self, audio: Any) -> Any:
        self._ensure_model()
        raise NotImplementedError("DiffRhythm CUDA VAE encode_mean not yet implemented")

    def decode(self, latents: Any) -> Any:
        self._ensure_model()
        raise NotImplementedError("DiffRhythm CUDA VAE decode not yet implemented")
