"""
DiffRhythm 2 MuQ-MuLan style encoder — PyTorch-only hot path.

DiffRhythm MLX generation uses this module for text style latents (512-d). The rest of the
stack (CFM + BigVGAN) runs on MLX. Full MLX MuQ port is deferred until an mlx-lm XLM-R path
exists; do not silently fall back to a weaker encoder.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

logger = logging.getLogger(__name__)

_STYLE_LATENT_DIM = 512


class MuQStyleEncoder:
    """MuQ-MuLan text style encoder (PyTorch inference, MLX array output)."""

    def __init__(self, cache_dir: Path, mulan_repo_id: str):
        self._cache_dir = Path(cache_dir)
        self._mulan_repo_id = mulan_repo_id
        self._model: Any = None

    def load(self) -> None:
        try:
            from muq import MuQMuLan
        except ImportError as exc:
            raise RuntimeError(
                "DiffRhythm 2 style encoding requires the muq package. "
                "Install with: pip install muq"
            ) from exc

        self._cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Loading MuQ-MuLan from %s (cache=%s)", self._mulan_repo_id, self._cache_dir)
        self._model = MuQMuLan.from_pretrained(self._mulan_repo_id, cache_dir=str(self._cache_dir))
        self._model.eval()

    def encode_text(self, style_prompt: str, *, array_fn: Any) -> mx.array:
        if self._model is None:
            raise RuntimeError("MuQStyleEncoder.load() must be called first")

        import torch

        text = (style_prompt or "").strip()
        if not text:
            raise RuntimeError("DiffRhythm 2 style prompt must be non-empty")

        with torch.no_grad():
            latent = self._model(texts=[text])
        np_latent = latent.detach().cpu().float().numpy()
        if np_latent.ndim == 2:
            np_latent = np_latent[0]
        if np_latent.shape[-1] != _STYLE_LATENT_DIM:
            raise RuntimeError(
                f"MuQ-MuLan text latent dim must be {_STYLE_LATENT_DIM}, got {np_latent.shape}"
            )
        return array_fn(np_latent.astype(np.float32))


# Backward-compatible alias used by generation_mlx imports.
MuQStyleEncoderMLX = MuQStyleEncoder
