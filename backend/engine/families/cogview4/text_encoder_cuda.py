"""CogView4 GLM-4 text encoder — HuggingFace ``GlmModel`` (penultimate hidden state)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def _torch_device() -> Any:
    import torch

    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Glm4TextEncoderTorch:
    """HF GLM stack — ``hidden_states[-2]`` (matches diffusers ``_get_glm_embeds``)."""

    def __init__(self, model_path: str):
        import torch
        from transformers import AutoModel

        self._torch = torch
        self._device = _torch_device()
        self._model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        self._model.to(self._device)
        self._model.eval()

    def encode_numpy(self, input_ids: np.ndarray) -> np.ndarray:
        torch = self._torch
        ids = torch.from_numpy(np.asarray(input_ids, dtype=np.int64)).to(self._device)
        with torch.no_grad():
            out = self._model(ids, output_hidden_states=True)
            emb = out.hidden_states[-2]
        return np.asarray(emb.float().cpu(), dtype=np.float32)


def build_glm4_text_encoder_torch(model_path: str) -> Glm4TextEncoderTorch:
    root = Path(model_path)
    if not (root / "config.json").is_file():
        raise RuntimeError(f"CogView4 text encoder: missing config.json under {root}")
    if not any(root.glob("*.safetensors")) and not (root / "model.safetensors.index.json").is_file():
        raise RuntimeError(f"CogView4 text encoder: no weights under {root}")
    return Glm4TextEncoderTorch(str(root))


def release_glm4_torch_cache() -> None:
    try:
        import torch

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass


def cogview4_encode_cuda(encoder: Any, texts: list[str]) -> Any:
    """CUDA / PyTorch encode — called from ``text_encoder.py``, not ``*_mlx.py``."""
    import torch

    if not hasattr(encoder, "_torch_encoder"):
        encoder._torch_encoder = build_glm4_text_encoder_torch(encoder.model_path)
    np_ids = np.asarray(encoder._tokenize_glm_np(texts), dtype=np.int64)
    out = encoder._torch_encoder.encode_numpy(np_ids)
    device = getattr(encoder.ctx, "_device", "cuda")
    return torch.tensor(out, dtype=torch.bfloat16, device=device)
