"""Flux.1 双编码器 — T5-xxl（context）+ CLIP pooled（timestep 条件）。"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from backend.engine.common.bundle_layout import t5_encoder_bundle_paths
from backend.engine.families.flux1.flux1_clip_mlx import Flux1CLIPEncoder
from backend.engine.families.flux1.flux1_t5_mlx import Flux1T5Encoder


class Flux1TextEncoder:
    """与 diffusers ``FluxPipeline`` 对齐：T5 → ``context_embedder``，CLIP pooled → ``time_text_embed``。"""

    def __init__(
        self,
        ctx: Any,
        bundle_root: str | Path,
        *,
        max_seq_len: int = 512,
        text_dim: int = 4096,
        pooled_dim: int = 768,
    ):
        del pooled_dim
        root = Path(bundle_root)
        t5_dir, t5_tok = t5_encoder_bundle_paths(root)
        clip_dir = root / "text_encoder"
        if not clip_dir.is_dir():
            raise RuntimeError(
                f"Flux.1 bundle missing CLIP text_encoder under {clip_dir}"
            )
        self._t5 = Flux1T5Encoder(
            ctx,
            t5_dir,
            max_seq_len=max_seq_len,
            tokenizer_path=t5_tok,
        )
        clip_tok = root / "tokenizer"
        self._clip = Flux1CLIPEncoder(
            ctx,
            str(clip_dir),
            tokenizer_path=str(clip_tok) if clip_tok.is_dir() else str(clip_dir),
        )

    def encode(self, texts: list[str]) -> tuple[Any, Any]:
        """Returns ``(t5_hidden_states, clip_pooled)`` — second value is not an attention mask."""
        txt = self._t5.encode(texts)
        pooled, _hidden = self._clip.encode(texts)
        return txt, pooled
