"""Flux.1 双编码器 — T5-xxl（context）+ CLIP pooled（timestep 条件）。"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from backend.engine.families.flux1.clip_encoder_mlx import Flux1CLIPEncoder
from backend.engine.families.flux1.t5_encoder_mlx import Flux1T5Encoder


def _t5_paths(bundle_root: Path) -> tuple[str, str]:
    """Flux bundle: T5 在 ``text_encoder_2`` + ``tokenizer_2``（与 ImagePipeline 约定一致）。"""
    te2 = bundle_root / "text_encoder_2"
    te1 = bundle_root / "text_encoder"
    enc_dir: Path | None = None
    tok_candidates: list[Path] = []
    if te2.is_dir() and any(te2.iterdir()):
        enc_dir = te2
        tok_candidates = [bundle_root / "tokenizer_2", te2 / "tokenizer"]
    elif te1.is_dir() and any(te1.iterdir()):
        enc_dir = te1
        tok_candidates = [bundle_root / "tokenizer", te1 / "tokenizer"]
    if enc_dir is None:
        raise RuntimeError(
            f"T5 text encoder directory missing under {bundle_root} "
            f"(expected text_encoder_2 or text_encoder)"
        )
    tok_dir = next((p for p in tok_candidates if p.is_dir()), None)
    if tok_dir is None:
        raise RuntimeError(f"T5 tokenizer directory missing under {bundle_root}")
    return str(enc_dir), str(tok_dir)


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
        root = Path(bundle_root)
        t5_dir, t5_tok = _t5_paths(root)
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
