"""
DiffRhythm 2 conditioning — lyrics G2P tokenization + MuQ-MuLan text style (MLX path).

Lyrics parsing follows ASLP-lab/DiffRhythm2 ``inference.py``.
MuQ text encoding uses PyTorch inside this ``*_mlx.py`` module only; outputs are MLX arrays.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, List

import mlx.core as mx
import numpy as np

from backend.engine.families.diffrhythm.generation import STRUCT_TOKEN_IDS

logger = logging.getLogger(__name__)

_STRUCT_PATTERN = re.compile(r"^\[.*?\]$")
_VOCAB_PATH = Path(__file__).resolve().parent / "data" / "vocab.json"


def _load_vocab() -> dict[str, int]:
    with open(_VOCAB_PATH, encoding="utf-8") as f:
        return json.load(f)["vocab"]


def _phonemize_text(text: str, *, language: str) -> List[str]:
    try:
        from phonemizer import phonemize
    except ImportError as exc:
        raise RuntimeError(
            "DiffRhythm 2 lyrics require phonemizer and espeak-ng "
            "(pip install phonemizer; brew install espeak-ng)"
        ) from exc

    ph = phonemize(
        text,
        language=language,
        backend="espeak",
        strip=True,
        preserve_punctuation=True,
        with_stress=False,
        njobs=1,
    )
    phones: List[str] = []
    for part in ph.replace("\n", " ").split("|"):
        part = part.strip()
        if part:
            phones.extend(part.split())
    return phones


def _is_chinese_char(ch: str) -> bool:
    return "\u4e00" <= ch <= "\u9fff"


def _segment_text(text: str) -> List[tuple[str, str]]:
    segments: List[tuple[str, str]] = []
    buf = ""
    lang = ""
    for ch in text:
        if _is_chinese_char(ch):
            ch_lang = "zh"
        elif ch.isalpha():
            ch_lang = "en"
        else:
            ch_lang = "other"
        if not buf:
            buf, lang = ch, ch_lang
        elif lang == "other" or ch_lang == lang or ch_lang == "other":
            buf += ch
            if lang == "other" and ch_lang != "other":
                lang = ch_lang
        else:
            segments.append((buf, lang))
            buf, lang = ch, ch_lang
    if buf:
        segments.append((buf, lang))
    return segments


def _encode_line(text: str, *, language: str) -> List[int]:
    vocab = _load_vocab()
    text = text.strip()
    if not text:
        return []

    lang = (language or "auto").strip().lower()
    phones: List[str] = []
    if lang == "auto":
        for seg, seg_lang in _segment_text(text):
            if seg_lang == "other":
                continue
            phones.extend(_encode_line_phones(seg, language=seg_lang))
    else:
        phones = _encode_line_phones(text, language=lang)

    tokens = [vocab[p] + 1 for p in phones if p in vocab]
    if not tokens and text:
        raise RuntimeError(
            f"DiffRhythm 2 G2P produced no vocab hits for line {text!r} (language={language!r}). "
            "Check espeak-ng / phonemizer or simplify lyrics."
        )
    return tokens


def _encode_line_phones(text: str, *, language: str) -> List[str]:
    if language == "zh":
        return _phonemize_text(text, language="cmn")
    if language in ("en", "en-us"):
        return _phonemize_text(text, language="en-us")
    raise RuntimeError(f"DiffRhythm 2 unsupported vocal language for G2P: {language!r}")


def parse_lyrics_to_token_ids(lyrics: str, *, vocal_language: str) -> List[int]:
    """Parse lyrics with structure tags into flat token id list (upstream ``parse_lyrics``)."""
    lang = (vocal_language or "auto").strip().lower()
    if lang in ("detect", "automatic"):
        lang = "auto"

    lyrics_with_time: List[List[int]] = []
    got_start = False
    for raw_line in lyrics.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if _STRUCT_PATTERN.match(line):
            struct_idx = STRUCT_TOKEN_IDS.get(line.lower())
            if struct_idx is None:
                continue
            if struct_idx == STRUCT_TOKEN_IDS["[start]"]:
                got_start = True
            lyrics_with_time.append([struct_idx, STRUCT_TOKEN_IDS["[stop]"]])
            continue

        line_lang = lang
        if line_lang == "auto":
            cjk = sum(1 for c in line if _is_chinese_char(c))
            latin = sum(1 for c in line if c.isalpha() and ord(c) < 128)
            line_lang = "zh" if cjk >= latin else "en"

        tokens = _encode_line(line, language=line_lang)
        tokens = tokens + [STRUCT_TOKEN_IDS["[stop]"]]
        lyrics_with_time.append(tokens)

    if lyrics_with_time and not got_start:
        lyrics_with_time = [[STRUCT_TOKEN_IDS["[start]"], STRUCT_TOKEN_IDS["[stop]"]]] + lyrics_with_time

    flat: List[int] = []
    for chunk in lyrics_with_time:
        flat.extend(chunk)
    if not flat:
        raise RuntimeError("DiffRhythm 2 lyrics parsing produced an empty token sequence")
    return flat


class MuQStyleEncoderMLX:
    """MuQ-MuLan text style encoder (PyTorch load, MLX output)."""

    def __init__(self, cache_dir: Path, mulan_repo_id: str):
        self._cache_dir = Path(cache_dir)
        self._mulan_repo_id = mulan_repo_id
        self._model: Any = None

    def load(self) -> None:
        try:
            from muq import MuQMuLan
        except ImportError as exc:
            raise RuntimeError(
                "DiffRhythm 2 style encoding requires the muq package (pip install muq)"
            ) from exc

        self._cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Loading MuQ-MuLan from %s (cache=%s)", self._mulan_repo_id, self._cache_dir)
        self._model = MuQMuLan.from_pretrained(self._mulan_repo_id, cache_dir=str(self._cache_dir))
        self._model.eval()

    def encode_text(self, style_prompt: str, *, array_fn: Any) -> mx.array:
        if self._model is None:
            raise RuntimeError("MuQStyleEncoderMLX.load() must be called first")

        import torch

        text = (style_prompt or "").strip()
        if not text:
            raise RuntimeError("DiffRhythm 2 style prompt must be non-empty")

        with torch.no_grad():
            latent = self._model(texts=[text])
        np_latent = latent.detach().cpu().float().numpy()
        if np_latent.ndim == 2:
            np_latent = np_latent[0]
        return array_fn(np_latent.astype(np.float32))


def lyrics_to_mx_array(token_ids: List[int], *, array_fn: Any) -> mx.array:
    return array_fn(np.asarray(token_ids, dtype=np.int32))
