"""
DiffRhythm 2 conditioning — lyrics G2P tokenization (MLX path).

Lyrics parsing follows ASLP-lab/DiffRhythm2 ``g2p/g2p_generation.py`` (bundle g2p + espeak).
MuQ style encoding lives in ``mulan_mlx.py`` (MLX path via ``mulan.py``).
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, List, Tuple

import mlx.core as mx
import numpy as np

from backend.engine.families.diffrhythm.generation import STRUCT_TOKEN_IDS

logger = logging.getLogger(__name__)

_STRUCT_PATTERN = re.compile(r"^\[.*?\]$")
_VOCAB_PATH = Path(__file__).resolve().parent / "data" / "vocab.json"
_BUNDLE_G2P_ROOT: Path | None = None

_SPECIAL_MAP = [
    ("t|ɹ", "tɹ"),
    ("d|ɹ", "dɹ"),
    ("t|s", "ts"),
    ("d|z", "dz"),
    ("ɪ|ɹ", "ɪɹ"),
    ("ɐ", "ɚ"),
    ("ᵻ", "ɪ"),
    ("əl", "l"),
    ("x", "k"),
    ("ɬ", "l"),
    ("ʔ", "t"),
    ("n̩", "n"),
    ("oː|ɹ", "oːɹ"),
]


def _load_vocab() -> dict[str, int]:
    with open(_VOCAB_PATH, encoding="utf-8") as f:
        return json.load(f)["vocab"]


def set_g2p_bundle_root(bundle_root: Path | None) -> None:
    """Set bundle path used to resolve ``{bundle}/g2p`` for Chinese lyrics."""
    global _BUNDLE_G2P_ROOT
    _BUNDLE_G2P_ROOT = Path(bundle_root) if bundle_root is not None else None


def _ensure_bundle_g2p() -> bool:
    from backend.engine.families.diffrhythm.g2p import bundle_g2p_ready

    return bundle_g2p_ready(_BUNDLE_G2P_ROOT)


def _install_bundle_g2p_path() -> None:
    from backend.engine.families.diffrhythm.g2p import install_bundle_g2p_path

    install_bundle_g2p_path(_BUNDLE_G2P_ROOT)


def _phoneme_to_token_ids(phonemes: str, *, vocab: dict[str, int]) -> List[int]:
    phonemes = phonemes.split("\t")[0]
    return [vocab[p] + 1 for p in phonemes.split("|") if p in vocab]


def _special_map(phonemes: str) -> str:
    text = phonemes
    for regex, replacement in _SPECIAL_MAP:
        escaped = regex.replace("|", r"\|")
        while re.search(rf"(^|[_|]){escaped}([_|]|$)", text):
            text = re.sub(rf"(^|[_|]){escaped}([_|]|$)", rf"\1{replacement}\2", text)
    return text


def _make_text_tokenizer(language: str):
    from phonemizer.backend import EspeakBackend
    from phonemizer.separator import Separator
    from phonemizer.utils import list2str, str2list

    class _TextTokenizer:
        def __init__(self, lang: str):
            self.backend = EspeakBackend(
                lang,
                punctuation_marks=",.?!;:'…",
                preserve_punctuation=True,
                with_stress=False,
                tie=False,
                language_switch="remove-flags",
                words_mismatch="ignore",
            )
            self.separator = Separator(word="|_|", syllable="-", phone="|")

        def _normalize(self, text: str) -> str:
            text = text.replace("，", ",").replace("。", ".").replace("！", "!")
            text = text.replace("？", "?").replace("；", ";").replace("：", ":")
            text = text.replace("、", ",").replace("‘", "'").replace("’", "'")
            text = text.replace("⋯", "…").replace("···", "…").replace("・・・", "…")
            text = text.replace("...", "…")
            text = re.sub(r"[^\w\s_,\.\?!;:\'…]", "", text.strip())
            text = re.sub(r"\s*([,\.\?!;:\'…])\s*", r"\1", text)
            return re.sub(r"\s+", " ", text)

        def __call__(self, text: str, *, strip: bool = True) -> str:
            lines = [self._normalize(line) for line in str2list(text) if line.strip()]
            phonemized = self.backend.phonemize(lines, separator=self.separator, strip=strip, njobs=1)
            out = list2str(phonemized)
            out = re.sub(r"([,\.\?!;:\'…])", r"|\1|", out)
            out = re.sub(r"\|+", "|", out).rstrip("|")
            return out

    lang_map = {"en": "en-us", "zh": "cmn"}
    return _TextTokenizer(lang_map[language])


def _expand_english_abbreviations(text: str) -> str:
    abbrevs = [
        (r"\bmrs\b", "misess"),
        (r"\bmr\b", "mister"),
        (r"\bdr\b", "doctor"),
        (r"\bst\b", "saint"),
    ]
    for pattern, repl in abbrevs:
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)
    return text


def _english_to_ipa(text: str, tokenizer) -> str:
    cleaned = _expand_english_abbreviations(text.strip())
    phonemes = tokenizer(cleaned)
    if phonemes and phonemes[-1] in "p⁼ʰmftnlkxʃs`ɹaoəɛɪeɑʊŋiuɥwæjː":
        phonemes += "|_"
    return _special_map(phonemes)


def _encode_via_bundle_g2p(text: str) -> List[int]:
    """Upstream ``CNENTokenizer.encode`` — bundle ``chn_eng_g2p`` for zh/en mixed lines."""
    if not _ensure_bundle_g2p() and _BUNDLE_G2P_ROOT is not None:
        from backend.engine.families.diffrhythm.g2p import ensure_bundle_g2p

        ensure_bundle_g2p(_BUNDLE_G2P_ROOT)
    if not _ensure_bundle_g2p():
        raise RuntimeError("DiffRhythm 2 bundle g2p is not available")
    _install_bundle_g2p_path()
    from g2p.g2p_generation import chn_eng_g2p

    _, tokens = chn_eng_g2p(text)
    return [t + 1 for t in tokens]


def _encode_line_upstream(text: str, *, language: str, sentence: str) -> List[int]:
    vocab = _load_vocab()
    text = text.strip()
    if not text:
        return []

    lang = language
    if _ensure_bundle_g2p() or _BUNDLE_G2P_ROOT is not None:
        try:
            return _encode_via_bundle_g2p(text)
        except RuntimeError:
            if lang == "zh":
                raise

    if lang == "zh":
        g2p_hint = f"{_BUNDLE_G2P_ROOT}/g2p" if _BUNDLE_G2P_ROOT is not None else "model bundle/g2p"
        raise RuntimeError(
            "DiffRhythm 2 Chinese lyrics require Amphion g2p under "
            f"{g2p_hint}. Re-install diffrhythm-v2 (fp16) from the model manager "
            "(bundle_repos includes g2p). English-only lyrics work without it."
        )

    if lang not in ("en", "en-us"):
        raise RuntimeError(f"DiffRhythm 2 unsupported vocal language for G2P: {language!r}")

    tokenizer = _make_text_tokenizer("en")
    phonemes = _english_to_ipa(text, tokenizer)
    tokens = _phoneme_to_token_ids(phonemes, vocab=vocab)
    if not tokens:
        raise RuntimeError(
            f"DiffRhythm 2 G2P produced no vocab hits for line {text!r} (language={language!r}). "
            "Check espeak-ng / phonemizer."
        )
    return tokens


def _is_chinese_char(ch: str) -> bool:
    return "\u4e00" <= ch <= "\u9fff"


def _segment_text(text: str) -> List[Tuple[str, str]]:
    segments: List[Tuple[str, str]] = []
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
    lang = (language or "auto").strip().lower()
    if lang in ("detect", "automatic"):
        lang = "auto"

    if lang == "auto":
        if _ensure_bundle_g2p() or _BUNDLE_G2P_ROOT is not None:
            try:
                return _encode_via_bundle_g2p(text)
            except RuntimeError:
                pass
        tokens: List[int] = []
        for seg, seg_lang in _segment_text(text):
            if seg_lang == "other":
                continue
            tokens.extend(_encode_line_upstream(seg, language=seg_lang, sentence=text))
        if not tokens and text.strip():
            raise RuntimeError(
                f"DiffRhythm 2 G2P produced no tokens for line {text!r} (auto language)."
            )
        return tokens

    return _encode_line_upstream(text, language=lang, sentence=text)


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


def lyrics_to_mx_array(token_ids: List[int], *, array_fn: Any) -> mx.array:
    return array_fn(np.asarray(token_ids, dtype=np.int32))
