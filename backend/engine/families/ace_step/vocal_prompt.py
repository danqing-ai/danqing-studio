"""
ACE-Step vocal-type prompt snippets (caption conditioning, not model tensors).

Maps UI/CLI ``vocal_type`` (male / female / chorus / duet) to natural-language
fragments appended to the user description before LM expansion and DiT encoding.
"""
from __future__ import annotations

import re
from typing import Optional, Tuple

# Canonical ids (also accepted: localized aliases below).
VOCAL_TYPES = frozenset({"male", "female", "chorus", "duet"})

_ALIASES: dict[str, str] = {
    "male": "male",
    "m": "male",
    "man": "male",
    "男": "male",
    "男声": "male",
    "male vocal": "male",
    "female": "female",
    "f": "female",
    "woman": "female",
    "女": "female",
    "女声": "female",
    "female vocal": "female",
    "chorus": "chorus",
    "choir": "chorus",
    "合唱": "chorus",
    "chorus vocals": "chorus",
    "duet": "duet",
    "对唱": "duet",
    "duo": "duet",
    "male-female": "duet",
    "male female": "duet",
}

# Caption fragments — language-aware; English works well with 5Hz LM; Chinese helps zh lyrics.
_SNIPPETS: dict[str, dict[str, str]] = {
    "male": {
        "en": (
            "Clear male lead vocal, single male singer, warm masculine voice "
            "prominent in the foreground, singing the provided lyrics."
        ),
        "zh": (
            "清晰男声主唱，单一男歌手，温暖有力的男声位于前景，"
            "按所给歌词演唱。"
        ),
    },
    "female": {
        "en": (
            "Clear female lead vocal, single female singer, bright feminine voice "
            "prominent in the foreground, singing the provided lyrics."
        ),
        "zh": (
            "清晰女声主唱，单一女歌手，明亮女声位于前景，"
            "按所给歌词演唱。"
        ),
    },
    "chorus": {
        "en": (
            "Layered choir and group vocals with harmonies, multiple voices in chorus "
            "sections, wide vocal stack, vocals forward in the mix."
        ),
        "zh": (
            "多层次合唱与群唱和声，副歌段落多人声叠唱，"
            "人声宽厚并位于前景。"
        ),
    },
    "duet": {
        "en": (
            "Male and female duet with two distinct singers, alternating lead vocals "
            "trading phrases, clear separation between voices."
        ),
        "zh": (
            "男女对唱，两位歌手音色分明，主唱句交替轮唱，"
            "人声清晰可分。"
        ),
    },
}

_VOCAL_KEYWORDS_RE = re.compile(
    r"(?i)(male|female|masculine|feminine|tenor|soprano|baritone|alto|"
    r"choir|chorus|group vocal|duet|duo|vocal|singer|voice|"
    r"男声|女声|合唱|对唱|主唱|演唱|人声)"
)


def normalize_vocal_type(raw: str) -> str:
    """Return canonical id or empty string when unset / auto."""
    key = (raw or "").strip().lower()
    if not key or key in ("auto", "none", "default", "unspecified"):
        return ""
    if key in VOCAL_TYPES:
        return key
    return _ALIASES.get(key, "")


def vocal_type_snippet(vocal_type: str, language: str = "en") -> str:
    """Return the caption fragment for a canonical vocal type."""
    canon = normalize_vocal_type(vocal_type)
    if not canon:
        return ""
    lang = (language or "en").strip().lower()
    block = _SNIPPETS.get(canon, {})
    if lang.startswith("zh"):
        return block.get("zh") or block.get("en", "")
    return block.get("en") or block.get("zh", "")


def prompt_already_specifies_vocals(prompt: str) -> bool:
    return bool(_VOCAL_KEYWORDS_RE.search(prompt or ""))


def apply_vocal_type_to_prompt(
    prompt: str,
    vocal_type: str,
    *,
    language: str = "en",
    instrumental: bool = False,
    force: bool = False,
) -> Tuple[str, Optional[str]]:
    """
    Append a vocal-type snippet to the user description.

    Returns ``(effective_prompt, log_message)``. Skips when instrumental, type unset,
    or the prompt already mentions vocals (unless ``force``).
    """
    canon = normalize_vocal_type(vocal_type)
    if not canon or instrumental:
        return (prompt or "").strip(), None

    base = (prompt or "").strip()
    snippet = vocal_type_snippet(canon, language)
    if not snippet:
        return base, None

    if not force and base and prompt_already_specifies_vocals(base):
        return base, (
            f"人声类型已选 {canon!r}，但描述中已含人声相关表述，未自动追加模板。"
        )

    if base:
        if language.startswith("zh"):
            combined = f"{base}。{snippet}"
        else:
            combined = f"{base}. {snippet}"
    else:
        combined = snippet

    log = f"已应用人声类型模板 ({canon}): {snippet[:72]}…"
    if canon == "duet":
        log += "（对唱效果不稳定，建议多试几个 seed。）"
    elif canon == "chorus":
        log += "（真多轨合唱不保证，属风格提示。）"
    return combined.strip(), log
