"""Language-agnostic prompt text overlap helpers (CJK + Latin tokens)."""
from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[\u4e00-\u9fff]{2,}|[A-Za-z]{3,}")


def prompt_token_set(text: str, *, min_len: int = 2) -> set[str]:
    parts = _TOKEN_RE.findall(text or "")
    return {p.lower() if p.isascii() else p for p in parts if len(p) >= min_len}


def prompt_token_coverage(haystack: str, needle: str) -> float:
    """Fraction of needle tokens also present in haystack (0..1)."""
    needle_tokens = prompt_token_set(needle)
    if not needle_tokens:
        return 1.0
    hay_tokens = prompt_token_set(haystack)
    if not hay_tokens:
        return 0.0
    hit = sum(1 for t in needle_tokens if t in hay_tokens)
    return hit / len(needle_tokens)
