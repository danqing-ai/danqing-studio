"""Post-process LLM enhanced prompts — drop repetition loops and cap length."""
from __future__ import annotations

import re

from backend.engine.llm.think_parse import extract_final_llm_content

_SEGMENT_SPLIT = re.compile(r"[，,;；]\s*")
_MAX_PROMPT_CHARS = 480
_MAX_SEGMENTS = 24

_REASONING_PREFIX_RE = re.compile(
    r"^(?:okay|ok|sure|let['']s|let me|first[,]|the user|analyzing|i need to|i should|hmm|well)[,\s]",
    re.IGNORECASE,
)


def _split_segments(text: str) -> list[str]:
    return [part.strip() for part in _SEGMENT_SPLIT.split(text) if part.strip()]


def _looks_chinese(text: str) -> bool:
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    return cjk >= max(1, len(text) // 8)


def _join_segments(segments: list[str], *, prefer_chinese: bool) -> str:
    sep = "，" if prefer_chinese else ", "
    return sep.join(segments)


def _truncate_cyclic_segments(segments: list[str], *, min_repeats: int = 1) -> list[str]:
    """Drop tail when a multi-segment pattern repeats (e.g. A,B,C,A,B,C,…)."""
    n = len(segments)
    if n < 4:
        return segments
    max_period = min(16, n // 2)
    for period in range(2, max_period + 1):
        if n >= period * 2 and segments[-period:] == segments[-period * 2 : -period]:
            return segments[:-period]
        for start in range(0, n - period * (min_repeats + 1) + 1):
            chunk = segments[start : start + period]
            reps = 1
            pos = start + period
            while pos + period <= n and segments[pos : pos + period] == chunk:
                reps += 1
                pos += period
            if reps >= min_repeats + 1:
                return segments[: start + period]
    return segments


def _truncate_segment_loop(text: str) -> str:
    """Cut comma-separated prompts when segments repeat consecutively or cyclically."""
    segments = _split_segments(text)
    if len(segments) < 3:
        return text.strip()

    out: list[str] = []
    for segment in segments:
        if out and out[-1] == segment:
            break
        out.append(segment)
        if len(out) >= _MAX_SEGMENTS:
            break

    out = _truncate_cyclic_segments(out)
    if not out:
        return text.strip()
    return _join_segments(out, prefer_chinese=_looks_chinese(text))


def _count_trailing_repeats(text: str, period: int) -> tuple[int, int]:
    """Return (start_index, repeat_count) for ``text[-period:]`` tiled at the tail."""
    n = len(text)
    if period <= 0 or n < period:
        return n, 0
    phrase = text[-period:]
    count = 0
    pos = n
    while pos >= period and text[pos - period : pos] == phrase:
        count += 1
        pos -= period
    return pos, count


def _strip_consecutive_tail_repeats(text: str, *, min_repeats: int = 3) -> str:
    """Cut when the tail is the same substring repeated (prefer shortest period)."""
    trimmed = text.rstrip()
    n = len(trimmed)
    if n < min_repeats * 2:
        return trimmed
    max_period = min(24, n // min_repeats)
    for period in range(1, max_period + 1):
        pos, count = _count_trailing_repeats(trimmed, period)
        if count >= min_repeats:
            return trimmed[:pos].rstrip("，,;； ")
    return trimmed


def _truncate_tail_phrase_loop(text: str) -> str:
    """Cut when the same short suffix repeats 3+ times without delimiters."""
    return _strip_consecutive_tail_repeats(text)


def _strip_wrapping_quotes(text: str) -> str:
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        return text[1:-1].strip()
    return text


def _has_degenerate_tail_repeat(text: str, *, min_repeats: int = 3) -> bool:
    trimmed = (text or "").rstrip()
    n = len(trimmed)
    if n < min_repeats * 2:
        return False
    max_period = min(24, n // min_repeats)
    for period in range(1, max_period + 1):
        _, count = _count_trailing_repeats(trimmed, period)
        if count >= min_repeats:
            return True
    return False


def looks_like_reasoning_trace(text: str) -> bool:
    """Heuristic guard against English chain-of-thought leaking into prompts."""
    t = (text or "").strip()
    if not t or len(t) < 40:
        return False
    if _looks_chinese(t) and not _REASONING_PREFIX_RE.match(t[:80]):
        return False
    if _REASONING_PREFIX_RE.match(t[:120]):
        return True
    low = t.lower()
    markers = (
        "the user wants",
        "let's tackle",
        "prompt rewrite",
        "red flags",
        "i recall",
        "this raises",
        "the request involves",
        "analyzing the original",
        "we are writing",
        "we'll structure",
        "as per the description",
    )
    if sum(1 for marker in markers if marker in low) >= 1 and not _looks_chinese(t):
        return True
    return sum(1 for marker in markers if marker in low) >= 2


def _has_cyclic_segment_loop(segments: list[str], *, min_repeats: int = 1) -> bool:
    truncated = _truncate_cyclic_segments(segments, min_repeats=min_repeats)
    return len(truncated) < len(segments)


def prompt_enhance_fidelity_ok(original: str, enhanced: str) -> bool:
    """Reject enhancements that balloon short briefs or drop user keywords."""
    orig = (original or "").strip()
    out = (enhanced or "").strip()
    if not orig or not out:
        return bool(orig == out)
    if len(orig) > 80:
        return True

    max_len = min(_MAX_PROMPT_CHARS, max(int(len(orig) * 2.5), 120))
    if len(out) > max_len:
        return False

    for segment in _split_segments(orig):
        token = segment.strip()
        if len(token) < 2:
            continue
        if token not in out:
            return False
    return True


def prompt_enhance_quality_ok(text: str, *, original: str = "") -> bool:
    """Heuristic guard against degenerate mlx-lm prompt loops."""
    cleaned = (text or "").strip()
    if not cleaned:
        return False
    if looks_like_reasoning_trace(cleaned):
        return False
    if len(cleaned) > _MAX_PROMPT_CHARS:
        return False
    if _has_degenerate_tail_repeat(cleaned):
        return False

    segments = _split_segments(cleaned)
    if _has_cyclic_segment_loop(segments):
        return False
    if len(segments) >= 4:
        unique_ratio = len(set(segments)) / len(segments)
        if unique_ratio < 0.45:
            return False

    for idx in range(len(segments) - 1):
        if segments[idx] == segments[idx + 1]:
            return False

    for segment in segments:
        if _has_degenerate_tail_repeat(segment):
            return False
        if len(segment) <= 4 and sum(1 for s in segments if s == segment) >= 3:
            return False

    if original.strip() and not prompt_enhance_fidelity_ok(original, cleaned):
        return False

    return True


def sanitize_structured_llm_response(text: str, *, think_enabled: bool = False) -> str:
    """Return post-thinking LLM content for JSON/schema tasks (no prompt-length trimming)."""
    return extract_final_llm_content(text, think_enabled=think_enabled).strip()


def sanitize_enhanced_prompt(text: str, *, think_enabled: bool = False) -> str:
    """Trim enhanced prompts after mlx-lm generation."""
    raw = _strip_wrapping_quotes(extract_final_llm_content(text, think_enabled=think_enabled))
    if not raw:
        return raw

    cleaned = _truncate_segment_loop(raw)
    segments = _split_segments(cleaned)
    if len(segments) > 1:
        cleaned = _join_segments(
            [_strip_consecutive_tail_repeats(seg) for seg in segments],
            prefer_chinese=_looks_chinese(cleaned),
        )
    else:
        cleaned = _strip_consecutive_tail_repeats(cleaned)
    cleaned = _truncate_tail_phrase_loop(cleaned)
    if len(cleaned) > _MAX_PROMPT_CHARS:
        cleaned = cleaned[:_MAX_PROMPT_CHARS].rstrip("，,;； ")
    return cleaned.strip().rstrip("，,;； ")
