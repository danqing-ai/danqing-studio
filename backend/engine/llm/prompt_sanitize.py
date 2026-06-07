"""Post-process LLM enhanced prompts — drop repetition loops and cap length."""
from __future__ import annotations

import re

_SEGMENT_SPLIT = re.compile(r"[，,;；]\s*")
_MAX_PROMPT_CHARS = 480
_MAX_SEGMENTS = 24


def _split_segments(text: str) -> list[str]:
    return [part.strip() for part in _SEGMENT_SPLIT.split(text) if part.strip()]


def _looks_chinese(text: str) -> bool:
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    return cjk >= max(1, len(text) // 8)


def _join_segments(segments: list[str], *, prefer_chinese: bool) -> str:
    sep = "，" if prefer_chinese else ", "
    return sep.join(segments)


def _truncate_segment_loop(text: str) -> str:
    """Cut comma-separated prompts when the same segment repeats 3+ times."""
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

    if not out:
        return text.strip()
    return _join_segments(out, prefer_chinese=_looks_chinese(text))


def _truncate_tail_phrase_loop(text: str) -> str:
    """Cut when the same short suffix repeats 3+ times without delimiters."""
    trimmed = text.rstrip()
    length = len(trimmed)
    max_phrase = min(40, length // 3)
    for phrase_len in range(max_phrase, 1, -1):
        phrase = trimmed[-phrase_len:]
        repeats = 0
        pos = length
        while pos >= phrase_len and trimmed[pos - phrase_len : pos] == phrase:
            repeats += 1
            pos -= phrase_len
        if repeats >= 3:
            return trimmed[: pos + phrase_len].rstrip("，,;； ")
    return trimmed


def _strip_wrapping_quotes(text: str) -> str:
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        return text[1:-1].strip()
    return text


def prompt_enhance_quality_ok(text: str) -> bool:
    """Heuristic guard against degenerate mlx-lm prompt loops."""
    cleaned = (text or "").strip()
    if not cleaned:
        return False
    if len(cleaned) > _MAX_PROMPT_CHARS:
        return False

    segments = _split_segments(cleaned)
    if len(segments) >= 4:
        unique_ratio = len(set(segments)) / len(segments)
        if unique_ratio < 0.45:
            return False

    for idx in range(len(segments) - 1):
        if segments[idx] == segments[idx + 1]:
            return False

    return True


def sanitize_enhanced_prompt(text: str) -> str:
    """Trim enhanced prompts after mlx-lm generation."""
    raw = _strip_wrapping_quotes((text or "").strip())
    if not raw:
        return raw

    cleaned = _truncate_segment_loop(raw)
    cleaned = _truncate_tail_phrase_loop(cleaned)
    if len(cleaned) > _MAX_PROMPT_CHARS:
        cleaned = cleaned[:_MAX_PROMPT_CHARS].rstrip("，,;； ")
    return cleaned.strip()
