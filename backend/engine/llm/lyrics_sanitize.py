"""Post-process LLM lyrics to drop repetition loops and cap length."""
from __future__ import annotations

import re

from backend.engine.llm.think_parse import extract_final_llm_content

_SECTION_TAG = re.compile(r"^\[[^\]]+\]\s*$", re.IGNORECASE)
_CHARS_ANNOTATION_RE = re.compile(r"\s*\(\d+\s*chars?\)", re.IGNORECASE)
_EN_TRANSLATION_SUFFIX_RE = re.compile(r'\s*[-–—]\s*["\'].+["\']\s*$')
_MAX_LINES = 36
_MAX_LINE_CHARS = 120
_MAX_LINE_WORDS = 16


def _truncate_word_loop(line: str) -> str:
    """Cut a line when the same token repeats 3+ times in a row."""
    words = line.split()
    if len(words) < 4:
        return line.strip()
    for i in range(len(words) - 2):
        if words[i] == words[i + 1] == words[i + 2]:
            return " ".join(words[:i]).strip()
    return line.strip()


def _line_is_degenerate(line: str) -> bool:
    words = line.split()
    if len(words) < 6:
        return False
    lowered = [w.lower() for w in words]
    unique_ratio = len(set(lowered)) / len(words)
    if unique_ratio < 0.3:
        return True
    if len(line) > _MAX_LINE_CHARS:
        return True
    if len(words) > _MAX_LINE_WORDS:
        return True
    return False


def lyric_line_has_annotations(line: str) -> bool:
    stripped = (line or "").strip()
    if not stripped:
        return False
    return bool(
        _CHARS_ANNOTATION_RE.search(stripped) or _EN_TRANSLATION_SUFFIX_RE.search(stripped)
    )


def _clean_lyric_line(line: str) -> str | None:
    """Keep singable text only; drop char counts and inline translations."""
    stripped = (line or "").strip()
    if not stripped:
        return None
    cleaned = _CHARS_ANNOTATION_RE.sub("", stripped)
    cleaned = _EN_TRANSLATION_SUFFIX_RE.sub("", cleaned).strip()
    if not cleaned:
        return None
    if cleaned.startswith('"') and cleaned.endswith('"'):
        return None
    if cleaned.startswith("'") and cleaned.endswith("'"):
        return None
    if lyric_line_has_annotations(cleaned):
        return None
    return cleaned


def _strip_trailing_section_tags(lines: list[str]) -> None:
    while lines and _SECTION_TAG.match(lines[-1]):
        lines.pop()


def _strip_lyrics_preamble(text: str) -> str:
    """Keep from the first section tag onward."""
    lines = (text or "").splitlines()
    start = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("[") and stripped.endswith("]"):
            start = i
            break
    return "\n".join(lines[start:]).strip()


def sanitize_lyrics_output(text: str, *, think_enabled: bool = False) -> str:
    """Trim ACE-Step lyrics after mlx-lm generation (repetition / runaway length)."""
    raw = _strip_lyrics_preamble(extract_final_llm_content(text, think_enabled=think_enabled))
    if not raw:
        return raw

    lines = raw.splitlines()
    out: list[str] = []
    outro_tag_seen = False
    outro_body_lines = 0

    for line in lines:
        if len(out) >= _MAX_LINES:
            break

        stripped = line.strip()
        if not stripped:
            if out:
                out.append("")
            continue

        if _SECTION_TAG.match(stripped):
            tag_lower = stripped.lower()
            if outro_tag_seen:
                break
            out.append(stripped)
            if "outro" in tag_lower:
                outro_tag_seen = True
                outro_body_lines = 0
            continue

        truncated = _truncate_word_loop(stripped)
        cleaned = _clean_lyric_line(truncated)
        if not cleaned:
            continue
        if _line_is_degenerate(cleaned):
            break

        out.append(cleaned)
        if truncated != stripped:
            break
        if outro_tag_seen:
            outro_body_lines += 1
            if outro_body_lines >= 4:
                break

    while out and out[-1] == "":
        out.pop()
    _strip_trailing_section_tags(out)

    return "\n".join(out).strip()
