"""Post-process LLM lyrics to drop repetition loops and cap length."""
from __future__ import annotations

import re

_SECTION_TAG = re.compile(r"^\[[^\]]+\]\s*$", re.IGNORECASE)
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


def sanitize_lyrics_output(text: str) -> str:
    """Trim ACE-Step lyrics after mlx-lm generation (repetition / runaway length)."""
    raw = (text or "").strip()
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

        cleaned = _truncate_word_loop(stripped)
        if not cleaned or _line_is_degenerate(cleaned):
            break

        out.append(cleaned)
        if len(cleaned.split()) < len(stripped.split()):
            break
        if outro_tag_seen:
            outro_body_lines += 1
            if outro_body_lines >= 4:
                break

    while out and out[-1] == "":
        out.pop()

    return "\n".join(out).strip()
