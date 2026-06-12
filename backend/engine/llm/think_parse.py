"""Parse thinking-model generations (Qwen3 ``think`` blocks, chat-template variants)."""
from __future__ import annotations

import re

_THINK_BLOCK_RE = re.compile(
    r"<(?:think|redacted_reasoning)>\s*(.*?)\s*</(?:think|redacted_reasoning)>",
    re.IGNORECASE | re.DOTALL,
)
_THINK_END_RE = re.compile(r"</(?:think|redacted_reasoning)>", re.IGNORECASE)
_THINK_START_RE = re.compile(r"<(?:think|redacted_reasoning)>\s*", re.IGNORECASE)
_IM_END_RE = re.compile(r"<\|im_end\|>\s*$")


def _strip_chat_special_tokens(text: str) -> str:
    return _IM_END_RE.sub("", (text or "").strip()).strip()


def parse_thinking_output(text: str) -> tuple[str, str]:
    """Return ``(thinking_text, answer_text)``; either part may be empty."""
    raw = (text or "").strip()
    if not raw:
        return "", ""

    end_matches = list(_THINK_END_RE.finditer(raw))
    if end_matches:
        last_end = end_matches[-1]
        answer = _strip_chat_special_tokens(raw[last_end.end() :])
        thinking_parts: list[str] = []
        for block in _THINK_BLOCK_RE.finditer(raw):
            body = (block.group(1) or "").strip()
            if body:
                thinking_parts.append(body)
        if not thinking_parts and last_end.start() > 0:
            prefix = raw[: last_end.start()]
            if _THINK_START_RE.search(prefix):
                thinking_parts.append(_strip_chat_special_tokens(_THINK_START_RE.sub("", prefix, count=1)))
        thinking = "\n\n".join(thinking_parts).strip()
        return thinking, answer

    if _THINK_START_RE.search(raw):
        thinking = _strip_chat_special_tokens(_THINK_START_RE.sub("", raw, count=1))
        return thinking, ""

    return "", _strip_chat_special_tokens(raw)


def extract_final_llm_content(text: str, *, think_enabled: bool = False) -> str:
    """Keep the post-reasoning answer; empty when generation stopped inside thinking."""
    _ = think_enabled  # reserved for callers documenting intent; parsing is tag-driven
    thinking, answer = parse_thinking_output(text)
    if answer:
        return answer
    if thinking:
        return ""
    return _strip_chat_special_tokens(text or "")
