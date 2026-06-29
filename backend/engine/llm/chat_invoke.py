"""Shared helpers for internal LLM/VLM calls via OpenAI-style messages."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from backend.core.contracts import ChatMessage


def build_text_messages(*, system: str, user: str) -> list[ChatMessage]:
    """Standard two-turn text chat: markdown system + business data in user."""
    return [
        ChatMessage(role="system", content=system.strip()),
        ChatMessage(role="user", content=user.strip()),
    ]


class TextChatFn(Protocol):
    def __call__(self, *, messages: list[ChatMessage], max_tokens: int) -> str: ...


def invoke_text_chat(
    chat_fn: TextChatFn,
    *,
    system: str,
    user: str,
    max_tokens: int,
    think_apply: Callable[[str], str] | None = None,
) -> str:
    apply = think_apply or (lambda text: text)
    return chat_fn(
        messages=build_text_messages(system=system, user=apply(user)),
        max_tokens=max_tokens,
    )
