"""LLM invoke with validation feedback retry (not blind repeat)."""
from __future__ import annotations

from typing import Any, Callable

from backend.engine.llm.chat_invoke import invoke_text_chat

ValidateFn = Callable[[str], tuple[bool, str]]


def invoke_text_chat_with_feedback(
    chat_fn: Callable[..., Any],
    *,
    system: str,
    user: str,
    max_tokens: int,
    think_apply: Callable[[str], str],
    validate: ValidateFn,
    max_attempts: int = 2,
    feedback_prefix: str = "Previous output failed validation. Fix ONLY the issues below; return full corrected JSON.\n\nIssues:\n",
) -> tuple[str, int]:
    """Call chat up to *max_attempts* times, appending validator feedback between tries."""
    attempts = max(1, int(max_attempts))
    calls = 0
    last_resp = ""
    pending_user = user
    for _ in range(attempts):
        last_resp = invoke_text_chat(
            chat_fn,
            system=system,
            user=pending_user,
            max_tokens=max_tokens,
            think_apply=think_apply,
        )
        calls += 1
        ok, feedback = validate(last_resp)
        if ok:
            return last_resp, calls
        if feedback.strip():
            pending_user = f"{user.strip()}\n\n---\n{feedback_prefix}{feedback.strip()}"
    return last_resp, calls
