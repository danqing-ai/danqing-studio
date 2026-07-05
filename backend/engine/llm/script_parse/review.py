"""Shared LLM invoke with validation feedback retry."""
from __future__ import annotations

from typing import Any, Callable

from pydantic import ValidationError

from backend.engine.llm.json_output import extract_json_object
from backend.engine.llm.llm_retry import invoke_text_chat_with_feedback

ValidateFn = Callable[[str], tuple[bool, str]]
ProgressFn = Callable[[str, str], None]


def invoke_pass_with_review(
    chat_fn: Callable[..., Any],
    *,
    system: str,
    user: str,
    max_tokens: int,
    think_apply: Callable[[str], str],
    validate: ValidateFn,
    max_attempts: int = 2,
    pass_name: str = "",
    on_progress: ProgressFn | None = None,
) -> tuple[str, int]:
    """Call LLM with validator feedback between attempts."""
    attempts = max(1, int(max_attempts))
    calls = 0
    pending_user = user
    last_resp = ""

    for attempt in range(1, attempts + 1):
        if attempt > 1 and on_progress and pass_name:
            on_progress("review_retry", pass_name)

        resp, n = invoke_text_chat_with_feedback(
            chat_fn,
            system=system,
            user=pending_user,
            max_tokens=max_tokens,
            think_apply=think_apply,
            validate=validate,
            max_attempts=1,
        )
        calls += n
        last_resp = resp
        ok, feedback = validate(resp)
        if ok:
            return last_resp, calls
        if attempt < attempts and feedback.strip():
            pending_user = (
                f"{user.strip()}\n\n---\n"
                f"Previous output failed validation (attempt {attempt}). "
                f"Fix ONLY the issues below; return full corrected JSON.\n\nIssues:\n{feedback.strip()}"
            )

    return last_resp, calls


def validate_pydantic_json(
    text: str,
    schema_type: type,
    extra_validate: Callable[[Any], tuple[bool, str]] | None = None,
) -> tuple[bool, str, Any | None]:
    """Parse JSON and validate with Pydantic; return (ok, feedback, payload)."""
    try:
        data = extract_json_object(text)
        payload = schema_type.model_validate(data)
    except (ValueError, ValidationError) as exc:
        return False, str(exc), None
    if extra_validate is not None:
        ok, msg = extra_validate(payload)
        if not ok:
            return False, msg, payload
    return True, "", payload
