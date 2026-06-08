"""L2 推理运行时辅助 — cancel / trace span / 共享 guard。"""
from __future__ import annotations

from contextlib import nullcontext
from typing import Any


def is_cancelled(cancel_token: Any | None) -> bool:
    """Return True when ``cancel_token.is_cancelled()`` is set."""
    if cancel_token is None:
        return False
    checker = getattr(cancel_token, "is_cancelled", None)
    if not callable(checker):
        raise RuntimeError(
            "cancel_token must implement is_cancelled(); "
            f"got {type(cancel_token).__name__}"
        )
    return bool(checker())


def raise_if_cancelled(cancel_token: Any | None) -> None:
    """Raise ``asyncio.CancelledError`` when the token reports cancelled."""
    if not is_cancelled(cancel_token):
        return
    import asyncio

    raise asyncio.CancelledError()


def inference_span(exec_ctx: Any, name: str):
    """Observability span for L2 infer entry points."""
    trace = getattr(exec_ctx, "trace", None)
    if trace is None:
        return nullcontext()
    return trace.span_ctx(name, kind="paradigm")
