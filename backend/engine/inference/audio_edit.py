"""Audio edit — registry handler under flow-matching / block-AR span."""

from __future__ import annotations

from typing import Any, Callable

from backend.engine.inference._runtime import inference_span
from backend.engine.inference.audio_waveform import _audio_span_name
from backend.engine.protocols.plugin import ParadigmKind


def run_audio_edit_handler(
    *,
    handler: Callable[..., Any],
    paradigm: ParadigmKind,
    exec_ctx: Any,
    **handler_kwargs: Any,
) -> Any:
    with inference_span(exec_ctx, _audio_span_name(paradigm)):
        return handler(**handler_kwargs)
